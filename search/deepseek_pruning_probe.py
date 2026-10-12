"""DeepSeek branch-width admission through actual BF16 and Marlin kernels."""

import json
import math
from pathlib import Path
import time

from skills.deepseek_vl2_pruning import dimensions, units, validate_config


KERNELS = {'dense': ('torch_bf16', 'marlin_w4a16'),
           'shared': ('torch_bf16', 'marlin_w4a16'),
           'routed': ('torch_bf16', 'triton_fused_moe_bf16', 'marlin_fused_moe_w4a16')}


def snapshot(config, model, evaluation, device):
    from search.model import PROJECT_ROOT
    from search.context import dense_root
    from search.structured_pruning import prefill_lengths, probe_widths
    from search.optimize import lines, read
    checkpoint = read(PROJECT_ROOT / model['path'] / 'config.json')
    dims = dimensions(checkpoint['language_config'])
    validate_config(config, dims)
    if model['dtype'] != 'bfloat16':
        raise ValueError('DeepSeek width admission requires the native BF16 activation dtype')
    protocol = config['hardware_probe']
    if (not protocol.get('worker_python') or not protocol.get('requirements')
            or not (PROJECT_ROOT / protocol['requirements']).is_file()):
        raise ValueError('DeepSeek kernel probes require an isolated runtime requirements file and Worker')
    metrics = dense_root(evaluation, project_root=PROJECT_ROOT) / 'benchmark/metrics.jsonl'
    lengths = prefill_lengths(lines(metrics), [s['stable_row_id'] for s in lines(PROJECT_ROOT / evaluation['data']['search'])])
    originals = {u['kind']: u['original_width'] for u in units(dims)}
    excluded = protocol.get('excluded_widths', {})
    if excluded:
        evidence = read(PROJECT_ROOT / protocol['exclusion_evidence'])
        if (evidence['context']['dimensions'] != dims or evidence['context']['device'] != device
                or evidence['context']['model_adapter'] != 'deepseek_vl2'):
            raise ValueError('Kernel exclusion evidence belongs to another topology/device')
        for kind, widths in excluded.items():
            for width in widths:
                if not any(row.get('kind') == kind and row.get('width') == width
                           and row.get('status') in ('execution_error', 'oom') for row in evidence['records']):
                    raise ValueError('Kernel exclusion has no matching measured failure')
    shapes = [{'kind': kind, 'kernel': kernel, 'phase': phase, 'input_tokens': length, 'width': width}
              for kind, original in originals.items()
              for width in probe_widths(original, config['selection']['channel_group_size'], protocol['minimum_width_ratio'])
              if width not in excluded.get(kind, [])
              for phase, length in [('decode', 1), *[('prefill', n) for n in lengths]]
              for kernel in KERNELS[kind]]
    return {'config': config, 'model': model, 'metrics_source': str(metrics), 'shapes': shapes,
            'context': {'device': device, 'backend': 'deepseek_native_swiglu_and_vllm_moe_marlin',
                        'dtype': model['dtype'], 'dimensions': dims, 'original_widths': originals,
                        'model_adapter': 'deepseek_vl2', 'kernels': {k: list(v) for k, v in KERNELS.items()},
                        'quantization': {'bits': 4, 'group_size': 128, 'symmetric': True, 'act_order': False},
                        'input_distribution': 'seed_0_standard_normal',
                        'weights': 'seed_0_normal_scaled_by_fan_in',
                        'timing': 'cuda_events_full_swiglu_forward',
                        'validation': 'same_routes_against_torch_with_dequantized_weights',
                        'reference_atol': .02, 'reference_rtol': .02}}


def _torch_forward(x, w13, w2, ids=None, gates=None):
    import torch
    from torch.nn import functional as F
    if ids is None:
        gate, up = F.linear(x, w13).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, w2)
    result = torch.zeros_like(x, dtype=torch.float32)
    order = ids.flatten().argsort()
    counts = torch.bincount(ids.flatten(), minlength=w13.shape[0]).tolist()
    sorted_tokens = x[order // ids.shape[1]]
    offset = 0
    for expert, count in enumerate(counts):
        if count:
            section = slice(offset, offset + count)
            gate, up = F.linear(sorted_tokens[section], w13[expert]).chunk(2, dim=-1)
            values = F.linear(F.silu(gate) * up, w2[expert]).float()
            values *= gates.flatten()[order[section], None]
            result.index_add_(0, order[section] // ids.shape[1], values)
            offset += count
    return result.to(x.dtype)


def _quantize(weight):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    return marlin_quantize(weight.T.contiguous(), scalar_types.uint4b8, 128, False)


def _kernel(shape, dims):
    """Return a callable, a same-weight reference and its actual tensor inventory."""
    import torch
    tokens, width, hidden = shape['input_tokens'], shape['width'], dims['hidden_size']
    routed = shape['kind'] == 'routed'
    experts = dims['n_routed_experts'] if routed else 1
    x = torch.randn(tokens, hidden, device='cuda', dtype=torch.bfloat16)
    prefix = (experts,) if routed else ()
    w13 = torch.randn(*prefix, 2 * width, hidden, device='cuda', dtype=torch.bfloat16) / math.sqrt(hidden)
    w2 = torch.randn(*prefix, hidden, width, device='cuda', dtype=torch.bfloat16) / math.sqrt(width)
    ids = gates = None
    if routed:
        ids = torch.rand(tokens, experts, device='cuda').topk(dims['num_experts_per_tok'], dim=-1).indices.to(torch.int32)
        gates = torch.softmax(torch.randn(tokens, dims['num_experts_per_tok'], device='cuda'), dim=-1)
    inventory = {'input': list(x.shape), 'gate_up': list(w13.shape), 'down': list(w2.shape),
                 'experts': experts, 'top_k': dims['num_experts_per_tok'] if routed else 1}
    kernel = shape['kernel']
    if kernel == 'torch_bf16':
        run = lambda: _torch_forward(x, w13, w2, ids, gates)
        return run, run(), inventory
    if kernel == 'triton_fused_moe_bf16':
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        reference = _torch_forward(x, w13, w2, ids, gates)
        return lambda: fused_experts(x, w13, w2, gates, ids), reference, inventory
    from vllm.scalar_type import scalar_types
    if kernel == 'marlin_w4a16':
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear, marlin_make_workspace_new
        q1, q2 = _quantize(w13), _quantize(w2)
        reference = _torch_forward(x, q1[0].T, q2[0].T)
        workspace = marlin_make_workspace_new(x.device)
        empty = torch.empty(0, device=x.device, dtype=torch.int32)

        def linear(value, q, in_features, out_features):
            return apply_gptq_marlin_linear(value, q[1], q[2], empty, q[3], q[4], workspace,
                                           scalar_types.uint4b8, out_features, in_features, True)

        def run():
            gate, up = linear(x, q1, hidden, 2 * width).chunk(2, dim=-1)
            return linear(torch.nn.functional.silu(gate) * up, q2, width, hidden)
        inventory.update(packed_gate_up=list(q1[1].shape), packed_down=list(q2[1].shape))
        return run, reference, inventory
    if kernel != 'marlin_fused_moe_w4a16':
        raise ValueError('Unknown DeepSeek admission kernel')
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    first = [_quantize(w) for w in w13]
    second = [_quantize(w) for w in w2]
    ref1 = torch.stack([q[0].T for q in first]).contiguous()
    ref2 = torch.stack([q[0].T for q in second]).contiguous()
    reference = _torch_forward(x, ref1, ref2, ids, gates)
    packed1 = torch.stack([q[1] for q in first])
    packed2 = torch.stack([q[1] for q in second])
    scales1 = torch.stack([q[2] for q in first])
    scales2 = torch.stack([q[2] for q in second])
    inventory.update(packed_gate_up=list(packed1.shape), packed_down=list(packed2.shape))
    return lambda: fused_marlin_moe(x, packed1, packed2, None, None, scales1, scales2,
                                   gates, ids, scalar_types.uint4b8.id), reference, inventory


def run(request):
    import torch
    import transformers
    import vllm
    from search.worker import save_json
    snap = request['probe']
    output = Path(request['probe_output']) / 'measurements.json'
    records = json.loads(output.read_text())['records'] if output.exists() else []
    protocol = snap['config']['hardware_probe']
    environment = {'torch': torch.__version__, 'transformers': transformers.__version__,
                   'vllm': vllm.__version__, 'cuda': torch.version.cuda, 'hardware': request['hardware']}
    failures = 0
    for shape in request['shapes_to_measure']:
        row = {**shape, **protocol, 'environment': environment}
        kernel = reference = actual = None
        try:
            torch.manual_seed(0)
            with torch.inference_mode():
                kernel, reference, inventory = _kernel(shape, snap['context']['dimensions'])
                for _ in range(protocol['warmup_iterations']):
                    actual = kernel()
                if not bool(torch.isfinite(actual).all()):
                    raise ValueError('Nonfinite kernel output')
                torch.testing.assert_close(actual, reference, atol=snap['context']['reference_atol'],
                                           rtol=snap['context']['reference_rtol'])
                relative_rms = (actual.float() - reference.float()).square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-8)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                events = []
                started = time.perf_counter()
                for _ in range(protocol['measurement_iterations']):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    actual = kernel()
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize()
                row.update(status='measured', finite_output=True, reference_checked=True,
                           reference_relative_rms=float(relative_rms), weight_shapes=inventory,
                           milliseconds=[a.elapsed_time(b) for a, b in events],
                           timed_batch_wall_seconds=time.perf_counter() - started,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
        except torch.OutOfMemoryError as error:
            row.update(status='oom', error=str(error))
        except (AssertionError, RuntimeError, ValueError, TypeError) as error:
            row.update(status='execution_error', error=f'{type(error).__name__}: {error}')
        finally:
            del kernel, reference, actual
            torch.cuda.empty_cache()
        failures += int(row['status'] != 'measured')
        records.append(row)
        save_json(output, {'context': snap['context'], 'records': records})
        print(f'{shape}: {row["status"]}', flush=True)
    return {'probe_status': 'incomplete' if failures else 'completed', 'failed_shapes': failures}
