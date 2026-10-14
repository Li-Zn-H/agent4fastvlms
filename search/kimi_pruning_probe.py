"""Kimi-specific width admission using shared BF16/Marlin kernel probes."""

from skills.kimi_vl_pruning import dimensions, units, validate_config
from search.deepseek_pruning_probe import KERNELS, run

def snapshot(config, model, evaluation, device):
    from search.model import PROJECT_ROOT
    from search.context import dense_root
    from search.structured_pruning import prefill_lengths, probe_widths
    from search.optimize import lines, read
    checkpoint = read(PROJECT_ROOT / model['path'] / 'config.json')
    dims = dimensions(checkpoint['text_config'])
    validate_config(config, dims)
    if model['dtype'] != 'bfloat16':
        raise ValueError('Kimi width admission requires the native BF16 activation dtype')
    protocol = config['hardware_probe']
    if (not protocol.get('worker_python') or not protocol.get('requirements')
            or not (PROJECT_ROOT / protocol['requirements']).is_file()):
        raise ValueError('Kimi kernel probes require an isolated runtime requirements file and Worker')
    metrics = dense_root(evaluation, project_root=PROJECT_ROOT) / 'benchmark/metrics.jsonl'
    lengths = prefill_lengths(lines(metrics), [s['stable_row_id'] for s in lines(PROJECT_ROOT / evaluation['data']['search'])])
    originals = {u['kind']: u['original_width'] for u in units(dims)}
    excluded = protocol.get('excluded_widths', {})
    if excluded:
        evidence = read(PROJECT_ROOT / protocol['exclusion_evidence'])
        if (evidence['context']['dimensions'] != dims or evidence['context']['device'] != device
                or evidence['context']['model_adapter'] != 'kimi_vl'):
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
            'context': {'device': device, 'backend': 'kimi_native_swiglu_and_vllm_moe_marlin',
                        'dtype': model['dtype'], 'dimensions': dims, 'original_widths': originals,
                        'model_adapter': 'kimi_vl', 'kernels': {k: list(v) for k, v in KERNELS.items()},
                        'quantization': {'bits': 4, 'group_size': 128, 'symmetric': True, 'act_order': False},
                        'input_distribution': 'seed_0_standard_normal',
                        'weights': 'seed_0_normal_scaled_by_fan_in',
                        'timing': 'cuda_events_full_swiglu_forward',
                        'validation': 'same_routes_against_torch_with_dequantized_weights',
                        'reference_atol': .02, 'reference_rtol': .02}}
