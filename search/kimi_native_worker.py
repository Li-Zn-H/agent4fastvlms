"""Kimi generation through the existing isolated native LMMS connection."""

import json
from multiprocessing.connection import Connection
from pathlib import Path
import sys

from search.internvl_native_worker import (
    NativeBackend as BaseBackend, PROJECT_ROOT, cpu_value, receive_message, send_message,
)
from skills.quantization import digest
from skills.kimi_vl_pruning import dimensions, modules, units

INPUT_KEYS = frozenset({'input_ids', 'attention_mask', 'pixel_values', 'image_grid_hws'})


def observed_topology(model):
    """Inspect actual expert geometry even after replacement by packed kernels."""
    dims = dimensions(model.config.text_config)
    widths = []
    for experts in modules(model, dims):
        width = experts[0].gate_proj.out_features
        for expert in experts:
            for name, shape in {'gate_proj': (dims['hidden_size'], width),
                                'up_proj': (dims['hidden_size'], width),
                                'down_proj': (width, dims['hidden_size'])}.items():
                linear = getattr(expert, name)
                if (linear.in_features, linear.out_features) != shape:
                    raise ValueError('Native Kimi expert dimensions disagree within a width unit')
        widths.append(width)
    routers = {name: str(value.dtype) for name, value in model.named_parameters()
               if name.endswith('mlp.gate.e_score_correction_bias')}
    required = {f"language_model.model.layers.{u['layer']}.mlp.gate.e_score_correction_bias"
                for u in units(dims) if u['kind'] == 'routed'}
    if set(routers) != required or any(v != 'torch.bfloat16' for v in routers.values()):
        raise ValueError('Native Kimi routing corrections must remain BF16')
    layers = model.language_model.model.layers
    return {'dimensions': dims, 'physical_widths': widths, 'router_dtypes': routers,
            'routed_experts': [len(layers[u['layer']].mlp.experts) for u in units(dims) if u['kind'] == 'routed'],
            'top_k': [layers[u['layer']].mlp.num_experts_per_tok for u in units(dims) if u['kind'] == 'routed'],
            'quantized_modules': getattr(model, '_agent4vlm_quantized_modules', []),
            'vision_padding': getattr(model, '_agent4vlm_vision_padding', None),
            'eos_token_id': model.generation_config.eos_token_id}


def serve(connection, config, *, loader=None):
    try:
        import torch
        from adapters.kimi_vl_dense import load_model
        from skills.kimi_vl_token import KimiTokenController

        if config.get('stack_adapter'):
            import gptqmodel  # noqa: F401 -- finish backend imports before loading weights
        model, processor = (loader or load_model)(config)
        ratio = config.get('token_keep_ratio', 1)
        controller = KimiTokenController(model, config.get('token_selection', {
            'score_dtype': 'float32', 'coverage_grid': {'rows': 4, 'columns': 4}}), ratio)
        send_message(connection, {'status': 'ready', 'model_digest': digest(config),
                                  **observed_topology(model)})
        while True:
            request = receive_message(connection)
            if request is None:
                break
            kwargs = request['kwargs']
            if not INPUT_KEYS.issubset(kwargs) or 'tokenizer' in kwargs:
                raise ValueError('Kimi worker requires complete native inputs and its own tokenizer')
            inputs = {key: value.to(model.device) for key, value in kwargs.items() if key in INPUT_KEYS}
            options = {key: value for key, value in kwargs.items() if key not in INPUT_KEYS}
            if options.get('stop_strings'):
                options['tokenizer'] = processor.tokenizer
            with torch.inference_mode():
                output = controller.generate(inputs, options)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError('Kimi worker requires one output token sequence')
            if ratio < 1 and not controller.last_metrics:
                raise RuntimeError('Kimi worker did not observe Token/KV mechanisms')
            suffix = output[:, inputs['input_ids'].shape[1]:]
            send_message(connection, {'status': 'completed', 'index': request['index'],
                                      'output_ids': cpu_value(suffix), 'metrics': controller.last_metrics})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            send_message(connection, {'status': 'error', 'error': f'{type(error).__name__}: {error}'})
        except (EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


class NativeBackend(BaseBackend):
    worker_module = 'search.kimi_native_worker'
    model_name = 'Kimi-VL'

    def validate_ready(self, ready):
        path = PROJECT_ROOT / self.config['path']
        header = json.loads((path / 'config.json').read_text())
        dims = dimensions(header['text_config'])
        domain = units(dims)
        count = sum(u['kind'] == 'routed' for u in domain)
        if (ready.get('dimensions') != dims
                or ready.get('routed_experts') != [dims['n_routed_experts']] * count
                or ready.get('top_k') != [dims['num_experts_per_tok']] * count):
            raise RuntimeError('Kimi native worker changed its expert counts or routing')
        routers = {f"language_model.model.layers.{u['layer']}.mlp.gate.e_score_correction_bias":
                   'torch.bfloat16' for u in domain if u['kind'] == 'routed'}
        if ready.get('router_dtypes') != routers or ready.get('eos_token_id') != [163585]:
            raise RuntimeError('Kimi native worker changed routing precision or termination')
        widths = header.get('a4fvlm_stack_widths', [u['original_width'] for u in domain])
        if ready.get('physical_widths') != widths:
            raise RuntimeError('Kimi native worker physical widths differ from its selected Parent')
        required = []
        if header.get('a4fvlm_kimi_gptq'):
            manifest = json.loads((path / 'manifest.json').read_text())
            required = sorted(n for n, p in manifest['module_assignment'].items() if p == 'int4_w4a16')
            if ready.get('vision_padding') != header['a4fvlm_kimi_gptq']['vision_padding']:
                raise RuntimeError('Kimi native worker changed physical MoonViT padding')
        if sorted(ready.get('quantized_modules', [])) != required:
            raise RuntimeError('Kimi native worker did not load every declared packed kernel')
        if self.config.get('stack_adapter') and (not required or header.get('a4fvlm_stack_widths') is None):
            raise RuntimeError('Kimi native Stack requires a freshly quantized physical Parent')


def main():
    fd, raw_config = sys.argv[1:]
    serve(Connection(int(fd)), json.loads(raw_config))


if __name__ == '__main__':
    main()
