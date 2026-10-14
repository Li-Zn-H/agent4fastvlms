"""Register Kimi's native Instruct generation for unchanged LMMS tasks."""

import json
import os
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_native_config(config):
    from adapters.kimi_vl_dense import native_path
    native_path(config)
    if (config.get('adapter') != 'kimi_vl' or config.get('dtype') != 'bfloat16'
            or config.get('attn_implementation') != 'eager'
            or config.get('execution_backend', 'native_transformers') not in {'native_transformers', 'vllm'}):
        raise ValueError('Kimi native LMMS requires its authenticated BF16 Instruct descriptor')
    if not config.get('stack_adapter'):
        header = json.loads((PROJECT_ROOT / config['path'] / 'config.json').read_text())
        if (config.get('token_keep_ratio', 1) != 1 or config.get('native_worker_python')
                or config.get('execution_backend', 'native_transformers') != 'native_transformers'
                or header.get('quantization_config') or header.get('a4fvlm_stack_widths') is not None):
            raise ValueError('Kimi Dense LMMS requires its original native BF16 Instruct descriptor')
        return
    from skills.vision_token import validate_selection
    from skills.kimi_vl_artifact import validate_quantized_bindings
    ratio = config.get('token_keep_ratio')
    if config['stack_adapter'] is not True or type(ratio) not in (int, float) or not 0 < ratio < 1:
        raise ValueError('Optimized Kimi LMMS requires an explicit Token keep ratio in (0, 1)')
    try:
        validate_selection(config.get('token_selection', {}))
    except (KeyError, TypeError) as error:
        raise ValueError('Optimized Kimi LMMS requires its complete Token selection policy') from error
    if config.get('execution_backend') == 'vllm':
        from skills.stack_runtime import validate_runtime_settings
        runtime = config.get('runtime', {})
        if runtime.get('vllm_max_model_len') != 131072 or runtime.get('cudagraph_mm_encoder', False):
            raise ValueError('Kimi serving Runtime requires native context 131072 and no encoder CUDA Graph')
        validate_runtime_settings(runtime)
    else:
        python = config.get('native_worker_python')
        if not isinstance(python, str) or not python or not (PROJECT_ROOT / python).is_file():
            raise ValueError('Optimized Kimi LMMS requires its isolated native_worker_python')
    path = PROJECT_ROOT / config['path']
    header = json.loads((path / 'config.json').read_text())
    if header.get('a4fvlm_stack_widths') is None or header.get('a4fvlm_kimi_gptq') is None:
        raise ValueError('Optimized Kimi LMMS requires a freshly quantized physical Parent')
    validate_quantized_bindings(path, header, json.loads((path / 'manifest.json').read_text()))


def register_native_model():
    from lmms_eval.models import MODEL_REGISTRY_V2, ModelManifest
    MODEL_REGISTRY_V2.register_manifest(ModelManifest(
        model_id='kimi_vl', simple_class_path='search.kimi_lmms.KimiVLLMMS'))


def main():
    config = json.loads(os.environ['A4FVLM_LMMS_KIMI_MODEL'])
    validate_native_config(config)
    register_native_model()
    runpy.run_module('lmms_eval', run_name='__main__')


if __name__ == '__main__':
    main()
