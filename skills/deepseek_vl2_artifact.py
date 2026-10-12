"""Physical DeepSeek SP Parent export and exact native topology replay."""

import copy
import json
from pathlib import Path

from skills.deepseek_vl2_pruning import dimensions, units, validate_recipe, apply_recipe, audit_recipe
from skills.quantization import digest

STACK_ARCHITECTURE = 'A4FVLMSPDeepseekVLV2ForCausalLM'
QUANT_ARCHITECTURE = 'A4FVLMQuantizedDeepseekVLV2ForCausalLM'
QUANT_METHOD = 'deepseek_vl2_multimodal_gptq_v1'


def config_dict(config):
    value = config.to_dict()
    convert = getattr(config, 'dict_dtype_to_str', None) or config.dict_torch_dtype_to_str
    convert(value)
    return value


def validate_physical_config(header, recipe):
    dims = dimensions(header['language_config'])
    validate_recipe(recipe, dims)
    if (header.get('model_type') != 'deepseek_vl_v2'
            or header.get('architectures') != [STACK_ARCHITECTURE]
            or header.get('a4fvlm_stack_physical_compaction') is not True
            or header.get('a4fvlm_stack_recipe_digest') != digest(recipe)
            or header.get('a4fvlm_stack_widths') != recipe['widths']
            or header.get('a4fvlm_stack_units') != units(dims)):
        raise ValueError('DeepSeek physical topology is not bound to its recipe')


def write_parent(model, processor, recipe, output_dir):
    from skills.internvl35_artifact import save_tensors
    if recipe != getattr(model, 'structured_pruning_recipe', None):
        raise ValueError('DeepSeek parent must contain the actually applied recipe')
    audit = audit_recipe(model, recipe)
    config = copy.deepcopy(model.config)
    config.architectures = [STACK_ARCHITECTURE]
    config.a4fvlm_stack_widths = recipe['widths']
    config.a4fvlm_stack_units = units(recipe['dimensions'])
    config.a4fvlm_stack_recipe_digest = digest(recipe)
    config.a4fvlm_stack_physical_compaction = True
    header = config_dict(config)
    validate_physical_config(header, recipe)
    if processor is None:
        raise ValueError('A native DeepSeek processor is required for a serving Parent')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_tensors(model.state_dict(), output)
    (output / 'config.json').write_text(json.dumps(header, indent=2) + '\n')
    (output / 'sp-recipe.json').write_text(json.dumps(recipe) + '\n')
    processor.save_pretrained(output)
    manifest = {'schema': 'a4fvlm.stack-parent.v1', 'architecture': STACK_ARCHITECTURE,
                'recipe_digest': digest(recipe), 'widths': recipe['widths'],
                'units': units(recipe['dimensions']), 'physical_compaction': True,
                'model_config_digest': digest(header), 'mechanism': audit}
    (output / 'stack-parent.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def read_parent(path):
    path = Path(path)
    header = json.loads((path / 'config.json').read_text())
    recipe = json.loads((path / 'sp-recipe.json').read_text())
    validate_physical_config(header, recipe)
    parent = json.loads((path / 'stack-parent.json').read_text())
    if (parent.get('schema') != 'a4fvlm.stack-parent.v1'
            or parent.get('architecture') != STACK_ARCHITECTURE
            or parent.get('recipe_digest') != digest(recipe)
            or parent.get('widths') != recipe['widths']
            or parent.get('units') != units(recipe['dimensions'])
            or parent.get('physical_compaction') is not True):
        raise ValueError('DeepSeek Parent manifest does not match its recipe')
    if not header.get('quantization_config') and parent.get('model_config_digest') != digest(header):
        raise ValueError('DeepSeek Parent configuration changed after export')
    return header, recipe, parent


def validate_quantized_bindings(path, header, manifest):
    """Check topology, complete native calibration and fresh physical ancestry."""
    from skills.quantization import validate_manifest
    from skills.deepseek_vl2_quantization import (
        expected_inventory, high_precision_exceptions, vision_padding,
    )
    path = Path(path)
    validate_manifest(manifest)
    policy = expected_inventory(header)
    selected = {name for name, precision in policy.items() if precision == 'int4_w4a16'}
    quant = header.get('quantization_config', {})
    expected = {'quant_method': 'gptq', 'bits': 4, 'group_size': 128, 'desc_act': False,
                'sym': True, 'checkpoint_format': 'gptq', 'lm_head': False,
                'modules_in_block_to_quantize': [sorted(selected)]}
    extra = {'method': QUANT_METHOD, 'format': 'gptq_v1', 'vision_padding': vision_padding(header)}
    architecture = STACK_ARCHITECTURE if header.get('a4fvlm_stack_widths') is not None else QUANT_ARCHITECTURE
    if (header.get('model_type') != 'deepseek_vl_v2' or quant != expected
            or header.get('architectures') != [architecture]
            or header.get('a4fvlm_deepseek_gptq') != extra
            or manifest.get('module_assignment') != policy
            or manifest.get('high_precision_exceptions') != high_precision_exceptions(header)):
        raise ValueError('DeepSeek quantization policy/padding does not match its native topology')
    receipt = json.loads((path / 'vision-quantization.json').read_text())
    visual = manifest.get('visual_quantization', {})
    if (manifest.get('status') != 'exported'
            or manifest.get('model', {}).get('adapter') != 'deepseek_vl2'
            or manifest.get('model_digest') != digest(manifest['model'])
            or manifest.get('calibration_digest') != digest(manifest['calibration'])
            or visual.get('method') != QUANT_METHOD
            or visual.get('artifact_config_digest') != digest(header)
            or visual.get('calibration_receipt_digest') != digest(receipt)):
        raise ValueError('DeepSeek quantization config/calibration binding mismatch')
    samples, modules = receipt.get('sample_ids', []), receipt.get('modules', [])
    if (receipt.get('method') != QUANT_METHOD
            or receipt.get('calibration') != manifest['calibration']
            or receipt.get('vision_padding') != extra['vision_padding']
            or receipt.get('sequential_projection_groups') is not True
            or receipt.get('vision_before_decoder') is not True
            or receipt.get('native_routing') is not True
            or receipt.get('sample_count') != manifest['calibration'].get('limit')
            or len(samples) != receipt.get('sample_count') or len(set(samples)) != len(samples)
            or len(modules) != len(selected) or {row.get('module') for row in modules} != selected
            or any(row.get('method') != 'gptq_hessian' or row.get('fallback') is not None
                   or row.get('samples', 0) <= 0 or row.get('observed_tokens', 0) <= 0
                   or row.get('bits') != 4 or row.get('group_size') != 128 for row in modules)):
        raise ValueError('DeepSeek requires complete native routed calibration without fallback')
    if header.get('a4fvlm_stack_widths') is not None:
        _, _, parent = read_parent(path)
        stack = manifest.get('stack', {})
        if (stack.get('parent_manifest_digest') != digest(parent)
                or stack.get('parent_recipe_digest') != parent['recipe_digest']
                or stack.get('fresh_calibration') is not True
                or receipt.get('source_config_digest') != parent['model_config_digest']):
            raise ValueError('DeepSeek quantization is not bound to its freshly calibrated Parent')
    elif manifest.get('stack') or any((path / name).exists() for name in ('sp-recipe.json', 'stack-parent.json')):
        raise ValueError('Independent DeepSeek quantization has unexpected SP ancestry')


def audit_quantized_artifact(path, manifest):
    """Inspect actual packed tensors and the native router precision exceptions."""
    from safetensors import safe_open
    from skills.quantization_evidence import audit_gptq_export
    audit = audit_gptq_export(path, manifest)
    metadata = {}
    for filename in sorted(Path(path).glob('*.safetensors')):
        with safe_open(str(filename), framework='numpy') as shard:
            for name in shard.keys():
                if name in metadata:
                    audit['errors'].append({'error': 'duplicate_tensor', 'tensor': name})
                metadata[name] = shard.get_slice(name).get_dtype()
    verified = 0
    for name, exception in manifest.get('high_precision_exceptions', {}).items():
        key = name if name.endswith(('.weight', '.e_score_correction_bias')) else name + '.weight'
        expected = {'bfloat16': 'BF16', 'float32': 'F32'}[exception['dtype']]
        if metadata.get(key) != expected:
            audit['errors'].append({'tensor': key, 'expected': expected, 'observed': metadata.get(key)})
        else:
            verified += 1
    audit['verified_high_precision_exceptions'] = verified
    audit['status'] = 'passed' if not audit['errors'] else 'blocked_preflight'
    return audit


def load_quantized_model(descriptor):
    """Reload the real packed weights into native CUDA Marlin projections."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from adapters.deepseek_vl2_dense import PROJECT_ROOT, native_model_class
    from skills.deepseek_vl2_quantization import expected_inventory, pad_vision

    if descriptor.get('dtype') != 'bfloat16' or not descriptor.get('device', '').startswith('cuda:'):
        raise ValueError('Quantized DeepSeek reload requires its BF16 CUDA Marlin worker')
    if descriptor.get('attn_implementation') != 'eager':
        raise ValueError('Quantized DeepSeek native reload requires original eager MLA')
    path = (PROJECT_ROOT / descriptor['path']).resolve()
    header = json.loads((path / 'config.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    validate_quantized_bindings(path, header, manifest)
    if audit_quantized_artifact(path, manifest)['status'] != 'passed':
        raise ValueError('DeepSeek artifact precision audit failed')
    recipe = read_parent(path)[1] if header.get('a4fvlm_stack_widths') is not None else None
    model_class = native_model_class(descriptor)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    native_config = DeepseekVLV2Config(**header)
    native_config._attn_implementation = 'eager'
    native_config.language_config._attn_implementation = 'eager'
    with init_empty_weights():
        model = model_class(native_config)
        if recipe is not None:
            apply_recipe(model, recipe)
            audit_recipe(model, recipe)
        pad_vision(model)
    modules = [(name, value) for name, value in model.named_modules() if isinstance(value, torch.nn.Linear)]
    policy = expected_inventory(header)
    if {name for name, _ in modules} != set(policy):
        raise ValueError('Reloaded DeepSeek Linear topology differs from the manifest')
    kernels = []
    for name, module in modules:
        if policy[name] != 'int4_w4a16':
            continue
        kernel = MarlinLinear(bits=4, group_size=128, desc_act=False, sym=True,
                              in_features=module.in_features, out_features=module.out_features,
                              bias=module.bias is not None, dtype=torch.bfloat16)
        owner, attribute = name.rsplit('.', 1)
        setattr(model.get_submodule(owner), attribute, kernel)
        kernels.append((name, kernel))
    state = {}
    for filename in sorted(path.glob('*.safetensors')):
        shard = load_file(str(filename), device='cpu')
        if set(state) & set(shard):
            raise ValueError('Duplicate DeepSeek artifact tensor keys')
        state.update(shard)
    model.load_state_dict(state, strict=True, assign=True)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_meta:
            raise ValueError(f'Unmaterialized DeepSeek artifact tensor: {name}')
        if value.is_floating_point() and name in state:
            dtype = torch.float32 if name.endswith('mlp.gate.e_score_correction_bias') else torch.bfloat16
            if value.dtype != dtype:
                raise ValueError(f'DeepSeek artifact changed native precision for {name}')
    model.to(device=descriptor['device']).eval()
    for name, kernel in kernels:
        if not bool((kernel.qzeros == 0x77777777).all()):
            raise ValueError(f'Expected symmetric GPTQ-v1 zero convention: {name}')
        expected_groups = torch.arange(kernel.in_features, device=kernel.g_idx.device, dtype=torch.int32) // 128
        if not torch.equal(kernel.g_idx, expected_groups):
            raise ValueError(f'DeepSeek artifact requires contiguous Group-128 packing: {name}')
        kernel.post_init()
        if kernel._marlin_tile_padding is not None:
            raise ValueError(f'Unexpected runtime padding in DeepSeek Marlin: {name}')
    model._agent4vlm_quantized_modules = [name for name, _ in kernels]
    model._agent4vlm_vision_padding = header['a4fvlm_deepseek_gptq']['vision_padding']
    return model


def load_physical_model(descriptor):
    """Exact BF16 Parent reload; quantized loading is a separate verified route."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from adapters.deepseek_vl2_dense import PROJECT_ROOT, native_model_class
    path = (PROJECT_ROOT / descriptor['path']).resolve()
    header, recipe, _ = read_parent(path)
    if header.get('quantization_config'):
        raise ValueError('A GPTQ Parent requires the quantized DeepSeek artifact loader')
    model_class = native_model_class(descriptor)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    native_config = DeepseekVLV2Config(**header)
    native_config._attn_implementation = 'eager'
    native_config.language_config._attn_implementation = 'eager'
    with init_empty_weights():
        model = model_class(native_config)
        apply_recipe(model, recipe)
    state = {}
    for filename in sorted(path.glob('*.safetensors')):
        shard = load_file(str(filename), device='cpu')
        if set(state) & set(shard):
            raise ValueError('Duplicate DeepSeek artifact tensor keys')
        state.update(shard)
    model.load_state_dict(state, strict=True, assign=True)
    if any(value.is_meta for value in model.parameters()):
        raise ValueError('DeepSeek artifact left unmaterialized parameters')
    dtype = getattr(torch, descriptor['dtype'])
    for name, parameter in model.named_parameters():
        expected = torch.float32 if name.endswith('mlp.gate.e_score_correction_bias') else dtype
        if parameter.is_floating_point() and parameter.dtype != expected:
            raise ValueError(f'DeepSeek artifact changed native precision for {name}')
    model.to(descriptor['device']).eval()
    audit_recipe(model, recipe)
    return model


def load_artifact(descriptor):
    from adapters.deepseek_vl2_dense import PROJECT_ROOT, load_processor
    header = json.loads((PROJECT_ROOT / descriptor['path'] / 'config.json').read_text())
    model = (load_quantized_model(descriptor) if header.get('quantization_config')
             else load_physical_model(descriptor))
    processor = load_processor(descriptor)
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        setattr(model.language.generation_config, key, getattr(processor.tokenizer, key))
    model.language.generation_config.use_cache = True
    return model, processor
