"""Native sequential GPTQ for DeepSeek vision, MLA and all routed/shared FFNs.

The original eager MLA reads kv_b_proj.weight directly; that projection remains
BF16. Every declared INT4 projection must receive actual native activations.
No synthetic expert inputs, RTN fallback or another model's statistics are used.
"""

import json
from pathlib import Path

from skills.deepseek_vl2_pruning import dimensions, units
from skills.internvl35_quantization import _move, unpack_gptq
from skills.quantization import assignment, digest


def expected_inventory(header):
    """Enumerate exact Linear names from the supported native topology."""
    if header.get('model_type') != 'deepseek_vl_v2':
        raise ValueError('Expected the native DeepSeek-VL2 topology')
    language = header['language_config']
    if language.get('q_lora_rank') is not None:
        raise ValueError('DeepSeek GPTQ currently requires the native full-model q_proj path')
    vision, projector = header['vision_config'], header.get('projector_config', {})
    if (vision.get('ignore_head', True) is not True or vision.get('global_pool', 'map') != 'map'
            or projector.get('projector_type', 'downsample_mlp_gelu') != 'downsample_mlp_gelu'
            or projector.get('depth', 2) != 2 or projector.get('token_pooling', False)):
        raise ValueError('Unsupported DeepSeek vision/projector topology')
    names = ['projector.layers.0', 'projector.layers.2', 'language.lm_head']
    # The native model stores this head but ignore_head=True bypasses it.
    names += [f'vision.attn_pool.{suffix}' for suffix in ('q', 'kv', 'proj', 'mlp.fc1', 'mlp.fc2')]
    dims = dimensions(language)
    for index in range(dims['num_hidden_layers']):
        names += [f'language.model.layers.{index}.self_attn.{suffix}'
                  for suffix in ('q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj')]
    for unit in units(dims):
        root = f"language.model.layers.{unit['layer']}.mlp"
        branches = ([root] if unit['kind'] == 'dense' else [root + '.shared_experts']
                    if unit['kind'] == 'shared' else
                    [f'{root}.experts.{i}' for i in range(unit['experts'])])
        names += [f'{branch}.{suffix}' for branch in branches
                  for suffix in ('gate_proj', 'up_proj', 'down_proj')]
    names += [f'vision.blocks.{i}.{suffix}' for i in range(header['vision_config']['layers'])
              for suffix in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')]
    return assignment(names, model_adapter='deepseek_vl2')


def high_precision_exceptions(header):
    result = {name: {'dtype': 'bfloat16', 'reason':
                     'native_MLA_direct_weight_access' if name.endswith('kv_b_proj') else
                     'unused_native_vision_pooling_head' if name.startswith('vision.attn_pool.') else
                     'preserved_multimodal_interface_or_output_head'}
              for name, precision in expected_inventory(header).items() if precision == 'bf16'}
    for unit in units(dimensions(header['language_config'])):
        if unit['kind'] == 'routed':
            root = f"language.model.layers.{unit['layer']}.mlp.gate"
            result[root + '.weight'] = {'dtype': 'bfloat16', 'reason': 'preserved_native_routing'}
            result[root + '.e_score_correction_bias'] = {
                'dtype': 'float32', 'reason': 'preserved_original_routing_correction'}
    return result


def vision_padding(header):
    from skills.llava_quantization import padded_width
    vision = header['vision_config']
    original = int(vision['width'] * vision['mlp_ratio'])
    return {'original_width': original, 'padded_width': padded_width(original),
            'added_channels': padded_width(original) - original,
            'layers': vision['layers'], 'method': 'zero_extend_internal_mlp_channels'}


def pad_vision(native):
    from skills.llava_quantization import pad_siglip_mlp
    from skills.deepseek_vl2_artifact import config_dict
    padding = vision_padding(config_dict(native.config))
    for block in native.vision.blocks:
        if (block.mlp.fc1.out_features != padding['original_width']
                or block.mlp.fc2.in_features != padding['original_width']):
            raise ValueError('DeepSeek vision padding requires a fresh native MLP')
    for block in native.vision.blocks:
        pad_siglip_mlp(block.mlp, padding['padded_width'])
    return padding


def capture_inputs(layer, samples, invoke):
    """Capture real native block kwargs without serializing a live MLA cache."""
    class Captured(Exception):
        pass

    rows = []
    def capture(module, args, kwargs):
        if len(args) != 1:
            raise ValueError('Expected native hidden states as the sole positional argument')
        kwargs = dict(kwargs)
        for name in ('past_key_value', 'past_key_values'):
            if name in kwargs:
                if kwargs[name] is not None:
                    raise ValueError('DeepSeek calibration must start without a cache')
                kwargs[name] = None
        if 'use_cache' in kwargs:
            if kwargs['use_cache']:
                raise ValueError('DeepSeek calibration requires use_cache=False')
            kwargs['use_cache'] = False
        rows.append((_move(args[0], 'cpu'), _move(kwargs, 'cpu')))
        raise Captured()

    handle = layer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for sample in samples:
            before = len(rows)
            try:
                invoke(sample)
            except Captured:
                pass
            if len(rows) != before + 1:
                raise ValueError('Native calibration must visit the first block exactly once')
    finally:
        handle.remove()
    return rows


def decoder_groups(layer):
    """All experts observe native routing in each sequential projection group."""
    mlp = layer.mlp
    if hasattr(mlp, 'experts'):
        branches = [f'mlp.experts.{i}' for i in range(len(mlp.experts))]
        if hasattr(mlp, 'shared_experts'):
            branches.append('mlp.shared_experts')
    else:
        branches = ['mlp']
    return [('self_attn.q_proj', 'self_attn.kv_a_proj_with_mqa'), ('self_attn.o_proj',),
            tuple(f'{branch}.{suffix}' for branch in branches for suffix in ('gate_proj', 'up_proj')),
            tuple(f'{branch}.down_proj' for branch in branches)]


def quantize_blocks(layers, prefix, rows, groups, packed, reports):
    import torch
    from skills.llava_quantization import new_gptq_task, pack_gptq_task

    def forward(layer, hidden, kwargs, device):
        output = layer(hidden.to(device), **_move(kwargs, device))
        # Native vision returns a Tensor; the decoder returns (hidden, ...).
        output = output[0] if isinstance(output, tuple) else output
        if not isinstance(output, torch.Tensor) or output.shape != hidden.shape:
            raise ValueError('DeepSeek block changed the native hidden-state shape')
        return output

    with torch.inference_mode():
        for index, layer in enumerate(layers):
            device = next(layer.parameters()).device
            for names in groups(layer) if callable(groups) else groups:
                tasks, handles, coverage = {}, [], {}
                try:
                    for name in names:
                        linear = layer.get_submodule(name)
                        if not isinstance(linear, torch.nn.Linear):
                            raise ValueError('GPTQ requires fresh physical Linear modules')
                        task = new_gptq_task(linear, bits=4, group_size=128)
                        tasks[name] = task
                        coverage[name] = 0
                        def collect(module, args, output, task=task, name=name):
                            count = args[0].numel() // args[0].shape[-1]
                            if count:
                                coverage[name] += count
                                task.add_batch(args[0].detach(), output.detach())
                        handles.append(linear.register_forward_hook(collect))
                    for hidden, kwargs in rows:
                        forward(layer, hidden, kwargs, device)
                    missing = [name for name, count in coverage.items() if count == 0]
                    if missing:
                        raise ValueError(f'Unobserved DeepSeek GPTQ modules in {prefix}.{index}: {missing}')
                    # Remove hooks before solving so packing cannot collect synthetic inputs.
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    for name, task in tasks.items():
                        full_name = f'{prefix}.{index}.{name}'
                        tensors, report = pack_gptq_task(task)
                        linear = layer.get_submodule(name)
                        tensors = {key: value.to(linear.weight.dtype) if value.is_floating_point() else value
                                   for key, value in tensors.items()}
                        linear.weight.copy_(unpack_gptq(tensors).to(device=device, dtype=linear.weight.dtype))
                        packed[full_name] = tensors
                        reports.append({'module': full_name, **report,
                                        'observed_tokens': coverage[name],
                                        'scale_dtype': str(tensors['scales'].dtype)})
                finally:
                    for handle in handles:
                        handle.remove()
                    for task in tasks.values():
                        if hasattr(task, 'quantizer'):
                            task.free()
                # Drop Hessian/task/module references before the next group.
                tasks.clear()
            rows = [(forward(layer, hidden, kwargs, device).detach().cpu(), kwargs)
                    for hidden, kwargs in rows]
            print(f'GPTQ {prefix} block {index + 1}/{len(layers)}', flush=True)
    return rows


def quantize_native_modules(native, samples, inputs):
    import torch
    if native.training:
        raise ValueError('DeepSeek calibration requires native eval routing')
    packed, reports = {}, []
    padding = pad_vision(native)
    with torch.inference_mode():
        vision_rows = capture_inputs(native.vision.blocks[0], samples,
                                     lambda row: native.prepare_inputs_embeds(**inputs(row)))
        quantize_blocks(native.vision.blocks, 'vision.blocks', vision_rows,
                        (('attn.qkv',), ('attn.proj',), ('mlp.fc1',), ('mlp.fc2',)), packed, reports)
        del vision_rows
        def invoke(row):
            batch = inputs(row)
            embeddings = native.prepare_inputs_embeds(**batch)
            native.language(inputs_embeds=embeddings, attention_mask=batch['attention_mask'],
                            use_cache=False, output_attentions=False)
        decoder_rows = capture_inputs(native.language.model.layers[0], samples, invoke)
        quantize_blocks(native.language.model.layers, 'language.model.layers', decoder_rows,
                        decoder_groups, packed, reports)
    return packed, reports, padding


def validate_loaded_parent(native, header, descriptor, recipe=None):
    """Bind actual Linear shapes and retained channels before calibration."""
    import torch
    from accelerate import init_empty_weights
    from adapters.deepseek_vl2_dense import native_model_class
    from skills.deepseek_vl2_pruning import apply_recipe, audit_recipe, modules

    dims = dimensions(header['language_config'])
    if dimensions(native.config.language_config) != dims:
        raise ValueError('DeepSeek loaded Parent dimensions differ from its source topology')
    if getattr(native, 'structured_pruning_recipe', None) != recipe:
        raise ValueError('DeepSeek loaded Parent does not match its source recipe')
    # Meta construction derives vision, projector and MLA shapes from the pinned
    # native implementation, without allocating another copy of the weights.
    model_class = native_model_class(descriptor)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    config = DeepseekVLV2Config(**header)
    config._attn_implementation = 'eager'
    config.language_config._attn_implementation = 'eager'
    with init_empty_weights():
        reference = model_class(config)
        if recipe is not None:
            apply_recipe(reference, recipe)
    def geometry(model):
        return {name: {'weight': list(module.weight.shape),
                       'bias': list(module.bias.shape) if module.bias is not None else None}
                for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    expected, actual = geometry(reference), geometry(native)
    if expected != actual:
        changed = sorted(name for name in expected.keys() | actual.keys()
                         if expected.get(name) != actual.get(name))
        raise ValueError(f'DeepSeek loaded Parent Linear topology differs from its source: {changed[:8]}')
    for name, module in native.named_modules():
        if isinstance(module, torch.nn.Linear) and (
                module.weight.dtype != torch.bfloat16
                or module.bias is not None and module.bias.dtype != torch.bfloat16):
            raise ValueError(f'DeepSeek loaded Parent Linear must remain BF16: {name}')
    # Shapes alone do not prove that expert count and routing top-k are intact.
    modules(native, dims)
    if recipe is not None:
        audit_recipe(native, recipe)
    return {'linear_count': len(actual), 'linear_shapes_digest': digest(actual),
            'recipe_digest': digest(recipe) if recipe is not None else None,
            'router_and_expert_counts_preserved': True}


def export_multimodal_gptq(*, candidate, model, processor, calibration, output_dir,
                          module_names, loaded_model=None):
    import shutil
    import torch
    # Import the backend before constructing a parent; first-import frames must
    # not retain a completed multi-billion-parameter native model.
    import gptqmodel  # noqa: F401
    from adapters.deepseek_vl2_dense import load_model
    from search.model import PROJECT_ROOT, prepare_inputs
    from search.evaluate import load_samples
    from skills.deepseek_vl2_artifact import (
        QUANT_METHOD, QUANT_ARCHITECTURE, config_dict, read_parent, audit_quantized_artifact,
    )
    from skills.internvl35_artifact import save_tensors
    from skills.quantization import make_manifest

    if (model.get('adapter') != 'deepseek_vl2' or candidate.get('route') != 'gptqmodel_w4a16'
            or candidate.get('bits') != 4 or candidate.get('group_size') != 128
            or candidate.get('vision_quantization') != QUANT_METHOD):
        raise ValueError('DeepSeek requires its explicit native multimodal GPTQ W4A16 Group-128 route')
    source = (PROJECT_ROOT / model['path']).resolve()
    header = json.loads((source / 'config.json').read_text())
    if header.get('quantization_config'):
        raise ValueError('GPTQ requires a fresh Dense or physically pruned BF16 Parent')
    policy = expected_inventory(header)
    if set(module_names) != set(policy):
        raise ValueError('DeepSeek export requires every native decoder, vision and preserved Linear')
    parent_data = read_parent(source) if header.get('a4fvlm_stack_widths') is not None else None
    recipe, parent = (parent_data[1], parent_data[2]) if parent_data is not None else (None, None)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError('Existing DeepSeek export must be inspected before replacement')
    rows = load_samples(PROJECT_ROOT / calibration['data'],
                        media_root=PROJECT_ROOT / calibration['media_root'])[:calibration['limit']]
    if len(rows) != calibration['limit'] or len({row['stable_row_id'] for row in rows}) != len(rows):
        raise ValueError('Incomplete or repeated native calibration inputs')
    if loaded_model is None:
        loaded_model, processor = load_model(model)
    native = loaded_model.eval()
    actual = {name for name, value in native.named_modules() if isinstance(value, torch.nn.Linear)}
    if actual != set(policy) or native.dtype != torch.bfloat16:
        raise ValueError('DeepSeek loaded BF16 Parent does not match the exact module inventory')
    topology = validate_loaded_parent(native, header, model, recipe)
    for name, parameter in native.named_parameters():
        if name.endswith('mlp.gate.e_score_correction_bias') and parameter.dtype != torch.float32:
            raise ValueError('DeepSeek router correction biases must remain original F32')
    def inputs(row):
        batch = prepare_inputs(processor, row, calibration['media_root']).to(native.device)
        if batch['input_ids'].shape[1] > native.config.language_config.max_position_embeddings:
            raise ValueError('DeepSeek calibration exceeds native context; no truncation is allowed')
        return batch
    packed, reports, padding = quantize_native_modules(native, rows, inputs)
    if (set(packed) != {name for name, precision in policy.items() if precision == 'int4_w4a16'}
            or any(row['fallback'] is not None or row['observed_tokens'] <= 0 for row in reports)):
        raise ValueError('Incomplete DeepSeek native GPTQ calibration or unsupported fallback')
    state = {key: value for key, value in native.state_dict().items() if key.rsplit('.', 1)[0] not in packed}
    for name, tensors in packed.items():
        state.update({name + '.' + key: value for key, value in tensors.items()})
    config = config_dict(native.config)
    if parent is None:
        config['architectures'] = [QUANT_ARCHITECTURE]
    config['a4fvlm_deepseek_gptq'] = {'method': QUANT_METHOD, 'format': 'gptq_v1',
                                    'vision_padding': padding}
    config['quantization_config'] = {
        'quant_method': 'gptq', 'bits': 4, 'group_size': 128, 'desc_act': False,
        'sym': True, 'checkpoint_format': 'gptq', 'lm_head': False,
        'modules_in_block_to_quantize': [sorted(packed)],
    }
    destination.mkdir(parents=True, exist_ok=False)
    save_tensors(state, destination)
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    processor.save_pretrained(destination)
    receipt = {'method': QUANT_METHOD, 'calibration': calibration,
               'sample_ids': [row['stable_row_id'] for row in rows], 'sample_count': len(rows),
               'sequential_projection_groups': True, 'vision_before_decoder': True,
               'native_routing': True, 'modules': reports, 'vision_padding': padding,
               'source_config_digest': digest(header), 'source_topology': topology}
    (destination / 'vision-quantization.json').write_text(json.dumps(receipt, indent=2) + '\n')
    manifest = make_manifest(model=model, config=candidate, calibration=calibration,
                             module_names=module_names, status='exported')
    manifest['visual_quantization'] = {'method': QUANT_METHOD, 'artifact_config_digest': digest(config),
                                      'calibration_receipt_digest': digest(receipt)}
    manifest['high_precision_exceptions'] = high_precision_exceptions(config)
    if parent is not None:
        manifest['stack'] = {'parent_recipe_digest': parent['recipe_digest'],
                             'parent_manifest_digest': digest(parent), 'fresh_calibration': True,
                             'fresh_shape_reparse': True}
        for name in ('stack-parent.json', 'sp-recipe.json'):
            shutil.copyfile(source / name, destination / name)
    manifest['artifact_digest'] = digest({key: value for key, value in manifest.items() if key != 'artifact_digest'})
    audit = audit_quantized_artifact(destination, manifest)
    (destination / 'precision-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if audit['status'] != 'passed':
        raise ValueError('DeepSeek GPTQ export does not implement its declared precision policy')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'manifest': str(destination / 'manifest.json'), 'vision_quantization': QUANT_METHOD}
