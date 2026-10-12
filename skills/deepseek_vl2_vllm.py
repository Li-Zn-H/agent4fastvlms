"""Lazy vLLM port of native DeepSeek geometry, GPTQ and physical MoE widths."""

import copy
import hashlib
import json
from math import ceil
import os
from pathlib import Path

from skills.deepseek_vl2_pruning import dimensions, units
from skills.deepseek_vl2_token import image_layout, select_image_indices
from skills.vllm_token import configured_runtime

_MODEL_CLASS = None


def layer_widths(header):
    """Keep dense/routed/shared widths distinct without changing expert counts."""
    dims = dimensions(header['language_config'])
    domain = units(dims)
    widths = header.get('a4fvlm_stack_widths', [unit['original_width'] for unit in domain])
    if len(widths) != len(domain):
        raise ValueError('DeepSeek vLLM width vector does not match its MoE unit domain')
    rows = [{} for _ in range(dims['num_hidden_layers'])]
    for width, unit in zip(widths, domain, strict=True):
        if type(width) is not int or not 0 < width <= unit['original_width'] or width % 128:
            raise ValueError('DeepSeek vLLM requires admitted Group-128 physical widths')
        rows[unit['layer']][unit['kind']] = width
    return rows


def image_keep_count(side, nx, ny, runtime=None):
    from skills.vision_token import keep_count
    tiles, protected, full = image_layout(side, nx, ny)
    if runtime is None or runtime['keep_ratio'] == 1:
        return full
    return len(tiles) * keep_count(side, side, runtime['keep_ratio'], runtime['selection']['coverage_grid']) + len(protected)


def compact_processed_image(batch, image_token_id, side, runtime):
    """Compact identical image placeholders; selection happens on real features."""
    import torch
    crop = batch['images_spatial_crop']
    ids = batch['input_ids']
    if tuple(crop.shape) != (1, 2) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('DeepSeek serving requires exactly one native image per request')
    nx, ny = [int(v) for v in crop[0].tolist()]
    full = image_layout(side, nx, ny)[2]
    locations = (ids[0] == image_token_id).nonzero().flatten()
    counts = batch['num_image_tokens']
    declared = counts.tolist() if isinstance(counts, torch.Tensor) else counts
    if (len(locations) != full or int(locations[-1] - locations[0]) + 1 != full
            or declared != [full]):
        raise ValueError('DeepSeek native placeholder run does not match its crop layout')
    keep = image_keep_count(side, nx, ny, runtime)
    if keep == full:
        return batch
    mask = torch.ones(ids.shape[1], device=ids.device, dtype=torch.bool)
    mask[locations[keep:]] = False
    batch['input_ids'] = ids[:, mask]
    batch['images_seq_mask'] = batch['images_seq_mask'][mask]
    batch['num_image_tokens'] = counts.new_tensor([keep]) if isinstance(counts, torch.Tensor) else [keep]
    return batch


def processor_classes():
    from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
    from vllm.model_executor.models.deepseek_vl2 import DeepseekVL2ProcessingInfo

    class NativeProcessor(DeepseekVLV2Processor):
        def process_one(self, prompt, images, inference_mode=True, **kwargs):
            if len(images) != 1 or not inference_mode:
                raise ValueError('DeepSeek serving requires one native image in inference mode')
            batch = super().process_one(prompt, images, inference_mode=inference_mode, **kwargs)
            side = ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            return compact_processed_image(batch, self.image_token_id, side, configured_runtime())

    class NativeProcessingInfo(DeepseekVL2ProcessingInfo):
        def get_supported_mm_limits(self):
            return {'image': 1}

        def get_hf_processor(self, **kwargs):
            return self.ctx.get_hf_processor(NativeProcessor, **kwargs)

        def get_num_image_tokens(self, *, image_width, image_height, cropping=True):
            if not cropping:
                raise ValueError('DeepSeek single-image serving retains native dynamic cropping')
            processor = self.get_hf_processor()
            width, height = processor.select_best_resolution((image_width, image_height))
            side = ceil((processor.image_size // processor.patch_size) / processor.downsample_ratio)
            return image_keep_count(side, width // processor.image_size, height // processor.image_size,
                                    configured_runtime())

    return NativeProcessor, NativeProcessingInfo


def build_vision(config, *, quant_config=None, prefix='vision', padding=None):
    """Construct the original floor patch grid; never pad pixels before ViT."""
    from functools import partial
    import torch
    from timm.models.vision_transformer import VisionTransformer
    from skills.llava_quantization import pad_siglip_mlp
    if not config.ignore_head or config.class_token or config.global_pool != 'map':
        raise ValueError('DeepSeek vision requires the native patch-only forward_features path')
    model = VisionTransformer(img_size=config.image_size, patch_size=config.patch_size,
        embed_dim=config.width, depth=config.layers, num_heads=config.heads,
        mlp_ratio=config.mlp_ratio, class_token=False, global_pool='map', num_classes=0,
        dynamic_img_size=False, dynamic_img_pad=False, weight_init='skip',
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), act_layer=partial(torch.nn.GELU, approximate='tanh'))
    if padding is not None:
        if padding['original_width'] != int(config.width * config.mlp_ratio) or padding['layers'] != config.layers:
            raise ValueError('DeepSeek vLLM vision padding does not match its native topology')
        for block in model.blocks:
            pad_siglip_mlp(block.mlp, padding['padded_width'])
    if quant_config is not None:
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or not name.startswith('blocks.'):
                continue
            linear = ReplicatedLinear(module.in_features, module.out_features, bias=module.bias is not None,
                                      quant_config=quant_config, prefix=f'{prefix}.{name}', return_bias=False)
            if not isinstance(linear.quant_method, AutoGPTQLinearMethod):
                raise ValueError(f'DeepSeek vision GPTQ was not activated for {prefix}.{name}')
            owner, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(owner), attr, linear)
    return model


def language_classes():
    """Retain vLLM MLA/MoE forwards and loaders with physical construction widths."""
    import torch
    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2DecoderLayer, DeepseekV2Model, DeepseekV2ForCausalLM, DeepseekV2MLP,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
    from vllm.platforms import current_platform

    class PhysicalDecoderLayer(DeepseekV2DecoderLayer):
        def __init__(self, *, vllm_config, prefix):
            config = copy.deepcopy(vllm_config.model_config.hf_config)
            index = int(prefix.rsplit('.', 1)[1])
            width = config.a4fvlm_mlp_widths[index]
            if 'dense' in width:
                config.intermediate_size = width['dense']
            else:
                config.moe_intermediate_size = width['routed']
            super().__init__(vllm_config=vllm_config, config=config, prefix=prefix)
            if 'shared' in width:
                wrapper = self.mlp.experts.shared_experts
                if wrapper is None or wrapper._layer is not self.mlp.shared_experts:
                    raise ValueError('Unexpected vLLM shared-expert ownership')
                # Replace both references before loading or any forward execution.
                # Expert count and routed top-k retain the original config values.
                shared = DeepseekV2MLP(hidden_size=config.hidden_size, intermediate_size=width['shared'],
                    hidden_act=config.hidden_act, quant_config=vllm_config.quant_config,
                    reduce_results=False, prefix=f'{prefix}.mlp.shared_experts')
                self.mlp.shared_experts = shared
                wrapper._layer = shared

    @support_torch_compile
    class PhysicalLanguageModel(torch.nn.Module):
        fall_back_to_pt_during_load = False
        forward = DeepseekV2Model.forward
        load_weights = DeepseekV2Model.load_weights
        embed_input_ids = DeepseekV2Model.embed_input_ids

        def __init__(self, *, vllm_config, prefix=''):
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.config, self.device = config, current_platform.device_type
            self.hidden_size, self.vocab_size = config.hidden_size, config.vocab_size
            self.is_v32, self.use_mha, self.num_redundant_experts = False, False, 0
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                quant_config=vllm_config.quant_config, prefix=f'{prefix}.embed_tokens')
            self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers,
                lambda prefix: PhysicalDecoderLayer(vllm_config=vllm_config, prefix=prefix),
                prefix=f'{prefix}.layers')
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
                ['hidden_states', 'residual'], config.hidden_size)
            self.aux_hidden_state_layers = ()

    class PhysicalLanguage(DeepseekV2ForCausalLM):
        model_cls = PhysicalLanguageModel
        packed_modules_mapping = {'gate_up_proj': ['gate_proj', 'up_proj']}

    return PhysicalDecoderLayer, PhysicalLanguageModel, PhysicalLanguage


def create_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import torch
    from vllm.model_executor.models.deepseek_vl2 import (
        DeepseekVLV2ForCausalLM, DeepseekVL2MultiModalProcessor, DeepseekVL2DummyInputsBuilder, MlpProjector,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.model_executor.models.utils import maybe_prefix
    from skills.deepseek_vl2_artifact import validate_quantized_bindings, audit_quantized_artifact

    _, ProcessingInfo = processor_classes()
    _, _, Language = language_classes()

    @MULTIMODAL_REGISTRY.register_processor(
        DeepseekVL2MultiModalProcessor, info=ProcessingInfo, dummy_inputs=DeepseekVL2DummyInputsBuilder,
    )
    class A4FVLMDeepseekVLV2ForCausalLM(DeepseekVLV2ForCausalLM):
        supports_encoder_cudagraph = False
        packed_modules_mapping = {'gate_up_proj': ['gate_proj', 'up_proj']}

        def __init__(self, *, vllm_config, prefix=''):
            # Construct each branch at the required shape before weight loading.
            torch.nn.Module.__init__(self)
            config = vllm_config.model_config.hf_config
            path = Path(vllm_config.model_config.model)
            header = json.loads((path / 'config.json').read_text())
            manifest = json.loads((path / 'manifest.json').read_text())
            validate_quantized_bindings(path, header, manifest)
            if audit_quantized_artifact(path, manifest)['status'] != 'passed':
                raise ValueError('DeepSeek vLLM artifact precision audit failed')
            parallel = vllm_config.parallel_config
            if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                    or parallel.enable_expert_parallel or parallel.enable_eplb
                    or parallel.use_sequence_parallel_moe or parallel.enable_dbo):
                raise ValueError('DeepSeek vLLM requires the verified single-GPU expert topology')
            if vllm_config.compilation_config.cudagraph_mm_encoder:
                raise ValueError('DeepSeek Token has no validated encoder CUDA Graph replay')
            if (not vllm_config.model_config.use_mla or not config.text_config.kv_lora_rank
                    or config.text_config.kv_lora_rank != header['language_config'].get('kv_lora_rank')):
                raise ValueError('DeepSeek runtime MLA dimensions differ from the exported native defaults')
            self.config = config
            self.multimodal_config = vllm_config.model_config.multimodal_config
            self.vision_config, self.projector_config, self.text_config = (
                config.vision_config, config.projector_config, config.text_config)
            self.text_config.a4fvlm_mlp_widths = layer_widths(header)
            self.text_config.moe_router_dtype = 'float32'
            self.image_token_id = cached_tokenizer_from_config(vllm_config.model_config).vocab['<image>']
            self.a4fvlm_token_runtime = configured_runtime()
            self.a4fvlm_token_pruning = bool(self.a4fvlm_token_runtime and self.a4fvlm_token_runtime['keep_ratio'] < 1)
            with self._mark_tower_model(vllm_config, 'image'):
                self.vision = build_vision(self.vision_config, quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, 'vision'), padding=header['a4fvlm_deepseek_gptq']['vision_padding'])
                self.projector = MlpProjector(self.projector_config)
                self.tile_tag, self.global_view_pos = config.tile_tag, config.global_view_pos
                self.image_newline = torch.nn.Parameter(torch.empty(self.projector_config.n_embed))
                self.view_seperator = torch.nn.Parameter(torch.empty(self.projector_config.n_embed))
            with self._mark_language_model(vllm_config):
                self.language_model = Language(vllm_config=vllm_config.with_hf_config(self.text_config),
                                               prefix=maybe_prefix(prefix, 'language_model'))
            self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
            self.a4fvlm_stack_widths = header.get('a4fvlm_stack_widths')

        def load_weights(self, weights):
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
            from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod, AutoGPTQMoEMethod
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
            loaded = super().load_weights(weights)
            layers = self.language_model.model.layers
            observed, moe_backends = [], set()
            for index, (layer, widths) in enumerate(zip(layers, self.text_config.a4fvlm_mlp_widths, strict=True)):
                attention = layer.self_attn
                if (not isinstance(attention.kv_b_proj.quant_method, UnquantizedLinearMethod)
                        or attention.kv_b_proj.weight.dtype != torch.bfloat16
                        or any(not isinstance(getattr(attention, name).quant_method, AutoGPTQLinearMethod)
                               for name in ('q_proj', 'kv_a_proj_with_mqa', 'o_proj'))):
                    raise ValueError('DeepSeek vLLM MLA precision policy was not applied')
                if 'dense' in widths:
                    actual = {'dense': layer.mlp.down_proj.input_size}
                else:
                    expert = layer.mlp.experts.routed_experts
                    method = expert.quant_method
                    if (not isinstance(method, AutoGPTQMoEMethod)
                            or method.wna16_moe_backend != WNA16MoEBackend.MARLIN):
                        raise ValueError('DeepSeek routed GPTQ did not select the admitted Marlin backend')
                    moe_backends.add(method.wna16_moe_backend.value)
                    if (layer.mlp.gate.e_score_correction_bias.dtype != torch.float32
                            or layer.mlp.gate.out_dtype != torch.float32
                            or layer.mlp.n_routed_experts != self.text_config.n_routed_experts
                            or layer.mlp.n_shared_experts != self.text_config.n_shared_experts
                            or expert.moe_config.experts_per_token != self.text_config.num_experts_per_tok):
                        raise ValueError('DeepSeek native router precision or expert counts changed')
                    actual = {'routed': expert.moe_config.intermediate_size,
                              'shared': layer.mlp.shared_experts.down_proj.input_size}
                    if layer.mlp.experts.shared_experts._layer is not layer.mlp.shared_experts:
                        raise ValueError('DeepSeek fused runner lost its physical shared-expert branch')
                if actual != widths:
                    raise ValueError('DeepSeek observed FFN widths differ from the physical artifact')
                observed.append({'layer': index, **actual})
            receipt = os.environ.get('A4FVLM_VLLM_MODEL_RECEIPT')
            if receipt:
                attentions = [layer.self_attn.mla_attn.mla_attn for layer in layers]
                Path(receipt).write_text(json.dumps({
                    'attention_backends': sorted({attn.attn_backend.get_name() for attn in attentions}),
                    'kv_cache_dtypes': sorted({attn.kv_cache_dtype for attn in attentions}),
                    'physical_widths': self.a4fvlm_stack_widths, 'moe_layer_widths': observed,
                    'moe_backends': sorted(moe_backends), 'router_logits_dtype': 'float32',
                    'vision_patch_grid': list(self.vision.patch_embed.grid_size),
                    'token_runtime': self.a4fvlm_token_runtime,
                    'token_pruning_enabled': self.a4fvlm_token_pruning,
                }) + '\n')
            return loaded

        def _pixel_values_to_embedding(self, pixel_values, images_spatial_crop):
            outputs = super()._pixel_values_to_embedding(pixel_values, images_spatial_crop)
            if not self.a4fvlm_token_pruning:
                return outputs
            side = ceil((self.vision_config.image_size // self.vision_config.patch_size)
                        / self.projector_config.downsample_ratio)
            result = []
            for features, crop in zip(outputs, images_spatial_crop, strict=True):
                nx, ny = [int(v) for v in crop.tolist()]
                runtime = self.a4fvlm_token_runtime
                indices = select_image_indices(features, side, nx, ny, runtime['keep_ratio'], runtime['selection'])
                selected = features.index_select(0, indices)
                if selected.shape[0] != image_keep_count(side, nx, ny, runtime):
                    raise ValueError('DeepSeek runtime features and compacted placeholders disagree')
                result.append(selected)
                trace = os.environ.get('A4FVLM_VLLM_SELECTION_TRACE')
                if trace:
                    with Path(trace).open('a') as stream:
                        stream.write(json.dumps({'method': 'vllm_score_based_post_encoder',
                            'token_keep_ratio': runtime['keep_ratio'], 'image_tokens_before': int(features.shape[0]),
                            'image_tokens_after': int(selected.shape[0]), 'tiles_selected': 1 + nx * ny,
                            'protected_layout_tokens': len(image_layout(side, nx, ny)[1]),
                            'position_type': '1d_compacted', 'positions_after': int(selected.shape[0]),
                            'selected_indices_sha256': hashlib.sha256(json.dumps(indices.tolist()).encode()).hexdigest()}) + '\n')
            return result

        def _process_image_input(self, image_input):
            if image_input['type'] != 'pixel_values':
                raise ValueError('DeepSeek serving requires native image pixels')
            return super()._process_image_input(image_input)

    _MODEL_CLASS = A4FVLMDeepseekVLV2ForCausalLM
    return _MODEL_CLASS


def __getattr__(name):
    if name == 'A4FVLMDeepseekVLV2ForCausalLM':
        return create_model_class()
    raise AttributeError(name)
