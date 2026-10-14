"""Lazy native Kimi vLLM geometry, physical FFNs, GPTQ and Token interfaces."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from skills.kimi_vl_pruning import dimensions, units
from skills.vllm_token import configured_runtime, image_keep_count, select_image_indices

_MODEL_CLASS = None
_PROCESSORS = {}


def layer_widths(header):
    dims = dimensions(header["text_config"])
    domain = units(dims)
    widths = header.get("a4fvlm_stack_widths", [u["original_width"] for u in domain])
    if not isinstance(widths, (list, tuple)) or len(widths) != len(domain):
        raise ValueError("Kimi vLLM widths must cover every Dense/routed/shared FFN unit")
    result = [{} for _ in range(dims["num_hidden_layers"])]
    for width, unit in zip(widths, domain, strict=True):
        if type(width) is not int or not 0 < width <= unit["original_width"] or width % 128:
            raise ValueError("Kimi vLLM requires admitted Group-128 physical widths")
        result[unit["layer"]][unit["kind"]] = width
    return result


def native_grid(image_width, image_height, processor):
    """Exact dimension arithmetic of the authenticated native rescale/pad path."""
    if any(type(v) is not int or v < 1 for v in (image_width, image_height)):
        raise ValueError("Kimi native image dimensions must be positive integers")
    patch = processor.patch_size
    width, height = image_width, image_height
    patches = (width // patch) * (height // patch)
    if patches > processor.in_token_limit:
        scale = math.sqrt(processor.in_token_limit / patches)
        width, height = int(width * scale), int(height * scale)
    mh, mw = processor.merge_kernel_size
    if processor.pad_input:
        height += (-height) % (mh * patch)
        width += (-width) % (mw * patch)
    else:
        height -= height % patch
        width -= width % patch
    ph, pw = height // patch, width // patch
    if not 0 < ph < 512 or not 0 < pw < 512 or ph % mh or pw % mw:
        raise ValueError("Kimi native grid exceeds positional capacity or merge alignment")
    return ph // mh, pw // mw


def retained_count(rows, columns, runtime=None):
    return rows * columns if runtime is None else image_keep_count(rows=rows, columns=columns, runtime=runtime)


def compact_processed_image(batch, image_token_id, merge_kernel_size, runtime):
    """Keep native pixels/grid and compact only the identical image placeholder run."""
    import torch
    grid, ids = batch["image_grid_hws"], batch["input_ids"]
    if grid.shape != (1, 2) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Kimi processing requires one native image per request")
    ph, pw = [int(v) for v in grid[0].tolist()]
    mh, mw = merge_kernel_size
    if ph < 1 or pw < 1 or ph % mh or pw % mw:
        raise ValueError("Kimi native patch grid is not merge-aligned")
    rows, columns = ph // mh, pw // mw
    full = rows * columns
    locations = (ids[0] == image_token_id).nonzero().flatten()
    if locations.numel() != full or int(locations[-1] - locations[0]) + 1 != full:
        raise ValueError("Kimi native image placeholder run differs from its grid")
    keep = retained_count(rows, columns, runtime)
    if keep == full:
        return batch
    mask = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
    mask[locations[keep:]] = False
    batch["input_ids"] = ids[:, mask]
    attention = batch.get("attention_mask")
    if attention is not None:
        if attention.shape != ids.shape:
            raise ValueError("Kimi native attention mask differs from the prompt")
        batch["attention_mask"] = attention[:, mask]
    return batch


def native_routing(gating_output, correction, config):
    """Kimi noaux_tc verbatim: FP32 sigmoid, zero mask, original unbiased weights."""
    import torch
    def get(name):
        return config[name] if isinstance(config, dict) else getattr(config, name)
    if (get("topk_method") != "noaux_tc" or get("scoring_func") != "sigmoid"
            or gating_output.dtype != torch.float32 or correction.dtype != torch.bfloat16
            or gating_output.ndim != 2 or gating_output.shape[1] != get("n_routed_experts")
            or correction.shape != (get("n_routed_experts"),)):
        raise ValueError("Kimi routing requires native FP32 logits and BF16 corrections")
    scores = gating_output.sigmoid()
    choice = scores + correction.unsqueeze(0)
    groups, top_groups = get("n_group"), get("topk_group")
    count = choice.shape[0]
    group_scores = choice.view(count, groups, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_ids = torch.topk(group_scores, k=top_groups, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_ids, 1)
    mask = group_mask.unsqueeze(-1).expand(count, groups, get("n_routed_experts") // groups).reshape(count, -1)
    _, indices = torch.topk(choice.masked_fill(~mask.bool(), 0.0),
                            k=get("num_experts_per_tok"), dim=-1, sorted=False)
    weights = scores.gather(1, indices)
    if get("num_experts_per_tok") > 1 and get("norm_topk_prob"):
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * get("routed_scaling_factor"), indices.to(torch.int32)


def native_router_logits(hidden_states, weight):
    """Original Kimi computes FP32 GEMM using its unchanged BF16 parameters."""
    import torch
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Kimi router inputs and stored weights must retain native BF16")
    return torch.nn.functional.linear(hidden_states.float(), weight.float())


def processor_class(descriptor):
    from adapters.kimi_vl_dense import load_processor, native_path
    path = native_path(descriptor)
    key = str(path)
    if key in _PROCESSORS:
        return _PROCESSORS[key]
    base = type(load_processor(descriptor))
    class NativeProcessor(base):
        def __call__(self, *args, **kwargs):
            batch = super().__call__(*args, **kwargs)
            if "image_grid_hws" not in batch:
                return batch
            return compact_processed_image(batch, self.tokenizer.convert_tokens_to_ids(self.image_token),
                                           self.image_processor.merge_kernel_size, configured_runtime())
    _PROCESSORS[key] = NativeProcessor
    return NativeProcessor


def build_vision(config, *, quant_config, prefix="vision_tower", padding):
    """Keep vLLM's native MoonViT forwards and activate every declared GPTQ Linear."""
    from vllm.model_executor.models.moonvit import MoonVitPretrainedModel
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
    if (config.intermediate_size != padding["original_width"]
            or config.num_hidden_layers != padding["layers"]):
        raise ValueError("Kimi vision padding is not bound to its original MoonViT")
    padded = copy.deepcopy(config)
    padded.intermediate_size = padding["padded_width"]
    model = MoonVitPretrainedModel(padded, prefix=prefix)
    for index, block in enumerate(model.encoder.blocks):
        for suffix, first, second, bias in (
                ("wqkv", config.hidden_size, 3 * config.hidden_size, True),
                ("wo", config.hidden_size, config.hidden_size, True),
                ("mlp.fc0", config.hidden_size, padded.intermediate_size, True),
                ("mlp.fc1", padded.intermediate_size, config.hidden_size, True)):
            root, _, attr = suffix.rpartition(".")
            owner = block.get_submodule(root) if root else block
            name = f"{prefix}.encoder.blocks.{index}.{suffix}"
            linear = ReplicatedLinear(first, second, bias=bias, quant_config=quant_config, prefix=name)
            if not isinstance(linear.quant_method, AutoGPTQLinearMethod):
                raise ValueError(f"Kimi native vision GPTQ was not activated: {name}")
            setattr(owner, attr, linear)
    return model


def language_classes():
    import torch
    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2DecoderLayer, DeepseekV2Model, DeepseekV2ForCausalLM, DeepseekV2MLP,
        DeepseekV2MoE, DeepseekV2MLAAttention,
    )
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
    from vllm.platforms import current_platform

    class NativeGate(GateLinear):
        def forward(self, hidden_states):
            return native_router_logits(hidden_states, self.weight), None

    class NativeMoE(DeepseekV2MoE):
        def __init__(self, config, widths, quant_config, prefix):
            torch.nn.Module.__init__(self)
            self.routed_scaling_factor = config.routed_scaling_factor
            self.n_routed_experts, self.n_shared_experts = config.n_routed_experts, config.n_shared_experts
            self.n_logical_experts = self.n_physical_experts = self.n_local_physical_experts = config.n_routed_experts
            self.n_redundant_experts, self.ep_size, self.ep_rank = 0, 1, 0
            self.physical_expert_start, self.physical_expert_end = 0, config.n_routed_experts
            self.is_sequence_parallel = self.enable_eplb = False
            self.gate = NativeGate(config.hidden_size, config.n_routed_experts,
                                   out_dtype=torch.float32, params_dtype=torch.bfloat16, prefix=f"{prefix}.gate")
            self.gate.e_score_correction_bias = torch.nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.bfloat16))
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size, intermediate_size=widths["shared"], hidden_act=config.hidden_act,
                quant_config=quant_config, reduce_results=False, prefix=f"{prefix}.shared_experts")
            def routing(hidden_states, gating_output, topk, renormalize):
                if topk != config.num_experts_per_tok or renormalize != config.norm_topk_prob:
                    raise ValueError("Kimi runtime changed native top-k or normalization")
                return native_routing(gating_output, self.gate.e_score_correction_bias, config)
            self.experts = FusedMoE(
                shared_experts=self.shared_experts, gate=self.gate, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, hidden_size=config.hidden_size,
                intermediate_size=widths["routed"], renormalize=config.norm_topk_prob,
                custom_routing_function=routing, quant_config=quant_config, prefix=f"{prefix}.experts",
                scoring_func=config.scoring_func, e_score_correction_bias=self.gate.e_score_correction_bias,
                router_logits_dtype=torch.float32, routed_scaling_factor=1.0,
                enable_eplb=False, num_redundant_experts=0)

    class PhysicalDecoderLayer(DeepseekV2DecoderLayer):
        def __init__(self, *, vllm_config, prefix):
            # Construct each mechanism once. Building and replacing a default
            # FusedMoE first would register duplicate static compilation layers.
            torch.nn.Module.__init__(self)
            config = copy.deepcopy(vllm_config.model_config.hf_config)
            index = int(prefix.rsplit(".", 1)[1])
            widths = config.a4fvlm_mlp_widths[index]
            self.layer_idx, self.hidden_size = index, config.hidden_size
            self.use_mha = self.use_sequence_parallel_moe = False
            self.routed_scaling_factor = config.routed_scaling_factor
            self.self_attn = DeepseekV2MLAAttention(
                vllm_config=vllm_config, config=config, hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads, qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank, kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings, cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.self_attn",
                topk_indices_buffer=None, reduce_results=True)
            from skills.kimi_vl_mla_compat import bind_packed_prefill
            bind_packed_prefill(self.self_attn.mla_attn.mla_attn.impl)
            if "routed" in widths:
                self.mlp = NativeMoE(config, widths, vllm_config.quant_config, f"{prefix}.mlp")
            else:
                self.mlp = DeepseekV2MLP(hidden_size=config.hidden_size, intermediate_size=widths["dense"],
                    hidden_act=config.hidden_act, quant_config=vllm_config.quant_config, prefix=f"{prefix}.mlp")
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @support_torch_compile
    class PhysicalLanguageModel(torch.nn.Module):
        fall_back_to_pt_during_load = False
        forward, load_weights = DeepseekV2Model.forward, DeepseekV2Model.load_weights
        embed_input_ids = DeepseekV2Model.embed_input_ids
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.config, self.device = config, current_platform.device_type
            self.hidden_size, self.vocab_size = config.hidden_size, config.vocab_size
            self.is_v32, self.use_mha, self.num_redundant_experts = False, False, 0
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.embed_tokens")
            self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers,
                lambda prefix: PhysicalDecoderLayer(vllm_config=vllm_config, prefix=prefix), prefix=f"{prefix}.layers")
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size)
            self.aux_hidden_state_layers = ()

    class PhysicalLanguage(DeepseekV2ForCausalLM):
        model_cls = PhysicalLanguageModel
        packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    return NativeGate, NativeMoE, PhysicalDecoderLayer, PhysicalLanguage


def create_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import torch
    from vllm.model_executor.models.kimi_vl import (
        KimiVLForConditionalGeneration, KimiVLProcessingInfo, KimiVLMultiModalProcessor,
        KimiVLDummyInputsBuilder, KimiVLMultiModalProjector,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.utils import maybe_prefix
    from skills.kimi_vl_artifact import validate_quantized_bindings, audit_quantized_artifact
    _, _, _, Language = language_classes()

    class NativeProcessingInfo(KimiVLProcessingInfo):
        def get_supported_mm_limits(self):
            return {"image": 1}
        def get_hf_processor(self, **kwargs):
            path = Path(self.ctx.model_config.model)
            manifest = json.loads((path / "manifest.json").read_text())
            descriptor = {**manifest["model"], "path": str(path)}
            return self.ctx.get_hf_processor(processor_class(descriptor), **kwargs)
        def get_num_image_tokens(self, *, image_width, image_height):
            rows, columns = native_grid(image_width, image_height, self.get_hf_processor().image_processor)
            return retained_count(rows, columns, configured_runtime())

    @MULTIMODAL_REGISTRY.register_processor(
        KimiVLMultiModalProcessor, info=NativeProcessingInfo, dummy_inputs=KimiVLDummyInputsBuilder,
    )
    class A4FVLMKimiVLForConditionalGeneration(KimiVLForConditionalGeneration):
        supports_encoder_cudagraph = False
        packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        def __init__(self, *, vllm_config, prefix=""):
            torch.nn.Module.__init__(self)
            path = Path(vllm_config.model_config.model)
            header = json.loads((path / "config.json").read_text())
            manifest = json.loads((path / "manifest.json").read_text())
            validate_quantized_bindings(path, header, manifest)
            if audit_quantized_artifact(path, manifest)["status"] != "passed":
                raise ValueError("Kimi vLLM artifact precision/geometry audit failed")
            parallel = vllm_config.parallel_config
            if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                    or parallel.enable_expert_parallel or parallel.enable_eplb
                    or parallel.use_sequence_parallel_moe or parallel.enable_dbo):
                raise ValueError("Kimi requires its single-GPU native expert topology")
            if not vllm_config.model_config.use_mla:
                raise ValueError("Kimi runtime requires its native MLA topology")
            if vllm_config.compilation_config.cudagraph_mm_encoder:
                raise ValueError("Kimi Token has no validated encoder CUDA Graph replay")
            self.config = vllm_config.model_config.hf_config
            self.a4fvlm_native_routers = json.loads((path / "vision-quantization.json").read_text())["routers"]
            self.quant_config = vllm_config.quant_config
            self.model_config = vllm_config.model_config
            self.hidden_size = self.config.text_config.hidden_size
            self.use_data_parallel = self.model_config.multimodal_config.mm_encoder_tp_mode == "data"
            self.media_placeholder = self.config.media_placeholder_token_id
            self.text_config = self.config.text_config
            self.text_config.a4fvlm_mlp_widths = layer_widths(header)
            self.text_config.moe_router_dtype = "float32"
            self.a4fvlm_stack_widths = header.get("a4fvlm_stack_widths")
            self.a4fvlm_token_runtime = configured_runtime()
            self.a4fvlm_token_pruning = bool(self.a4fvlm_token_runtime and self.a4fvlm_token_runtime["keep_ratio"] < 1)
            with self._mark_tower_model(vllm_config, "image"):
                self.vision_tower = build_vision(self.config.vision_config, quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "vision_tower"), padding=header["a4fvlm_kimi_gptq"]["vision_padding"])
                self.multi_modal_projector = KimiVLMultiModalProjector(self.config, prefix=maybe_prefix(prefix, "multi_modal_projector"))
            with self._mark_language_model(vllm_config):
                self.language_model = Language(vllm_config=vllm_config.with_hf_config(self.text_config),
                                               prefix=maybe_prefix(prefix, "language_model"))
            self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        def load_weights(self, weights):
            loaded = super().load_weights(weights)
            self.audit_loaded_policy()
            return loaded

        def audit_loaded_policy(self):
            from skills.kimi_vl_artifact import router_fingerprints
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
            from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod, AutoGPTQMoEMethod
            from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
            observed, backends = [], set()
            for index, (layer, widths) in enumerate(zip(self.language_model.model.layers,
                                                       self.text_config.a4fvlm_mlp_widths, strict=True)):
                if any(not isinstance(getattr(layer.self_attn, suffix).quant_method, AutoGPTQLinearMethod)
                       for suffix in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")):
                    raise ValueError("Kimi MLA GPTQ coverage differs from its native inventory")
                if "dense" in widths:
                    if any(not isinstance(getattr(layer.mlp, name).quant_method, AutoGPTQLinearMethod)
                           for name in ("gate_up_proj", "down_proj")):
                        raise ValueError("Kimi Dense FFN GPTQ was not physically loaded")
                    actual = {"dense": layer.mlp.down_proj.input_size}
                else:
                    if any(not isinstance(getattr(layer.mlp.shared_experts, name).quant_method, AutoGPTQLinearMethod)
                           for name in ("gate_up_proj", "down_proj")):
                        raise ValueError("Kimi shared FFN GPTQ was not physically loaded")
                    expert = layer.mlp.experts.routed_experts
                    method = expert.quant_method
                    if not isinstance(method, AutoGPTQMoEMethod) or method.wna16_moe_backend != WNA16MoEBackend.MARLIN:
                        raise ValueError("Kimi routed GPTQ did not use its admitted Marlin backend")
                    backends.add(method.wna16_moe_backend.value)
                    gate = layer.mlp.gate
                    if (gate.weight.dtype != torch.bfloat16 or gate.e_score_correction_bias.dtype != torch.bfloat16
                            or gate.out_dtype != torch.float32 or layer.mlp.n_routed_experts != self.text_config.n_routed_experts
                            or layer.mlp.n_shared_experts != self.text_config.n_shared_experts
                            or expert.moe_config.experts_per_token != self.text_config.num_experts_per_tok):
                        raise ValueError("Kimi native router precision or expert topology changed")
                    actual = {"routed": expert.moe_config.intermediate_size,
                              "shared": layer.mlp.shared_experts.down_proj.input_size}
                    if layer.mlp.experts.shared_experts._layer is not layer.mlp.shared_experts:
                        raise ValueError("Kimi fused runner lost its physical shared FFN")
                if actual != widths:
                    raise ValueError("Kimi observed FFN widths differ from its physical artifact")
                observed.append({"layer": index, **actual})
            for block in self.vision_tower.encoder.blocks:
                if any(not isinstance(block.get_submodule(name).quant_method, AutoGPTQLinearMethod)
                       for name in ("wqkv", "wo", "mlp.fc0", "mlp.fc1")):
                    raise ValueError("Kimi MoonViT GPTQ was not physically loaded")
            for name in ("linear_1", "linear_2"):
                if not isinstance(getattr(self.multi_modal_projector, name).quant_method, UnquantizedLinearMethod):
                    raise ValueError("Kimi multimodal projector must remain native BF16")
            if router_fingerprints(self) != self.a4fvlm_native_routers:
                raise ValueError("Kimi loaded router values differ from the exact native artifact")
            receipt = os.environ.get("A4FVLM_VLLM_MODEL_RECEIPT")
            if receipt:
                attentions = [layer.self_attn.mla_attn.mla_attn for layer in self.language_model.model.layers]
                Path(receipt).write_text(json.dumps({
                    "attention_backends": sorted({a.attn_backend.get_name() for a in attentions}),
                    "kv_cache_dtypes": sorted({a.kv_cache_dtype for a in attentions}),
                    "physical_widths": self.a4fvlm_stack_widths, "moe_layer_widths": observed,
                    "moe_backends": sorted(backends), "router_logits_dtype": "float32",
                    "router_parameter_dtype": "bfloat16", "native_routing": "noaux_tc_sigmoid_zero_mask",
                    "verified_router_tensors": len(self.a4fvlm_native_routers),
                    "expert_topology": {"routed": self.text_config.n_routed_experts,
                                        "shared": self.text_config.n_shared_experts,
                                        "top_k": self.text_config.num_experts_per_tok},
                    "vision_padding": {"original_width": self.config.vision_config.intermediate_size,
                                       "padded_width": self.vision_tower.encoder.blocks[0].mlp.fc0.output_size},
                    "token_runtime": self.a4fvlm_token_runtime, "token_pruning_enabled": self.a4fvlm_token_pruning,
                }) + "\n")

        def _process_image_input(self, image_input):
            outputs = super()._process_image_input(image_input)
            if not self.a4fvlm_token_pruning:
                return outputs
            result = []
            mh, mw = self.config.vision_config.merge_kernel_size
            for features, grid in zip(outputs, image_input["image_grid_hws"], strict=True):
                ph, pw = [int(v) for v in grid.tolist()]
                if ph % mh or pw % mw:
                    raise ValueError("Kimi vLLM image grid is not merge-aligned")
                rows, columns = ph // mh, pw // mw
                runtime = self.a4fvlm_token_runtime
                if features.shape[0] != rows * columns:
                    raise ValueError("Kimi post-projector features differ from the native grid")
                indices = select_image_indices(features, rows=rows, columns=columns, runtime=runtime)
                selected = features.index_select(0, indices)
                if selected.shape[0] != retained_count(rows, columns, runtime):
                    raise ValueError("Kimi selected features differ from compacted prompt placeholders")
                result.append(selected)
                trace = os.environ.get("A4FVLM_VLLM_SELECTION_TRACE")
                if trace:
                    with Path(trace).open("a") as stream:
                        stream.write(json.dumps({"method": "vllm_score_based_post_encoder",
                            "token_keep_ratio": runtime["keep_ratio"], "image_tokens_before": int(features.shape[0]),
                            "image_tokens_after": int(selected.shape[0]), "patch_grid": [ph, pw],
                            "merged_grid": [rows, columns], "position_type": "1d_compacted",
                            "positions_after": int(selected.shape[0]),
                            "selected_indices_sha256": hashlib.sha256(json.dumps(indices.tolist()).encode()).hexdigest()}) + "\n")
            return result

    _MODEL_CLASS = A4FVLMKimiVLForConditionalGeneration
    return _MODEL_CLASS


def __getattr__(name):
    if name == "A4FVLMKimiVLForConditionalGeneration":
        return create_model_class()
    raise AttributeError(name)
