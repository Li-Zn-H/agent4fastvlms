"""Lazy vLLM adapter for the physical Stack Qwen3-VL topology.

The rest of the repository must remain importable in the Dense environment,
which does not install vLLM.  Consequently this module contains only JSON and
configuration helpers at import time.  The vLLM classes are imported when the
vLLM Worker explicitly registers the Stack architecture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from skills.stack_model import LLAVA_STACK_ARCHITECTURE, LLAVA15_STACK_ARCHITECTURE, STACK_ARCHITECTURE
from skills.vllm_token import (
    configured_runtime,
    configure_runtime,
    image_keep_count,
    select_image_indices,
)


_SHAPES = {
    "qwen3_vl": (36, 2560, 9728, STACK_ARCHITECTURE),
    "llava_onevision": (28, 3584, 18944, LLAVA_STACK_ARCHITECTURE),
    "llava": (32, 4096, 11008, LLAVA15_STACK_ARCHITECTURE),
    "internvl_chat": (40, 5120, 17408, "A4FVLMSPQuantizedInternVLChatModel"),
}
_GROUP_SIZE = 128
_VLLM_CLASSES: dict[str, type[Any]] = {}


def _value(config: object, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def stack_widths(config: object) -> tuple[int, ...]:
    """Read and validate the physical per-layer width vector from HF config."""

    model_type = _value(config, "model_type", "qwen3_vl")
    if model_type == "deepseek_vl_v2":
        from skills.deepseek_vl2_artifact import STACK_ARCHITECTURE as deepseek_architecture
        from skills.deepseek_vl2_pruning import dimensions, units
        from skills.deepseek_vl2_vllm import layer_widths
        dims = dimensions(_value(config, "language_config", _value(config, "text_config")))
        domain = units(dims)
        raw = _value(config, "a4fvlm_stack_widths")
        if (not isinstance(raw, (list, tuple))
                or _value(config, "architectures") != [deepseek_architecture]
                or _value(config, "a4fvlm_stack_physical_compaction") is not True
                or _value(config, "a4fvlm_stack_units") != domain
                or not _value(config, "a4fvlm_stack_recipe_digest")):
            raise ValueError("DeepSeek vLLM physical topology metadata is incomplete")
        layer_widths({"language_config": dims, "a4fvlm_stack_widths": raw})
        if not any(width < unit["original_width"] for width, unit in zip(raw, domain, strict=True)):
            raise ValueError("DeepSeek physical Stack is still dense")
        return tuple(raw)
    if model_type not in _SHAPES:
        raise ValueError(f"Unsupported Stack vLLM model type: {model_type}")
    layer_count, hidden_size, original_width, architecture = _SHAPES[model_type]
    if model_type == "internvl_chat":
        text = _value(config, "llm_config", _value(config, "text_config"))
        decoder = _value(text, "model_type")
        if decoder == "qwen2":
            layer_count, hidden_size, original_width = 48, 5120, 13824
        elif decoder != "qwen3":
            raise ValueError("Unsupported native InternVL Stack decoder family")
    raw = _value(config, "a4fvlm_stack_widths")
    if not isinstance(raw, (list, tuple)) or len(raw) != layer_count:
        raise ValueError(f"Stack vLLM width vector must contain {layer_count} layers")
    widths = tuple(raw)
    if any(type(width) is not int for width in widths):
        raise ValueError("Stack vLLM widths must be exact integers")
    if any(
        width <= 0 or width > original_width or width % _GROUP_SIZE
        for width in widths
    ):
        raise ValueError("Stack vLLM widths must be positive Group-128 widths")
    if not any(width < original_width for width in widths):
        raise ValueError("Stack vLLM width vector is still dense")
    if _value(config, "a4fvlm_stack_physical_compaction") is not True:
        raise ValueError("Stack vLLM artifact does not declare physical compaction")
    if _value(config, "architectures") != [architecture]:
        raise ValueError("Stack vLLM architecture does not match its adapter")
    text_config = _value(config, "text_config", _value(config, "llm_config"))
    if int(_value(text_config, "hidden_size", 0)) != hidden_size:
        raise ValueError("Stack vLLM hidden size drifted")
    if int(_value(text_config, "num_hidden_layers", 0)) != layer_count:
        raise ValueError("Stack vLLM layer count drifted")
    if int(_value(text_config, "intermediate_size", 0)) != original_width:
        raise ValueError("Stack vLLM dense intermediate size drifted")
    return widths


def validate_stack_artifact(artifact: str | Path) -> dict[str, Any]:
    """Validate the persisted Stack contract before starting a vLLM engine."""

    root = Path(artifact)
    config_path = root / "config.json"
    parent_path = root / "stack-parent.json"
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"Stack vLLM artifact is missing config.json: {root}")
    if not parent_path.is_file():
        raise ValueError("Stack vLLM artifact is missing stack-parent.json")
    if not manifest_path.is_file():
        raise ValueError("Stack vLLM artifact is missing manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    widths = stack_widths(config)
    if config.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_artifact import validate_quantized_bindings, audit_quantized_artifact
        validate_quantized_bindings(root, config, manifest)
        if audit_quantized_artifact(root, manifest)["status"] != "passed":
            raise ValueError("DeepSeek physical Stack precision audit failed")
    if config.get("model_type") == "internvl_chat":
        from adapters import select_model
        from skills.internvl35_artifact import validate_quantized_bindings
        from skills.quantization_evidence import audit_gptq_export
        select_model(config)
        validate_quantized_bindings(root, config, manifest)
        if audit_gptq_export(root, manifest)["status"] != "passed":
            raise ValueError("InternVL physical Stack precision audit failed")
    if parent.get("schema") != "a4fvlm.stack-parent.v1":
        raise ValueError("Stack vLLM parent manifest schema is unsupported")
    architecture = config.get("architectures", [None])[0]
    if parent.get("architecture") != architecture:
        raise ValueError("Stack vLLM parent manifest architecture drifted")
    if parent.get("widths") != list(widths):
        raise ValueError("Stack vLLM parent widths do not match config")
    if parent.get("physical_compaction") is not True:
        raise ValueError("Stack vLLM parent is not physically compacted")
    if manifest.get("route") != "gptqmodel_w4a16":
        raise ValueError("Stack vLLM adapter requires the GPTQModel W4A16 route")
    quantization = config.get("quantization_config") or {}
    if quantization.get("bits") != 4 or quantization.get("group_size") != _GROUP_SIZE:
        raise ValueError("Stack vLLM artifact is not GPTQ Group-128")
    return {
        "architecture": architecture,
        "widths": list(widths),
        "route": manifest["route"],
        "physical_compaction": True,
    }


def create_vllm_stack_model_class(model_adapter: str = "qwen3_vl") -> type[Any]:
    """Create the custom Stack class only inside a vLLM process."""

    if model_adapter in _VLLM_CLASSES:
        return _VLLM_CLASSES[model_adapter]

    if model_adapter in {"llava_onevision", "llava15"}:
        return _create_vllm_llava_stack_model_class(model_adapter)

    import torch

    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLDummyInputsBuilder,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMultiModalProcessor,
        Qwen3VLProcessingInfo,
    )
    from vllm.multimodal.evs import compute_mrope_for_media
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

    class StackQwen3VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
        """Qwen3-VL processor whose image placeholders match score pruning.

        The retained count is known from the configured keep ratio and grid
        before the vision encoder runs.  The actual indices are selected later
        from the post-encoder features by the model class below.
        """

        def _get_prompt_updates(
            self,
            mm_items: Any,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: Any,
        ) -> Any:
            updates = super()._get_prompt_updates(
                mm_items,
                hf_processor_mm_kwargs,
                out_mm_kwargs,
            )
            runtime = configured_runtime()
            if runtime is None or float(runtime["keep_ratio"]) >= 1:
                return updates

            hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

            def image_replacement(item_idx: int) -> PromptUpdateDetails:
                item = out_mm_kwargs["image"][item_idx]
                grid = item["image_grid_thw"].data
                t, h, w = (int(value) for value in grid.tolist())
                if t != 1:
                    raise ValueError("Score-based vLLM Token pruning supports still images")
                rows = h // int(hf_processor.image_processor.merge_size)
                columns = w // int(hf_processor.image_processor.merge_size)
                count = image_keep_count(
                    rows=rows,
                    columns=columns,
                    runtime=runtime,
                )
                return PromptUpdateDetails.from_seq(
                    [hf_processor.image_token_id] * count
                )

            rewritten = []
            for update in updates:
                if update.modality == "image":
                    rewritten.append(
                        PromptReplacement(
                            modality="image",
                            target=update.target,
                            replacement=image_replacement,
                        )
                    )
                else:
                    rewritten.append(update)
            return rewritten

    @MULTIMODAL_REGISTRY.register_processor(
        StackQwen3VLMultiModalProcessor,
        info=Qwen3VLProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class StackQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
        """Qwen3-VL with one quantized fused MLP per physical SP width."""

        def __init__(self, *, vllm_config: Any, prefix: str = "model") -> None:
            config = vllm_config.model_config.hf_config
            widths = stack_widths(config)
            super().__init__(vllm_config=vllm_config, prefix=prefix)

            self.a4fvlm_token_runtime = configured_runtime()
            self.a4fvlm_token_pruning = bool(
                self.a4fvlm_token_runtime
                and float(self.a4fvlm_token_runtime["keep_ratio"]) < 1
            )
            if self.a4fvlm_token_pruning:
                # The native Qwen3-VL path already has dynamic multimodal
                # pruning and mRoPE recomputation support.  Enabling the flag
                # also disables encoder CUDA graphs, which cannot represent a
                # variable post-encoder token count.
                self.is_multimodal_pruning_enabled = True

            text_config = config.text_config
            language_model = self.language_model
            layers = language_model.model.layers
            start_layer = int(getattr(language_model.model, "start_layer", 0))
            if start_layer + len(layers) > len(widths):
                raise ValueError("Stack vLLM decoder layer partition is invalid")
            quant_config = getattr(language_model, "quant_config", None)
            if quant_config is None:
                quant_config = getattr(vllm_config, "quant_config", None)
            layer_facts: list[dict[str, int]] = []
            for local_index, layer in enumerate(layers):
                layer_index = start_layer + local_index
                width = widths[layer_index]
                mlp_prefix = (
                    f"{prefix}.language_model.model.layers.{layer_index}.mlp"
                )
                layer.mlp = Qwen2MLP(
                    hidden_size=int(text_config.hidden_size),
                    intermediate_size=width,
                    hidden_act=str(text_config.hidden_act),
                    quant_config=quant_config,
                    prefix=mlp_prefix,
                )
                output_sizes = tuple(int(value) for value in layer.mlp.gate_up_proj.output_sizes)
                if output_sizes != (width, width) or int(layer.mlp.down_proj.input_size) != width:
                    raise ValueError("Stack vLLM fused MLP shape does not match width vector")
                layer_facts.append(
                    {
                        "layer_index": layer_index,
                        "intermediate_width": width,
                    }
                )
            self.a4fvlm_stack_widths = tuple(widths)
            self.a4fvlm_stack_layer_facts = tuple(layer_facts)
            self.a4fvlm_stack_quant_config_present = quant_config is not None
            model_receipt = os.environ.get("A4FVLM_VLLM_MODEL_RECEIPT")
            if model_receipt:
                # Record in the engine process itself; no callable RPC or
                # pickle-based serialization is required for this audit.
                Path(model_receipt).write_text(json.dumps({
                    "attention_backends": sorted({layer.self_attn.attn.attn_backend.get_name() for layer in layers}),
                    "kv_cache_dtypes": sorted({layer.self_attn.attn.kv_cache_dtype for layer in layers}),
                    "physical_widths": list(self.a4fvlm_stack_widths),
                    "token_runtime": self.a4fvlm_token_runtime,
                    "token_pruning_enabled": self.a4fvlm_token_pruning,
                }) + "\n")

        def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
            runtime = self.a4fvlm_token_runtime
            if not self.a4fvlm_token_pruning:
                return super()._postprocess_image_embeds_evs(
                    image_embeds_split, image_input
                )

            merge_size = self.visual.spatial_merge_size
            grid_thw = image_input["image_grid_thw"]
            output = []
            for emb, size in zip(image_embeds_split, grid_thw.tolist()):
                t, h, w = (int(value) for value in size)
                if t != 1:
                    raise ValueError("Score-based vLLM Token pruning supports still images")
                rows, columns = h // merge_size, w // merge_size
                # Qwen3-VL concatenates the main visual projection and each
                # DeepStack projection along the final dimension.  Scores use
                # only the main projection, while one index mask is shared by
                # every stream just as in the Transformers controller.
                indices = select_image_indices(
                    emb[:, : self.visual_dim],
                    rows=rows,
                    columns=columns,
                    runtime=runtime,
                )
                emb = emb.index_select(0, indices)

                positions = compute_mrope_for_media(size, merge_size).to(
                    emb.device, non_blocking=True
                )
                positions = positions.index_select(0, indices)
                positions = torch.cat(
                    [positions, torch.zeros_like(positions[:, 0:1])], dim=1
                )
                output.append(torch.cat([emb, positions], dim=1))
                # Opt-in LMMS audit, after actual feature/DeepStack/mRoPE gather.
                # No tracing or device synchronization in normal benchmarks.
                trace = os.environ.get("A4FVLM_VLLM_SELECTION_TRACE")
                if trace:
                    import hashlib

                    selected = indices.detach().cpu().tolist()
                    record = {
                        "controller": "vllm_score_based_post_encoder",
                        "token_keep_ratio": runtime["keep_ratio"],
                        "image_tokens_before": rows * columns,
                        "image_tokens_after": int(emb.shape[0]),
                        "mrope_positions": int(positions.shape[0]),
                        "selected_indices_sha256": hashlib.sha256(
                            json.dumps(selected).encode()
                        ).hexdigest(),
                    }
                    with open(trace, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
            return tuple(output)

    StackQwen3VLForConditionalGeneration.__name__ = STACK_ARCHITECTURE
    StackQwen3VLForConditionalGeneration.__qualname__ = STACK_ARCHITECTURE
    globals()[STACK_ARCHITECTURE] = StackQwen3VLForConditionalGeneration
    _VLLM_CLASSES[model_adapter] = StackQwen3VLForConditionalGeneration
    return StackQwen3VLForConditionalGeneration


def _create_vllm_llava_stack_model_class(model_adapter="llava_onevision") -> type[Any]:
    """Create the selected LLaVA physical MLP and post-encoder Token path."""

    import hashlib
    import torch
    if model_adapter == "llava15":
        from vllm.model_executor.models.llava import (
            LlavaDummyInputsBuilder as DummyInputsBuilder,
            LlavaForConditionalGeneration as BaseModel,
            LlavaMultiModalProcessor as BaseProcessor,
            LlavaProcessingInfo as ProcessingInfo,
        )
        from vllm.model_executor.models.llama import LlamaMLP as DecoderMLP
        from skills.llava15_vllm import clip_weights_mapper as weights_mapper, normalize_clip_weight_name as normalize_weight_name
        architecture = LLAVA15_STACK_ARCHITECTURE
    else:
        from vllm.model_executor.models.llava_onevision import (
            LlavaOnevisionDummyInputsBuilder as DummyInputsBuilder,
            LlavaOnevisionForConditionalGeneration as BaseModel,
            LlavaOnevisionMultiModalProcessor as BaseProcessor,
            LlavaOnevisionProcessingInfo as ProcessingInfo,
        )
        from vllm.model_executor.models.qwen2 import Qwen2MLP as DecoderMLP
        from skills.llava_vllm import siglip_weights_mapper as weights_mapper, normalize_siglip_weight_name as normalize_weight_name
        architecture = LLAVA_STACK_ARCHITECTURE
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
    from skills.llava_token import clip_grid, select_clip_indices, select_packed_indices

    select_features = select_clip_indices if model_adapter == "llava15" else select_packed_indices

    class StackLlavaMultiModalProcessor(BaseProcessor):
        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            updates = super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )
            runtime = configured_runtime()
            if runtime is None or float(runtime["keep_ratio"]) >= 1:
                return updates
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_index
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            def replacement(item_idx):
                if isinstance(images, ImageEmbeddingItems):
                    before = images.get_feature_size(item_idx)
                else:
                    size = images.get_image_size(item_idx)
                    before = self.info.get_num_image_tokens(
                        image_width=size.width, image_height=size.height
                    )
                rows, columns = clip_grid(int(before)) if model_adapter == "llava15" else (1, int(before))
                count = image_keep_count(rows=rows, columns=columns, runtime=runtime)
                return [image_token_id] * count

            result = []
            for update in updates:
                if update.modality == "image":
                    result.append(
                        PromptReplacement(
                            modality="image", target=update.target, replacement=replacement
                        )
                    )
                else:
                    result.append(update)
            return result

    @MULTIMODAL_REGISTRY.register_processor(
        StackLlavaMultiModalProcessor,
        info=ProcessingInfo,
        dummy_inputs=DummyInputsBuilder,
    )

    class StackLlavaForConditionalGeneration(BaseModel):
        hf_to_vllm_mapper = weights_mapper(BaseModel.hf_to_vllm_mapper)
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                                  "gate_up_proj": ["gate_proj", "up_proj"]}

        def load_weights(self, weights):
            return super().load_weights((normalize_weight_name(name), tensor)
                                        for name, tensor in weights)

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            config = vllm_config.model_config.hf_config
            runtime = configured_runtime()
            self.a4fvlm_token_runtime = runtime
            self.a4fvlm_token_pruning = bool(
                runtime and float(runtime["keep_ratio"]) < 1
            )
            layers = self.language_model.model.layers
            widths = stack_widths(config)
            start_layer = int(getattr(self.language_model.model, "start_layer", 0))
            quant_config = getattr(self.language_model, "quant_config", None)
            for local_index, layer in enumerate(layers):
                layer_index = start_layer + local_index
                width = widths[layer_index]
                layer.mlp = DecoderMLP(
                    hidden_size=int(config.text_config.hidden_size),
                    intermediate_size=width,
                    hidden_act=str(config.text_config.hidden_act),
                    quant_config=quant_config,
                    prefix=(f"{prefix}." if prefix else "") + f"language_model.model.layers.{layer_index}.mlp",
                )
            receipt = os.environ.get("A4FVLM_VLLM_MODEL_RECEIPT")
            if receipt:
                Path(receipt).write_text(json.dumps({
                    "attention_backends": sorted({layer.self_attn.attn.attn_backend.get_name() for layer in layers}),
                    "kv_cache_dtypes": sorted({layer.self_attn.attn.kv_cache_dtype for layer in layers}),
                    "physical_widths": list(widths),
                    "token_runtime": runtime,
                    "token_pruning_enabled": self.a4fvlm_token_pruning,
                }) + "\n")

        def _process_image_input(self, image_input):
            output = super()._process_image_input(image_input)
            runtime = self.a4fvlm_token_runtime
            if not self.a4fvlm_token_pruning:
                return output
            trace = os.environ.get("A4FVLM_VLLM_SELECTION_TRACE")
            selected_output = []
            for emb in output:
                before = int(emb.shape[0])
                selected = select_features(emb, runtime["selection"], runtime["keep_ratio"])
                selected = selected.to(device=emb.device)
                compacted = emb.index_select(0, selected)
                if trace:
                    selected_list = [int(value) for value in selected.tolist()]
                    with Path(trace).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "method": "vllm_score_based_post_encoder",
                            "token_keep_ratio": runtime["keep_ratio"],
                            "image_tokens_before": before,
                            "image_tokens_after": int(compacted.shape[0]),
                            "position_type": "1d_compacted",
                            "positions_after": int(compacted.shape[0]),
                            "selected_indices_sha256": hashlib.sha256(json.dumps(selected_list).encode()).hexdigest(),
                        }) + "\n")
                selected_output.append(compacted)
            return selected_output

    StackLlavaForConditionalGeneration.__name__ = architecture
    StackLlavaForConditionalGeneration.__qualname__ = architecture
    globals()[architecture] = StackLlavaForConditionalGeneration
    _VLLM_CLASSES[model_adapter] = StackLlavaForConditionalGeneration
    return StackLlavaForConditionalGeneration


def register_stack_vllm_model(model_adapter: str = "qwen3_vl") -> dict[str, str]:
    """Register the Stack architecture with the current vLLM registry."""

    model_class = create_vllm_stack_model_class(model_adapter)
    from vllm.model_executor.models.registry import ModelRegistry

    architecture = {"llava_onevision": LLAVA_STACK_ARCHITECTURE,
                    "llava15": LLAVA15_STACK_ARCHITECTURE, "qwen3_vl": STACK_ARCHITECTURE}[model_adapter]

    ModelRegistry.register_model(
        architecture,
        f"{__name__}:{model_class.__name__}",
    )
    return {
        "architecture": architecture,
        "model_class": model_class.__name__,
        "module": __name__,
    }


def prepare_vllm_artifact(
    artifact: str | Path,
    *,
    token_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and register a custom artifact, leaving Dense routes untouched."""

    configured = configure_runtime(token_runtime)
    root = Path(artifact)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if config.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_artifact import validate_quantized_bindings, audit_quantized_artifact
        manifest = json.loads((root / "manifest.json").read_text())
        validate_quantized_bindings(root, config, manifest)
        if audit_quantized_artifact(root, manifest)["status"] != "passed":
            raise ValueError("DeepSeek vLLM artifact precision audit failed")
        return {"custom_stack": config.get("a4fvlm_stack_widths") is not None,
                "architecture": architectures[0], "model_type": "deepseek_vl2",
                "worker_start_method": os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
                "widths": config.get("a4fvlm_stack_widths"),
                "model_class_overrides": {architectures[0]:
                    "skills.deepseek_vl2_vllm:A4FVLMDeepseekVLV2ForCausalLM"},
                "vision_token": {"runtime": configured}}
    if config.get("model_type") == "internvl_chat":
        from adapters import select_model
        from skills.internvl35_artifact import validate_quantized_bindings
        native_adapter = select_model(config)
        if config.get("a4fvlm_internvl_gptq") is not None:
            from skills.quantization_evidence import audit_gptq_export
            manifest = json.loads((root / "manifest.json").read_text())
            validate_quantized_bindings(root, config, manifest)
            if audit_gptq_export(root, manifest)["status"] != "passed":
                raise ValueError("InternVL vLLM artifact precision audit failed")
        # On the pinned InternVL host, fork stalled before weight loading;
        # spawn completed loading and CUDA warmup in a clean engine process.
        # This execution-only preparation is never called by the catalog.
        start_method = os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        return {"custom_stack": config.get("a4fvlm_stack_widths") is not None,
                "architecture": architectures[0], "model_type": native_adapter.id,
                "worker_start_method": start_method,
                "widths": config.get("a4fvlm_stack_widths"),
                "model_class_overrides": {architectures[0]: "skills.internvl35_vllm:A4FVLMInternVLChatModel"},
                "vision_token": {"runtime": configured}}
    if config.get("model_type") == "llava_onevision" and architectures == ["LlavaOnevisionForConditionalGeneration"]:
        return {"custom_stack": False, "architecture": architectures[0], "model_type": config["model_type"],
                "model_class_overrides": {architectures[0]:
                    "skills.llava_vllm:A4FVLMLlavaOnevisionForConditionalGeneration"}}
    if config.get("model_type") == "llava" and architectures == ["LlavaForConditionalGeneration"]:
        return {"custom_stack": False, "model_type": "llava", "model_class_overrides": {
            "LlavaForConditionalGeneration": "skills.llava15_vllm:A4FVLMLlava15ForConditionalGeneration"}}
    if architectures not in ([STACK_ARCHITECTURE], [LLAVA_STACK_ARCHITECTURE], [LLAVA15_STACK_ARCHITECTURE]):
        return {"custom_stack": False, "architecture": config.get("architectures"),
                "model_type": config.get("model_type")}
    validation = validate_stack_artifact(root)
    model_adapter = {LLAVA_STACK_ARCHITECTURE: "llava_onevision", LLAVA15_STACK_ARCHITECTURE: "llava15",
                     STACK_ARCHITECTURE: "qwen3_vl"}[architectures[0]]
    registration = register_stack_vllm_model(model_adapter)
    return {
        "custom_stack": True,
        "model_type": model_adapter,
        **validation,
        "registration": registration,
        "vision_token": {
            "status": "score_based_post_encoder"
            if configured is not None and configured["keep_ratio"] < 1
            else "disabled",
            "runtime": configured,
        },
    }


__all__ = [
    "create_vllm_stack_model_class",
    "prepare_vllm_artifact",
    "register_stack_vllm_model",
    "stack_widths",
    "validate_stack_artifact",
]


# vLLM's model registry inspects a lazy `<module>:<class>` target in a fresh
# subprocess.  That subprocess has already imported vLLM, so materialize the
# class there to make the target discoverable.  Dense/CPU imports never have
# vLLM in sys.modules and therefore remain dependency-free.
if "vllm" in sys.modules:  # pragma: no cover - exercised by vLLM registry
    create_vllm_stack_model_class()
    create_vllm_stack_model_class("llava_onevision")
    create_vllm_stack_model_class("llava15")
