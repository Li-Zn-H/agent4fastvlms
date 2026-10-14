"""Lazy topology helpers for physically compacted model-specific Stack parents."""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from typing import Any, Iterator

from skills.quantization import digest, validate_manifest


STACK_ARCHITECTURE = "A4FVLMSPQuantizedQwen3VLForConditionalGeneration"
LLAVA_STACK_ARCHITECTURE = "A4FVLMSPQuantizedLlavaOnevisionForConditionalGeneration"
LLAVA15_STACK_ARCHITECTURE = "A4FVLMSPQuantizedLlava15ForConditionalGeneration"


def validate_resumable_export(artifact, saved_request, request):
    """Authenticate a completed export before retrying its local evaluation."""
    for key in ("model", "evaluation", "candidate", "calibration", "quantization_runtime", "selection"):
        if saved_request.get(key) != request.get(key):
            raise ValueError(f"Stack export request mismatch: {key}")
    root = Path(artifact)
    from skills.stack_vllm import validate_stack_artifact

    validate_stack_artifact(root)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    parent = json.loads((root / "stack-parent.json").read_text())
    config = json.loads((root / "config.json").read_text())
    candidate = request["candidate"]
    recipe = candidate["structured_pruning"]["recipe_digest"]
    if (manifest.get("status") != "exported"
            or manifest.get("calibration") != request["calibration"]
            or parent.get("recipe_digest") != recipe
            or manifest.get("stack", {}).get("parent_recipe_digest") != recipe
            or manifest["stack"].get("parent_manifest_digest") != digest(parent)
            or manifest["stack"].get("fresh_calibration") is not True):
        raise ValueError("Stack export does not match its physical parent/calibration")
    visual = manifest.get("visual_quantization")
    if visual is not None:
        receipt = json.loads((root / "vision-quantization.json").read_text())
        if (visual.get("artifact_config_digest") != digest(config)
                or visual.get("calibration_receipt_digest") != digest(receipt)):
            raise ValueError("Stack visual export configuration/calibration changed")
    return {"artifact_digest": manifest["artifact_digest"],
            "source_request_digest": digest(saved_request),
            "reused_export_only": True, "quality_reused": False}


def stack_architecture_for_config(config: Any) -> str:
    """Return the Stack architecture for the checkpoint family."""

    model_type = getattr(config, "model_type", None)
    if model_type is None and isinstance(config, dict):
        model_type = config.get("model_type")
    if model_type == "deepseek_vl_v2":
        from skills.deepseek_vl2_artifact import STACK_ARCHITECTURE as deepseek_architecture
        return deepseek_architecture
    if model_type == "kimi_vl":
        from skills.kimi_vl_artifact import STACK_ARCHITECTURE as kimi_architecture
        return kimi_architecture
    if model_type == "llava":
        return LLAVA15_STACK_ARCHITECTURE
    if model_type == "internvl_chat":
        from skills.internvl35_artifact import STACK_ARCHITECTURE as internvl_architecture
        return internvl_architecture
    return LLAVA_STACK_ARCHITECTURE if model_type == "llava_onevision" else STACK_ARCHITECTURE


def create_pruned_model_class(model_adapter: str = "qwen3_vl") -> type[Any]:
    """Create the exact per-layer HF topology only inside a Worker."""

    if model_adapter == "llava15":
        from transformers import LlavaForConditionalGeneration
        from transformers.models.llama.modeling_llama import LlamaMLP

        base_class, mlp_class = LlavaForConditionalGeneration, LlamaMLP
        architecture = LLAVA15_STACK_ARCHITECTURE
    elif model_adapter == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        base_class = LlavaOnevisionForConditionalGeneration
        mlp_class = Qwen2MLP
        architecture = LLAVA_STACK_ARCHITECTURE
    else:
        from transformers import Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        base_class = Qwen3VLForConditionalGeneration
        mlp_class = Qwen3VLTextMLP
        architecture = STACK_ARCHITECTURE

    class A4FVLMSPQuantizedModel(base_class):
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            if model_adapter not in {"llava_onevision", "llava15"}:
                return super().from_pretrained(*args, **kwargs)
            requested_info = kwargs.pop("output_loading_info", False)
            model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"LLaVA Stack checkpoint did not load exactly: {info}")
            return (model, info) if requested_info else model

        @classmethod
        def from_config(cls, config: Any, **kwargs: Any) -> Any:
            """Compatibility factory required by GPTQModel 7.x loaders."""

            # GPTQModel calls this inside its own ``init_empty_weights``
            # context. Transformers 5.14 removed the legacy PreTrainedModel
            # factory, so the topology class must expose the small adapter
            # explicitly. Loader kwargs (dtype, trust_remote_code, etc.) are
            # consumed by GPTQModel around construction.
            del kwargs
            return cls(config)

        def __init__(self, config: Any) -> None:
            raw = getattr(config, "a4fvlm_stack_widths", None)
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValueError("Stack topology width vector is missing")
            widths = tuple(int(value) for value in raw)
            text_config = config.text_config
            if len(widths) != int(text_config.num_hidden_layers):
                raise ValueError("Stack topology width vector has the wrong depth")
            original = int(text_config.intermediate_size)
            if any(value <= 0 or value > original or value % 128 for value in widths):
                raise ValueError("Stack topology widths are not Group-128 legal")
            super().__init__(config)
            layers = self.model.language_model.layers
            import torch

            model_dtype = getattr(text_config, "dtype", None)
            if isinstance(model_dtype, str):
                model_dtype = getattr(torch, model_dtype, None)
            if not isinstance(model_dtype, torch.dtype):
                model_dtype = getattr(config, "torch_dtype", None)
            if isinstance(model_dtype, str):
                model_dtype = getattr(torch, model_dtype, None)
            for layer_index, width in enumerate(widths):
                layer_config = copy.deepcopy(text_config)
                layer_config.intermediate_size = width
                mlp = mlp_class(layer_config)
                if isinstance(model_dtype, torch.dtype):
                    mlp = mlp.to(dtype=model_dtype)
                layers[layer_index].mlp = mlp

    A4FVLMSPQuantizedModel.__name__ = architecture
    A4FVLMSPQuantizedModel.__qualname__ = architecture
    return A4FVLMSPQuantizedModel


@contextlib.contextmanager
def bind_gptqmodel_loader(model_adapter: str = "qwen3_vl") -> Iterator[type[Any]]:
    """Temporarily bind GPTQModel's loader to the exact Stack topology."""

    if model_adapter in {"llava_onevision", "llava15"}:
        from skills.quantization_adapters import _bind_llava_onevision_gptqmodel

        loader = create_pruned_model_class(model_adapter)
        with _bind_llava_onevision_gptqmodel(loader_override=loader, model_adapter=model_adapter) as bound:
            yield bound
        return

    from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel

    loader = create_pruned_model_class(model_adapter)
    previous = Qwen3_VLQModel.loader
    Qwen3_VLQModel.loader = loader
    try:
        yield loader
    finally:
        Qwen3_VLQModel.loader = previous


def write_pruned_parent(
    model: Any,
    processor: Any,
    recipe: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist a physical SP parent with an explicit topology manifest."""

    if getattr(model.config, "model_type", None) == "deepseek_vl_v2":
        from skills.deepseek_vl2_artifact import write_parent
        return write_parent(model, processor, recipe, output_dir)

    if getattr(model.config, "model_type", None) == "kimi_vl":
        from skills.kimi_vl_artifact import write_parent
        return write_parent(model, processor, recipe, output_dir)

    if getattr(model.config, "model_type", None) == "internvl_chat":
        from skills.internvl35_artifact import write_parent
        return write_parent(model, processor, recipe, output_dir)

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Stack parent already exists: {destination}")
    destination.mkdir(parents=True)
    widths = recipe.get("widths")
    if not isinstance(widths, list) or not widths:
        raise ValueError("SP recipe has no width vector")
    text_config = getattr(model.config, "text_config", None)
    original = int(getattr(text_config, "intermediate_size", 0))
    layers = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(layers, "layers", None)
    if not original or layers is None or not hasattr(layers, "__len__") or len(widths) != len(layers):
        raise ValueError("SP parent topology does not cover the loaded model")
    if any(type(width) is not int or width <= 0 or width > original or width % 128 for width in widths):
        raise ValueError("SP parent widths are not Group-128 legal")
    for layer, width in zip(layers, widths, strict=True):
        shapes = {
            name: tuple(getattr(layer.mlp, name).weight.shape)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        if shapes["gate_proj"] != (width, int(text_config.hidden_size)) or shapes["up_proj"] != (width, int(text_config.hidden_size)) or shapes["down_proj"] != (int(text_config.hidden_size), width):
            raise ValueError("SP parent weights do not match the recipe widths")
    model.config.a4fvlm_stack_widths = list(widths)
    model.config.a4fvlm_stack_recipe_digest = digest(recipe)
    model.config.a4fvlm_stack_physical_compaction = True
    model.config.architectures = [stack_architecture_for_config(model.config)]
    model.save_pretrained(destination, safe_serialization=True)
    if processor is None:
        raise ValueError("SP parent requires a processor")
    processor.save_pretrained(destination)
    manifest = {
        "schema": "a4fvlm.stack-parent.v1",
        "architecture": stack_architecture_for_config(model.config),
        "recipe_digest": digest(recipe),
        "widths": list(widths),
        "physical_compaction": True,
        "model_config_digest": digest(json.loads((destination / "config.json").read_text())),
    }
    (destination / "stack-parent.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_stack_model(model_config: dict[str, Any]) -> tuple[Any, Any]:
    """Load a Stack topology directly; Dense loading remains unchanged."""

    if model_config.get("adapter") == "deepseek_vl2":
        from skills.deepseek_vl2_artifact import load_artifact
        return load_artifact(model_config)
    if model_config.get("adapter") == "kimi_vl":
        from skills.kimi_vl_artifact import load_artifact
        return load_artifact(model_config)
    if model_config.get("adapter") == "internvl35":
        from skills.internvl35_artifact import load_artifact
        return load_artifact(model_config)

    import torch
    from transformers import AutoProcessor

    path = Path(model_config["path"])
    config = json.loads((path / "config.json").read_text())
    architecture = config.get("architectures", [None])[0]
    if architecture not in {STACK_ARCHITECTURE, LLAVA_STACK_ARCHITECTURE, LLAVA15_STACK_ARCHITECTURE}:
        raise ValueError("Stack model config does not declare the Stack architecture")
    parent_manifest = path / "stack-parent.json"
    if not parent_manifest.is_file():
        raise ValueError("Stack model is missing stack-parent.json")
    manifest = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "a4fvlm.stack-parent.v1" or manifest.get("architecture") != architecture:
        raise ValueError("Stack parent manifest is incompatible")
    widths = config.get("a4fvlm_stack_widths")
    if widths != manifest.get("widths") or config.get("a4fvlm_stack_physical_compaction") is not True:
        raise ValueError("Stack model topology metadata does not match its parent manifest")
    model_adapter = {LLAVA_STACK_ARCHITECTURE: "llava_onevision",
                     LLAVA15_STACK_ARCHITECTURE: "llava15", STACK_ARCHITECTURE: "qwen3_vl"}[architecture]
    cls = create_pruned_model_class(model_adapter)
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    context = contextlib.nullcontext()
    loading = {}
    if model_adapter in {"llava_onevision", "llava15"}:
        if model_adapter == "llava15":
            from adapters.llava15_dense import quantized_clip_initialization as quantized_initialization
        else:
            from adapters.llava_onevision_dense import quantized_siglip_initialization as quantized_initialization
        context = quantized_initialization(path)
        if config.get("quantization_config"):
            # Optimum chooses the GPTQ kernel during module replacement.
            # Loading on CPU and moving afterwards pins its CPU-compatible
            # fallback; the native LMMS loader supplies this device map too.
            loading["device_map"] = model_config.get("device", "cuda:0")
    with context:
        model = cls.from_pretrained(
            str(path),
            dtype=getattr(torch, model_config.get("dtype", "bfloat16")),
            attn_implementation=model_config.get("attn_implementation", "sdpa"),
            local_files_only=True,
            **loading,
        ).to(model_config.get("device", "cuda:0"))
    model.eval()
    return model, processor


__all__ = [
    "STACK_ARCHITECTURE",
    "LLAVA_STACK_ARCHITECTURE",
    "LLAVA15_STACK_ARCHITECTURE",
    "bind_gptqmodel_loader",
    "create_pruned_model_class",
    "load_stack_model",
    "write_pruned_parent",
    "stack_architecture_for_config",
]
