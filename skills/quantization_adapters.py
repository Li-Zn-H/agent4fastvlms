"""Opt-in backend adapters for the independent quantization skill.

This module is imported only inside a quantization Worker.  The adapters keep
framework-specific imports and API assumptions out of planning, SP, and Stack.
Each adapter must write a reloadable artifact and a checked manifest; failures
are returned to the Worker as candidate failures and never become a selected
artifact.
"""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from search.model import PROJECT_ROOT
from skills.quantization import make_manifest, validate_manifest


class AdapterError(RuntimeError):
    """A backend was present but could not produce a validated artifact."""


def _llava_checkpoint_weight_map(source):
    """Read tensor locations for both native shards and compact single files."""
    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text())["weight_map"]
    single = source / "model.safetensors"
    if not single.is_file():
        raise RuntimeError(f"LLaVA checkpoint has no safetensors file or shard index: {source}")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}


def _llava_checkpoint_prefix(weight_map, prefix, strip):
    """Resolve original HF4 and physically saved HF5 auxiliary tensor names."""
    selected = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if selected:
        return selected, strip
    if prefix == "vision_tower.":
        mapped, source_strip = "model.vision_tower.", "model.vision_tower."
    elif prefix == "image_newline":
        mapped, source_strip = "model.image_newline", "model."
    else:
        aliases = (("language_model.model.", "model.language_model."),
                   ("language_model.lm_head.", "lm_head."),
                   ("multi_modal_projector.", "model.multi_modal_projector."))
        for old, new in aliases:
            if prefix.startswith(old) and strip.startswith(old):
                mapped = new + prefix[len(old):]
                source_strip = new + strip[len(old):]
                break
        else:
            raise RuntimeError(f"LLaVA checkpoint has no {prefix} tensors")
    selected = {key: shard for key, shard in weight_map.items() if key.startswith(mapped)}
    if not selected:
        raise RuntimeError(f"LLaVA checkpoint has no {prefix} tensors")
    return selected, source_strip


@contextmanager
def _bind_llava_onevision_gptqmodel(*, loader_override=None, model_adapter="llava_onevision"):
    """Temporarily teach GPTQModel the LLaVA-OneVision/Qwen2 topology.

    GPTQModel 7.5 has a ``llava_qwen2`` definition for older LLaVA wrappers,
    but this checkpoint's decoder lives at
    ``model.language_model.layers`` and its processor emits
    ``image_sizes``/``batch_num_images``.  Keep the mapping local to the
    isolated quantization Worker so the Qwen3-VL catalog and loader remain
    untouched.
    """

    if model_adapter not in {"llava_onevision", "llava15"}:
        raise ValueError("Unsupported native LLaVA GPTQ adapter")
    model_type = "llava" if model_adapter == "llava15" else "llava_onevision"
    auto = importlib.import_module("gptqmodel.models.auto")
    from gptqmodel.models.base import BaseQModel
    from gptqmodel.utils.model import MODALITY, move_to
    from transformers import AutoModelForImageTextToText, AutoProcessor

    class LlavaOneVisionQModel(BaseQModel):
        loader = loader_override or AutoModelForImageTextToText
        pre_lm_head_norm_module = "model.language_model.norm"
        module_tree = [
            "model",
            "language_model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        ]
        modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
        require_load_processor = True

        @classmethod
        def extract_layers_node(cls):
            return ["model.language_model.layers", "language_model.layers"]

        @staticmethod
        def _images(conversation):
            from PIL import Image

            images = []
            for message in conversation:
                for item in message.get("content", []):
                    if item.get("type") != "image":
                        continue
                    value = item.get("image") or item.get("image_url")
                    if isinstance(value, dict):
                        value = value.get("url") or value.get("path")
                    if not value:
                        raise ValueError("LLaVA calibration image is missing a path")
                    with Image.open(value) as image:
                        images.append(image.convert("RGB"))
            return images

        def load_processor(self):
            return AutoProcessor.from_pretrained(
                self.model_local_path, trust_remote_code=self.require_trust_remote_code
            )

        def _materialize_module(self, owner, name, module_path):
            module = getattr(owner, name)
            if "_turtle_lock" not in self.__dict__ and "shell_module_materialize" not in self.__dict__:
                materialized = move_to(module, device=self.quantize_config.device)
            else:
                materialized = self.shell_module_materialize(
                    module, self.quantize_config.device, module_path=module_path
                )
            setattr(owner, name, materialized)

        def _load_dense_auxiliary_modules(self):
            """Load the non-decoder vision path omitted by GPTQModel's turtle.

            GPTQModel's module tree intentionally indexes only decoder layers;
            its lazy shell therefore has meta tensors for LLaVA's vision tower
            and projector. Read those checkpoint prefixes into their native HF
            modules before multimodal calibration capture.
            """
            import torch
            from safetensors import safe_open
            if model_adapter == "llava15":
                from transformers.models.llava.modeling_llava import LlavaMultiModalProjector as Projector
                from transformers import CLIPVisionModel as VisionModel
            else:
                from transformers.models.llava_onevision.modeling_llava_onevision import (
                    LlavaOnevisionMultiModalProjector as Projector,
                )
                from transformers.models.siglip.modeling_siglip import SiglipVisionModel as VisionModel

            source = Path(self.model_local_path)
            weight_map = _llava_checkpoint_weight_map(source)
            device = self.quantize_config.device

            def load_prefix(prefix, strip):
                selected, source_strip = _llava_checkpoint_prefix(weight_map, prefix, strip)
                state = {}
                for shard in sorted(set(selected.values())):
                    with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
                        for key, filename in selected.items():
                            if filename == shard:
                                state[key[len(source_strip):]] = handle.get_tensor(key).to(device)
                return state

            core = self.model.model
            vision = VisionModel(self.model.config.vision_config)
            vision.load_state_dict(
                load_prefix("vision_tower.", "vision_tower.vision_model."), strict=True
            )
            core.vision_tower = vision.to(device=device, dtype=torch.float16)
            projector = Projector(self.model.config)
            projector.load_state_dict(
                load_prefix("multi_modal_projector.", "multi_modal_projector."), strict=True
            )
            core.multi_modal_projector = projector.to(device=device, dtype=torch.float16)
            if model_adapter == "llava_onevision":
                newline = load_prefix("image_newline", "")
                core.image_newline = torch.nn.Parameter(newline["image_newline"].to(device))
            embed = load_prefix(
                "language_model.model.embed_tokens.",
                "language_model.model.embed_tokens.",
            )
            core.language_model.embed_tokens = torch.nn.Embedding.from_pretrained(
                embed["weight"].to(device), freeze=False
            )
            norm = load_prefix("language_model.model.norm.", "language_model.model.norm.")
            core.language_model.norm.load_state_dict(norm, strict=True, assign=True)
            # GPTQModel keeps the language head in bf16, but its lazy shell
            # still needs a real tensor during the final offload pass.
            lm_head = load_prefix("language_model.lm_head.", "language_model.lm_head.")
            self.model.lm_head.load_state_dict(lm_head, strict=True, assign=True)
            for index, layer in enumerate(core.language_model.layers):
                state = load_prefix(
                    f"language_model.model.layers.{index}.",
                    f"language_model.model.layers.{index}.",
                )
                layer.load_state_dict(state, strict=True, assign=True)
                layer.to(device=device, dtype=torch.float16)

        def pre_quantize_generate_hook_start(self):
            # GPTQModel replaces decoder layers with lazy shell modules during
            # capture. LLaVA's vision tower and multimodal mapper still need
            # real weights to build the first decoder-layer inputs.
            core = self.model.model
            self._materialize_module(core, "vision_tower", "model.vision_tower")
            self._materialize_module(
                core, "multi_modal_projector", "model.multi_modal_projector"
            )
            self._materialize_module(
                core.language_model,
                "embed_tokens",
                "model.language_model.embed_tokens",
            )
            self._materialize_module(
                core.language_model,
                "rotary_emb",
                "model.language_model.rotary_emb",
            )
            self._load_dense_auxiliary_modules()

        def pre_quantize_generate_hook_end(self):
            core = self.model.model
            core.vision_tower = move_to(core.vision_tower, device="cpu")
            core.multi_modal_projector = move_to(core.multi_modal_projector, device="cpu")
            core.language_model.embed_tokens = move_to(
                core.language_model.embed_tokens, device="cpu"
            )
            core.language_model.rotary_emb = move_to(
                core.language_model.rotary_emb, device="cpu"
            )

        def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
            from gptqmodel.utils.calibration import batched_conversations

            processor = self.load_processor()
            batches = []
            for batch in batched_conversations(calibration_dataset, batch_size):
                text = processor.apply_chat_template(
                    batch, tokenize=False, add_generation_prompt=True
                )
                images = [self._images(conversation)[0] for conversation in batch]
                batches.append(
                    processor(
                        text=text,
                        images=images,
                        padding=True,
                        return_tensors="pt",
                    )
                )
            del processor
            return batches

    LlavaOneVisionQModel.__name__ = "Llava15QModel" if model_adapter == "llava15" else "LlavaOneVisionQModel"
    previous = auto.MODEL_MAP.get(model_type)
    was_supported = model_type in auto.SUPPORTED_MODELS
    auto.MODEL_MAP[model_type] = LlavaOneVisionQModel
    if not was_supported:
        auto.SUPPORTED_MODELS.append(model_type)
    try:
        yield LlavaOneVisionQModel
    finally:
        if previous is None:
            auto.MODEL_MAP.pop(model_type, None)
        else:
            auto.MODEL_MAP[model_type] = previous
        if not was_supported:
            auto.SUPPORTED_MODELS.remove(model_type)


def _calibration_batches(processor, calibration: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Build native multimodal conversations for GPTQModel.

    GPTQModel's Qwen-VL adapter owns the processor step and expects chat
    conversations (with ``content`` entries), rather than the already
    tensorized inputs returned by ``prepare_inputs``.  Passing tensorized
    inputs makes its vision extractor index ``message["content"]`` and fails
    before calibration starts.  Keep the calibration contract in the skill
    while letting the backend perform its own image preprocessing.
    """
    data = calibration.get("data")
    if not data:
        return []
    from search.evaluate import load_samples

    rows = load_samples(
        PROJECT_ROOT / data,
        media_root=PROJECT_ROOT / calibration.get("media_root", "data/evaluation"),
    )
    limit = calibration.get("limit", len(rows))
    conversations = []
    for sample in rows[:limit]:
        image_path = (
            PROJECT_ROOT
            / calibration.get("media_root", "data/evaluation")
            / sample["media"]["logical_path"]
        ).resolve()
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": sample["logical_question"]},
                    ],
                }
            ]
        )
    return conversations


def _write_manifest(
    *,
    candidate: dict[str, Any],
    model: dict[str, Any],
    calibration: dict[str, Any],
    module_names: list[str],
    output_dir: Path,
) -> Path:
    manifest = make_manifest(
        model=model,
        config=candidate,
        calibration=calibration,
        module_names=module_names,
        status="exported",
    )
    validate_manifest(manifest)
    if model.get("adapter") in {"llava_onevision", "llava15"}:
        from skills.quantization_evidence import audit_gptq_export

        audit = audit_gptq_export(output_dir, manifest)
        if audit["status"] != "passed":
            (output_dir / "precision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            raise AdapterError("blocked_preflight: exported tensors do not implement the declared precision policy")
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def _export_gptqmodel(*, candidate, model, processor, calibration, output_dir, module_names):
    if model.get("adapter") == "kimi_vl":
        from skills.kimi_vl_quantization import export_multimodal_gptq
        return export_multimodal_gptq(candidate=candidate, model=model, processor=processor,
                                      calibration=calibration, output_dir=output_dir,
                                      module_names=module_names)
    if model.get("adapter") == "deepseek_vl2":
        from skills.deepseek_vl2_quantization import export_multimodal_gptq
        return export_multimodal_gptq(candidate=candidate, model=model, processor=processor,
                                      calibration=calibration, output_dir=output_dir,
                                      module_names=module_names)
    if model.get("adapter") == "internvl35":
        from skills.internvl35_quantization import export_multimodal_gptq
        return export_multimodal_gptq(candidate=candidate, model=model, processor=processor,
                                      calibration=calibration, output_dir=output_dir,
                                      module_names=module_names)
    if model.get("adapter") == "llava15":
        from skills.llava15_quantization import export_multimodal_gptq

        return export_multimodal_gptq(candidate=candidate, model=model, processor=processor,
                                     calibration=calibration, output_dir=output_dir,
                                     module_names=module_names)
    if model.get("adapter") == "llava_onevision" and candidate.get("vision_quantization") == "siglip_gptq_zero_pad_v1":
        import hashlib
        import shutil
        from skills.llava_quantization import export_visual_completion

        cache = PROJECT_ROOT / candidate["export_cache"]
        source = PROJECT_ROOT / candidate["decoder_source_artifact"]
        identity = {"candidate": candidate, "model": model, "calibration": calibration,
                    "implementation_sha256": hashlib.sha256(
                        (PROJECT_ROOT / "skills/llava_quantization.py").read_bytes()).hexdigest()}
        if cache.exists():
            if json.loads((cache / "cache-identity.json").read_text()) != identity:
                raise AdapterError("Visual export cache identity changed; preserve it and use a new cache")
            validate_manifest(json.loads((cache / "manifest.json").read_text()))
        else:
            pending = cache.with_name(cache.name + ".pending")
            if pending.exists():
                raise AdapterError(f"Incomplete visual export retained at {pending}; inspect before recovery")
            export_visual_completion(source=source, destination=pending, processor=processor,
                                     calibration=calibration, model=model, candidate=candidate)
            _write_manifest(candidate=candidate, model=model, calibration=calibration,
                            module_names=module_names, output_dir=pending)
            (pending / "cache-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
            pending.rename(cache)
        shutil.copytree(cache, output_dir)
        return {"manifest": str(output_dir / "manifest.json"), "export_cache": str(cache),
                "vision_quantization": "siglip_gptq_zero_pad_v1"}
    if model.get("adapter") == "llava_onevision":
        from skills.quantization_evidence import require_llava_gptq_support

        require_llava_gptq_support(PROJECT_ROOT / model["path"], candidate["group_size"])
    package = importlib.import_module("gptqmodel")
    GPTQModel = getattr(package, "GPTQModel", None)
    QuantizeConfig = getattr(package, "QuantizeConfig", None)
    if GPTQModel is None or QuantizeConfig is None:
        raise AdapterError("GPTQModel public load/quantize API is unavailable")
    batches = _calibration_batches(processor, calibration)
    if not batches:
        raise AdapterError("GPTQModel requires non-empty multimodal calibration")
    quantize_config = QuantizeConfig(
        bits=candidate["bits"], group_size=candidate["group_size"]
    )
    source = str((PROJECT_ROOT / model["path"]).resolve())
    context = (
        _bind_llava_onevision_gptqmodel()
        if model.get("adapter") == "llava_onevision"
        else nullcontext()
    )
    with context:
        quantized = GPTQModel.load(
            source, quantize_config=quantize_config, device="cuda:0", trust_remote_code=True
        )
        quantized.quantize(batches)
        output_dir.mkdir(parents=True, exist_ok=True)
        quantized.save_quantized(str(output_dir))
        # GPTQModel may serialize a generic tokenizer wrapper whose
        # ``chat_template`` is a list.  LLaVA-OneVision's native processor
        # requires the original Jinja string and image special-token fields;
        # preserve that application-level processor beside the quantized
        # weights so a clean Transformers reload uses the native contract.
        if model.get("adapter") == "llava_onevision" and processor is not None:
            processor.save_pretrained(str(output_dir))
        del quantized
    return {
        "manifest": str(
            _write_manifest(
                candidate=candidate,
                model=model,
                calibration=calibration,
                module_names=module_names,
                output_dir=output_dir,
            )
        )
    }


def _export_llm_compressor(*, candidate, model, processor, calibration, output_dir, module_names):
    package = importlib.import_module("llmcompressor")
    oneshot = getattr(package, "oneshot", None)
    if oneshot is None:
        raise AdapterError("LLM Compressor oneshot API is unavailable")
    # Keep the policy explicit. The modifier's ignore list protects the
    # visual mapper and language head; a backend release that cannot preserve
    # these modules fails closed in the manifest/reload gate.
    recipe = {
        "targets": ["Linear"],
        "scheme": candidate.get("scheme", "W4A16"),
        "ignore": [
            "lm_head",
            "visual.patch_embed",
            "visual.merger",
            "visual.deepstack_merger_list",
        ],
    }
    source = str((PROJECT_ROOT / model["path"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=source,
        dataset=calibration.get("data"),
        recipe=recipe,
        output_dir=str(output_dir),
    )
    return {
        "manifest": str(
            _write_manifest(
                candidate=candidate,
                model=model,
                calibration=calibration,
                module_names=module_names,
                output_dir=output_dir,
            )
        )
    }


def export_candidate(*, candidate, model, processor, calibration, output_dir, module_names=()):
    """Export one candidate using its isolated backend adapter."""

    kwargs = {
        "candidate": candidate,
        "model": model,
        "processor": processor,
        "calibration": calibration,
        "output_dir": Path(output_dir),
        "module_names": list(module_names),
    }
    if candidate["route"] == "gptqmodel_w4a16":
        return _export_gptqmodel(**kwargs)
    if candidate["route"] in {"llm_compressor_w4a16", "llm_compressor_fp8"}:
        return _export_llm_compressor(**kwargs)
    raise AdapterError(f"No validated export adapter for {candidate['route']}")
