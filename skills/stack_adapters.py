"""Backend-specific adapters for Stack materialization."""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

from skills.quantization import digest, make_manifest, validate_manifest
from skills.quantization_adapters import _calibration_batches
from skills.stack_model import bind_gptqmodel_loader, stack_architecture_for_config


class StackAdapterError(RuntimeError):
    """A Stack backend cannot consume the exact physical parent."""


def export_pruned_gptqmodel(
    *,
    candidate: dict[str, Any],
    parent_dir: str | Path,
    processor: Any,
    calibration: dict[str, Any],
    output_dir: str | Path,
    model: dict[str, Any],
    module_names: list[str],
    recipe_digest: str,
) -> dict[str, str]:
    """Quantize a physically compacted parent with a fresh calibration pass."""

    if candidate.get("route") != "gptqmodel_w4a16":
        raise StackAdapterError("Only GPTQModel W4A16 has a Stack adapter")
    if model.get("adapter") == "deepseek_vl2":
        from skills.deepseek_vl2_artifact import read_parent
        from skills.deepseek_vl2_quantization import export_multimodal_gptq
        _, _, parent = read_parent(parent_dir)
        if parent.get("recipe_digest") != recipe_digest:
            raise StackAdapterError("DeepSeek physical parent differs from selected recipe")
        importlib.import_module("gptqmodel")
        return export_multimodal_gptq(candidate=candidate.get("candidate_config", candidate),
                                      model={**model, "path": str(parent_dir)}, processor=processor,
                                      calibration=calibration, output_dir=output_dir,
                                      module_names=module_names)
    if model.get("adapter") == "internvl35":
        from skills.internvl35_quantization import export_multimodal_gptq
        parent = json.loads((Path(parent_dir) / "stack-parent.json").read_text())
        if parent.get("recipe_digest") != recipe_digest:
            raise StackAdapterError("InternVL physical parent differs from selected recipe")
        # Backend initialization can retain dependency import frames. Complete
        # it before entering the exporter frame that owns the physical model,
        # so releasing that model really frees its GPU weights before reload.
        importlib.import_module("gptqmodel")
        return export_multimodal_gptq(candidate=candidate.get("candidate_config", candidate),
                                      model={**model, "path": str(parent_dir)}, processor=processor,
                                      calibration=calibration, output_dir=output_dir,
                                      module_names=module_names)
    parent = Path(parent_dir)
    model_adapter = model.get("adapter", "qwen3_vl")
    padded_vision = (model_adapter == "llava_onevision"
                     and candidate.get("candidate_config", candidate).get("vision_quantization") == "siglip_gptq_zero_pad_v1")
    if not (parent / "config.json").is_file() or not (parent / "stack-parent.json").is_file():
        raise StackAdapterError("SP parent is missing its topology manifest")
    if model_adapter == "llava_onevision" and not padded_vision:
        from skills.quantization_evidence import require_llava_gptq_support

        require_llava_gptq_support(parent, candidate.get("group_size"))
    parent_metadata = json.loads((parent / "stack-parent.json").read_text(encoding="utf-8"))
    group_size = candidate.get("group_size")
    widths = parent_metadata.get("widths")
    if (
        type(group_size) is not int
        or group_size < 1
        or not isinstance(widths, list)
        or any(type(width) is not int or width % group_size for width in widths)
    ):
        raise StackAdapterError("SP parent widths are not legal for the GPTQ group size")
    package = importlib.import_module("gptqmodel")
    GPTQModel = getattr(package, "GPTQModel", None)
    QuantizeConfig = getattr(package, "QuantizeConfig", None)
    if GPTQModel is None or QuantizeConfig is None:
        raise StackAdapterError("GPTQModel public API is unavailable")
    batches = _calibration_batches(processor, calibration)
    if not batches:
        raise StackAdapterError("Stack GPTQ requires non-empty calibration")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Stack quantized artifact already exists: {destination}")
    final_destination = destination
    if padded_vision:
        destination = destination.with_name(destination.name + ".decoder")
    config = QuantizeConfig(bits=candidate["bits"], group_size=candidate["group_size"])
    if model_adapter == "llava15":
        from skills.llava15_quantization import export_multimodal_gptq
        from skills.stack_model import create_pruned_model_class

        export_multimodal_gptq(
            candidate={**candidate, "vision_quantization": candidate.get("candidate_config", candidate).get("vision_quantization")},
            model={**model, "path": str(parent.resolve())}, processor=processor,
            calibration=calibration, output_dir=destination, module_names=module_names,
            loader_override=create_pruned_model_class("llava15"))
    else:
        with bind_gptqmodel_loader(model_adapter=model_adapter):
            quantized = GPTQModel.load(
                str(parent.resolve()),
                quantize_config=config,
                device="cuda:0",
                trust_remote_code=True,
            )
            quantized.quantize(batches)
            destination.mkdir(parents=True)
            quantized.save_quantized(str(destination))
            del quantized
    if model_adapter == "llava_onevision":
        processor.save_pretrained(destination)
    # Preserve the physical topology contract even if the backend rewrites the
    # model architecture list while exporting its quantization metadata.
    parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    child_config_path = destination / "config.json"
    if not child_config_path.is_file():
        raise StackAdapterError("GPTQModel did not write a model config")
    child_config = json.loads(child_config_path.read_text(encoding="utf-8"))
    for key in (
        "a4fvlm_stack_widths",
        "a4fvlm_stack_recipe_digest",
        "a4fvlm_stack_physical_compaction",
    ):
        if key not in parent_config:
            raise StackAdapterError(f"SP parent config is missing {key}")
        child_config[key] = parent_config[key]
    child_config["architectures"] = [stack_architecture_for_config(parent_config)]
    child_config_path.write_text(
        json.dumps(child_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if padded_vision:
        from skills.llava_quantization import export_visual_completion

        partial = make_manifest(model={**model, "path": str(parent.resolve())}, config=candidate,
                                calibration=calibration, module_names=module_names, status="decoder_exported")
        (destination / "manifest.json").write_text(json.dumps(partial, indent=2) + "\n")
        export_visual_completion(source=destination, destination=final_destination, processor=processor,
                                 calibration=calibration, model=partial["model"],
                                 candidate={**candidate, "decoder_source_manifest_digest": partial["artifact_digest"]})
        shutil.rmtree(destination)
        destination = final_destination
    manifest = make_manifest(
        model={**model, "path": str(parent.resolve())},
        config=candidate,
        calibration=calibration,
        module_names=module_names,
        status="exported",
    )
    manifest["stack"] = {
        "parent_recipe_digest": recipe_digest,
        "fresh_shape_reparse": True,
        "fresh_calibration": True,
        "parent_manifest_digest": digest(parent_metadata),
    }
    if padded_vision or model_adapter == "llava15":
        manifest["visual_quantization"] = {
            "method": "clip_gptq_v1" if model_adapter == "llava15" else "siglip_gptq_zero_pad_v1",
            "calibration_receipt_digest": digest(json.loads((destination / "vision-quantization.json").read_text())),
            "artifact_config_digest": digest(json.loads((destination / "config.json").read_text())),
        }
    manifest["artifact_digest"] = digest(
        {key: value for key, value in manifest.items() if key != "artifact_digest"}
    )
    validate_manifest(manifest)
    if model_adapter in {"llava_onevision", "llava15"}:
        from skills.quantization_evidence import audit_gptq_export

        audit = audit_gptq_export(destination, manifest)
        (destination / "precision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if audit["status"] != "passed":
            raise StackAdapterError("blocked_preflight: Stack export violates the declared precision policy")
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # GPTQModel does not promise to copy application topology metadata. Keep
    # the width vector beside the exported child so a clean Stack reload can
    # reconstruct the exact non-uniform MLP layout.
    shutil.copyfile(parent / "stack-parent.json", destination / "stack-parent.json")
    return {"manifest": str(manifest_path)}


__all__ = ["StackAdapterError", "export_pruned_gptqmodel"]
