"""Read exported tensor metadata without loading a model or allocating a GPU."""

from pathlib import Path
import json


def require_llava_gptq_support(checkpoint_dir, group_size):
    """Fail before backend allocation when this physical shape has no path."""
    checkpoint = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    report = llava_gptq_preflight(checkpoint, group_size)
    if report["status"] == "blocked_preflight":
        raise ValueError("blocked_preflight: " + "; ".join(report["blockers"]))


def llava_gptq_preflight(checkpoint, group_size, *, padding=False):
    """Check the pinned SigLIP GPTQ path's actual alignment requirements.

    vLLM 0.26 SiglipMLP disables GPTQ for FFNs not divisible by 64. The
    The original route cannot pad those FFNs; the opt-in visual GPTQ route
    declares zero-extension explicitly. An unaligned route is an
    implementation/kernel block, never a quality rejection.
    """
    vision = checkpoint.get("vision_config", {})
    hidden, intermediate = vision.get("hidden_size"), vision.get("intermediate_size")
    original = intermediate
    if padding and type(intermediate) is int and type(group_size) is int and group_size > 0:
        from skills.llava_quantization import padded_width

        intermediate = padded_width(intermediate, group_size)
    blockers = []
    if type(hidden) is not int or type(intermediate) is not int:
        blockers.append("missing_siglip_dimensions")
    else:
        if hidden % 64 or intermediate % 64:
            blockers.append("vllm_siglip_mlp_gptq_requires_dimensions_divisible_by_64")
        if type(group_size) is not int or group_size < 1 or hidden % group_size or intermediate % group_size:
            blockers.append("siglip_gptq_group_alignment_requires_unimplemented_padding")
    return {"status": "blocked_preflight" if blockers else "ready_for_probe",
            "vision_hidden_size": hidden, "vision_intermediate_size": intermediate,
            "original_vision_intermediate_size": original, "padding_enabled": padding,
            "group_size": group_size, "blockers": blockers}


def audit_gptq_export(root, manifest):
    """Compare declared precision to real safetensors keys and storage dtypes.

    Intended module assignment is not proof that a quantizer visited a layer.
    This audit is used for the LLaVA port; it does not relabel legacy receipts.
    """
    from safetensors import safe_open

    root = Path(root)
    metadata = {}
    files = sorted(root.glob("*.safetensors"))
    for path in files:
        with safe_open(str(path), framework="numpy") as handle:
            for key in handle.keys():
                value = handle.get_slice(key)
                metadata[key] = {"dtype": value.get_dtype(), "shape": value.get_shape()}
    errors, quantized, preserved = [], 0, 0
    for module, precision in manifest.get("module_assignment", {}).items():
        if precision == "int4_w4a16":
            required = {name: metadata.get(module + "." + name)
                        for name in ("qweight", "qzeros", "scales", "g_idx")}
            valid = (all(required.values())
                     and required["qweight"]["dtype"] == "I32"
                     and required["qzeros"]["dtype"] == "I32"
                     and required["g_idx"]["dtype"] == "I32"
                     and required["scales"]["dtype"] in {"F16", "BF16"}
                     and module + ".weight" not in metadata)
            quantized += int(valid)
            if not valid:
                errors.append({"module": module, "expected": precision,
                               "missing": [key for key, value in required.items() if value is None],
                               "dense_weight": metadata.get(module + ".weight")})
        elif precision == "bf16":
            tensor = metadata.get(module + ".weight")
            valid = tensor is not None and tensor["dtype"] == "BF16"
            preserved += int(valid)
            if not valid:
                errors.append({"module": module, "expected": precision, "dense_weight": tensor})
        else:
            errors.append({"module": module, "expected": precision, "error": "unsupported_precision"})
    if not files or not manifest.get("module_assignment"):
        errors.append({"error": "missing_tensor_or_assignment_evidence"})
    return {"status": "passed" if not errors else "blocked_preflight",
            "artifact": str(root), "manifest_digest": manifest.get("artifact_digest"),
            "tensor_files": [str(path) for path in files],
            "declared_int4_modules": sum(v == "int4_w4a16" for v in manifest.get("module_assignment", {}).values()),
            "verified_int4_modules": quantized, "verified_bf16_modules": preserved,
            "errors": errors}


def llava15_gptq_preflight(checkpoint, group_size):
    """The CLIP path uses its original aligned dimensions, without padding."""
    vision = checkpoint.get("vision_config", {})
    blockers = []
    if checkpoint.get("model_type") != "llava" or vision.get("model_type") != "clip_vision_model":
        blockers.append("expected_llava_clip_architecture")
    if group_size != 128 or type(group_size) is not int:
        blockers.append("clip_gptq_requires_group_128")
    expected = {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
                "image_size": 336, "patch_size": 14}
    if any(type(vision.get(key)) is not int or vision[key] != value for key, value in expected.items()):
        blockers.append("unsupported_clip_topology")
    return {"status": "blocked_preflight" if blockers else "ready_for_probe",
            "vision_model": "clip", "padding_enabled": False, "group_size": group_size,
            "blockers": blockers}
