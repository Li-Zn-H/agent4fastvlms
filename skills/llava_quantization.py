"""LLaVA-specific SigLIP padding and GPTQ primitives, with lazy backend imports.

Padding extends only an FFN's internal channels. The vision-token geometry,
projector shape, native processor and language-model dimensions are unchanged.
"""

import math
import hashlib
import json
from pathlib import Path


def padded_width(width, group_size=128):
    if type(width) is not int or width < 1 or type(group_size) is not int or group_size < 1:
        raise ValueError("width and group_size must be positive integers")
    alignment = math.lcm(group_size, 64)
    return (width + alignment - 1) // alignment * alignment


def pad_siglip_mlp(mlp, width):
    """Extend fc1 rows/bias and fc2 columns with zeros, retaining the function."""
    import torch

    first, second = mlp.fc1, mlp.fc2
    original = first.out_features
    if second.in_features != original or width < original:
        raise ValueError("Inconsistent FFN dimensions or requested shrinking")
    if width == original:
        return {"original_width": original, "padded_width": width, "added_channels": 0}
    fc1 = torch.nn.Linear(first.in_features, width, bias=first.bias is not None,
                          device=first.weight.device, dtype=first.weight.dtype)
    fc2 = torch.nn.Linear(width, second.out_features, bias=second.bias is not None,
                          device=second.weight.device, dtype=second.weight.dtype)
    with torch.no_grad():
        fc1.weight.zero_()
        fc1.weight[:original].copy_(first.weight)
        fc2.weight.zero_()
        fc2.weight[:, :original].copy_(second.weight)
        if fc1.bias is not None:
            fc1.bias.zero_()
            fc1.bias[:original].copy_(first.bias)
        if fc2.bias is not None:
            fc2.bias.copy_(second.bias)
    mlp.fc1, mlp.fc2 = fc1, fc2
    return {"original_width": original, "padded_width": width,
            "added_channels": width - original}


def new_gptq_task(linear, *, bits=4, group_size=128):
    """Create a real Hessian-calibrated task; never substitute RTN fallback."""
    from gptqmodel.quantization import GPTQ, QuantizeConfig

    config = QuantizeConfig(bits=bits, group_size=group_size, desc_act=False,
                            act_group_aware=False, fallback=None)
    task = GPTQ(linear, qcfg=config)
    task.quantizer.configure(perchannel=True)
    return task


def pack_gptq_task(task):
    """Solve a calibrated module and emit portable GPTQ-v1 packed tensors."""
    import copy
    import torch
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.model import convert_gptq_v2_to_v1_format_module

    if task.nsamples <= 0:
        raise ValueError("GPTQ packing requires observed calibration activations")
    weight, scales, zeros, indices, seconds, loss, damp, samples = task.quantize()
    if not torch.isfinite(weight).all():
        raise ValueError("GPTQ produced nonfinite weights")
    linear = copy.deepcopy(task.module).cpu()
    linear.weight.data.copy_(weight.cpu())
    config = task.qcfg
    packed = TorchLinear(bits=config.bits, group_size=config.group_size, sym=config.sym,
                         desc_act=config.desc_act, in_features=linear.in_features,
                         out_features=linear.out_features, bias=linear.bias is not None)
    packed.pack(linear, scales.cpu(), zeros.cpu(), indices.cpu())
    # The writer uses standard GPTQ-v1, matching vLLM and the decoder export.
    convert_gptq_v2_to_v1_format_module(packed, config)
    tensors = {key: value.detach().cpu().contiguous() for key, value in packed.state_dict().items()}
    # Independently unpack the portable format to catch a wrong zero convention
    # before a numerically finite but incorrect checkpoint can be exported.
    shifts = torch.arange(8, dtype=torch.int64) * 4
    codes = ((tensors["qweight"].long()[:, None, :] >> shifts[None, :, None]) & 15).reshape(
        linear.in_features, linear.out_features)
    zero_codes = ((tensors["qzeros"].long()[:, :, None] >> shifts[None, None, :]) & 15).reshape(
        -1, linear.out_features) + 1
    group = tensors["g_idx"].long()
    unpacked = (tensors["scales"].float()[group] * (codes - zero_codes[group])).T
    reference = weight.detach().cpu().float()
    torch.testing.assert_close(unpacked, reference, rtol=0.002, atol=0.002)
    report = {"method": "gptq_hessian", "bits": config.bits,
              "group_size": config.group_size, "samples": int(samples),
              "seconds": float(seconds), "average_loss": float(loss),
              "damp_percent": float(damp), "fallback": None,
              "in_features": linear.in_features, "out_features": linear.out_features,
              "packing_max_abs_error": float((unpacked-reference).abs().max())}
    task.free()
    return tensors, report


def export_visual_completion(*, source, destination, processor, calibration, model, candidate):
    """Reuse authenticated decoder GPTQ weights and freshly calibrate SigLIP.

    The result is a NEW artifact. No quality or performance result is copied
    from the decoder-only source. Calibration uses the same held-out inputs.
    """
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from huggingface_hub import split_torch_state_dict_into_shards
    from search.model import PROJECT_ROOT
    from skills.quantization import digest, validate_manifest
    from skills.quantization_evidence import audit_gptq_export

    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    validate_manifest(manifest)
    if (manifest["artifact_digest"] != candidate["decoder_source_manifest_digest"]
            or manifest["model"] != model or manifest["calibration"] != calibration
            or manifest["quantization"]["bits"] != candidate["bits"]
            or manifest["quantization"]["group_size"] != candidate["group_size"]):
        raise ValueError("Decoder export source does not match model/calibration/quantization")
    language_manifest = {**manifest, "module_assignment": {
        name: precision for name, precision in manifest["module_assignment"].items()
        if name.startswith("model.language_model.") and precision == "int4_w4a16"}}
    if audit_gptq_export(source, language_manifest)["status"] != "passed":
        raise ValueError("Decoder source has no matching actual GPTQ tensors")
    config = json.loads((source / "config.json").read_text())
    tensors = {}
    for path in sorted(source.glob("*.safetensors")):
        tensors.update(load_file(str(path), device="cpu"))
    vision_config = SiglipVisionConfig(**config["vision_config"])
    vision = SiglipVisionModel(vision_config).to(dtype=torch.float16)
    prefix = "model.vision_tower."
    vision.load_state_dict({key[len(prefix):]: value for key, value in tensors.items()
                           if key.startswith(prefix)}, strict=True)
    original_width = vision_config.intermediate_size
    width = padded_width(original_width, candidate["group_size"])
    padding = [pad_siglip_mlp(layer.mlp, width) for layer in vision.encoder.layers]
    vision.config.intermediate_size = width
    vision.to("cuda:0").eval()
    tasks, handles = {}, []
    for name, layer in vision.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        full_name = prefix + name
        if manifest["module_assignment"].get(full_name) != "int4_w4a16":
            continue
        task = new_gptq_task(layer, bits=candidate["bits"], group_size=candidate["group_size"])
        tasks[full_name] = task

        def capture(module, args, output, task=task):
            task.add_batch(args[0].detach(), output.detach())

        handles.append(layer.register_forward_hook(capture))
    if len(tasks) != 156:
        raise ValueError(f"Unexpected SigLIP quantization scope: {len(tasks)} modules")
    data_path = PROJECT_ROOT / calibration["data"]
    samples = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    samples = samples[:calibration["limit"]]
    if len(samples) != calibration["limit"]:
        raise ValueError("Incomplete held-out visual calibration")
    try:
        with torch.inference_mode():
            for index, sample in enumerate(samples):
                path = PROJECT_ROOT / calibration["media_root"] / sample["media"]["logical_path"]
                with Image.open(path) as source_image:
                    image = source_image.convert("RGB")
                try:
                    # Native any-resolution crops; neither image resizing nor
                    # evaluation answers are introduced by the calibration code.
                    pixels = processor.image_processor(images=[image], return_tensors="pt")["pixel_values"]
                    pixels = pixels.flatten(0, 1) if pixels.ndim == 5 else pixels
                    vision(pixels.to(device="cuda:0", dtype=torch.float16))
                finally:
                    image.close()
                if (index + 1) % 16 == 0:
                    print(f"SigLIP calibration {index + 1}/{len(samples)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    reports = []
    for index, (name, task) in enumerate(tasks.items()):
        packed, report = pack_gptq_task(task)
        for key in list(tensors):
            if key in (name + ".weight", name + ".bias"):
                del tensors[key]
        tensors.update({name + "." + key: value for key, value in packed.items()})
        reports.append({"module": name, **report})
        print(f"SigLIP GPTQ {index + 1}/{len(tasks)}: {name}", flush=True)
    del vision, tasks
    # Preserve the declared non-quantized families in actual BF16 storage.
    for key, value in tensors.items():
        if value.is_floating_point() and not key.endswith(".scales"):
            tensors[key] = value.to(torch.bfloat16)
    destination.mkdir(parents=True, exist_ok=False)
    shard = split_torch_state_dict_into_shards(tensors, max_shard_size="2GB")
    for filename, names in shard.filename_to_tensors.items():
        save_file({name: tensors[name].contiguous() for name in names}, str(destination / filename),
                  metadata={"format": "pt"})
    if shard.is_sharded:
        (destination / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": shard.metadata, "weight_map": shard.tensor_to_filename}, indent=2) + "\n")
    config["vision_config"]["intermediate_size"] = width
    config["a4fvlm_siglip_padding"] = {"method": "zero_channels_v1", "original_width": original_width,
                                      "padded_width": width, "layers": len(padding)}
    config["quantization_config"].update(
        block_name_to_quantize="model",
        modules_in_block_to_quantize=[[name for name, precision in manifest["module_assignment"].items()
                                      if precision == "int4_w4a16"]],
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    processor.save_pretrained(destination)
    report = {"method": "siglip_gptq_zero_pad_v1", "decoder_source": str(source),
              "decoder_source_manifest_digest": manifest["artifact_digest"],
              "calibration_digest": digest(calibration),
              "calibration_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "sample_ids": [row["stable_row_id"] for row in samples],
              "padding": config["a4fvlm_siglip_padding"], "modules": reports}
    (destination / "vision-quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
