"""LLaVA-1.5 Llama and CLIP GPTQ export in an isolated backend Worker."""
import json
import shutil
from pathlib import Path


def export_multimodal_gptq(*, candidate, model, processor, calibration, output_dir,
                          module_names, loader_override=None):
    """Freshly calibrate both towers, then audit the actual portable tensors."""
    import torch
    from PIL import Image
    from gptqmodel import GPTQModel, QuantizeConfig
    from safetensors.torch import load_file, save_file
    from transformers import CLIPVisionConfig, CLIPVisionModel
    from huggingface_hub import split_torch_state_dict_into_shards
    from search.model import PROJECT_ROOT
    from search.evaluate import load_samples
    from skills.quantization import make_manifest
    from skills.quantization_adapters import (
        _bind_llava_onevision_gptqmodel, _calibration_batches, _write_manifest,
    )
    from skills.quantization_evidence import audit_gptq_export
    from skills.llava_quantization import new_gptq_task, pack_gptq_task

    if candidate.get("vision_quantization") != "clip_gptq_v1":
        raise ValueError("LLaVA-1.5 requires the explicit CLIP GPTQ route")
    if candidate["bits"] != 4 or candidate["group_size"] != 128:
        raise ValueError("LLaVA-1.5 export supports W4A16 Group-128")
    source = PROJECT_ROOT / model["path"]
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "llava" or config["vision_config"]["model_type"] != "clip_vision_model":
        raise ValueError("Expected the native LLaVA/CLIP checkpoint")
    output_dir = Path(output_dir)
    decoder_dir = output_dir.with_name(output_dir.name + ".decoder")
    if output_dir.exists() or decoder_dir.exists():
        raise ValueError("Existing export must be inspected for recovery before replacement")
    manifest = make_manifest(model=model, config=candidate, calibration=calibration,
                             module_names=module_names, status="exported")
    batches = _calibration_batches(processor, calibration)
    if len(batches) != calibration["limit"]:
        raise ValueError("Incomplete LLaVA-1.5 calibration inputs")
    with _bind_llava_onevision_gptqmodel(loader_override=loader_override, model_adapter="llava15"):
        quantized = GPTQModel.load(str(source), device="cuda:0", trust_remote_code=True,
                                  quantize_config=QuantizeConfig(bits=4, group_size=128))
        quantized.quantize(batches)
        quantized.save_quantized(str(decoder_dir))
        del quantized
    del batches
    torch.cuda.empty_cache()
    language_manifest = {**manifest, "module_assignment": {
        name: precision for name, precision in manifest["module_assignment"].items()
        if name.startswith("model.language_model.") and precision == "int4_w4a16"}}
    language_audit = audit_gptq_export(decoder_dir, language_manifest)
    if language_audit["status"] != "passed" or language_audit["verified_int4_modules"] != 224:
        raise ValueError("LLaVA-1.5 decoder export does not contain all 224 GPTQ linears")
    config = json.loads((decoder_dir / "config.json").read_text())
    tensors = {}
    for path in sorted(decoder_dir.glob("*.safetensors")):
        tensors.update(load_file(str(path), device="cpu"))
    vision_config = CLIPVisionConfig(**config["vision_config"])
    if (vision_config.hidden_size, vision_config.intermediate_size, vision_config.num_hidden_layers) != (1024, 4096, 24):
        raise ValueError("Unsupported CLIP dimensions")
    vision = CLIPVisionModel(vision_config).to(dtype=torch.float16)
    prefix = "model.vision_tower."
    vision.load_state_dict({key[len(prefix):]: tensor for key, tensor in tensors.items()
                           if key.startswith(prefix)}, strict=True)
    vision.to("cuda:0").eval()
    tasks, handles = {}, []
    for name, module in vision.named_modules():
        full_name = prefix + name
        if not isinstance(module, torch.nn.Linear) or manifest["module_assignment"].get(full_name) != "int4_w4a16":
            continue
        task = new_gptq_task(module, bits=4, group_size=128)
        tasks[full_name] = task

        def capture(module, inputs, output, task=task):
            task.add_batch(inputs[0].detach(), output.detach())

        handles.append(module.register_forward_hook(capture))
    if len(tasks) != 144:
        raise ValueError(f"Expected 144 CLIP GPTQ linears, found {len(tasks)}")
    rows = load_samples(PROJECT_ROOT / calibration["data"],
                        media_root=PROJECT_ROOT / calibration["media_root"])[:calibration["limit"]]
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                path = PROJECT_ROOT / calibration["media_root"] / row["media"]["logical_path"]
                with Image.open(path) as source_image, source_image.convert("RGB") as image:
                    pixels = processor.image_processor(images=[image], return_tensors="pt")["pixel_values"]
                if tuple(pixels.shape) != (1, 3, 336, 336):
                    raise ValueError("CLIP calibration changed the native image geometry")
                vision(pixels.to(device="cuda:0", dtype=torch.float16))
                if index % 16 == 0:
                    print(f"CLIP calibration {index}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    reports = []
    for index, (name, task) in enumerate(tasks.items(), 1):
        packed, report = pack_gptq_task(task)
        tensors.pop(name + ".weight", None)
        tensors.pop(name + ".bias", None)
        tensors.update({name + "." + key: value for key, value in packed.items()})
        reports.append({"module": name, **report})
        print(f"CLIP GPTQ {index}/{len(tasks)}: {name}", flush=True)
    del vision, tasks
    for key, value in tensors.items():
        if value.is_floating_point() and not key.endswith(".scales"):
            tensors[key] = value.to(torch.bfloat16)
    output_dir.mkdir(parents=True, exist_ok=False)
    shard = split_torch_state_dict_into_shards(tensors, max_shard_size="2GB")
    for filename, names in shard.filename_to_tensors.items():
        save_file({name: tensors[name].contiguous() for name in names}, str(output_dir / filename),
                  metadata={"format": "pt"})
    if shard.is_sharded:
        (output_dir / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": shard.metadata, "weight_map": shard.tensor_to_filename}, indent=2) + "\n")
    config["a4fvlm_clip_gptq"] = {"method": "clip_gptq_v1", "layers": 24, "padding": False}
    config["quantization_config"].update(block_name_to_quantize="model", modules_in_block_to_quantize=[
        [name for name, precision in manifest["module_assignment"].items() if precision == "int4_w4a16"]])
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    processor.save_pretrained(output_dir)
    report = {"method": "clip_gptq_v1", "padding": False,
              "sample_ids": [row["stable_row_id"] for row in rows], "calibration": calibration,
              "decoder_audit": language_audit, "modules": reports}
    (output_dir / "vision-quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest_path = _write_manifest(candidate=candidate, model=model, calibration=calibration,
                                    module_names=module_names, output_dir=output_dir)
    shutil.rmtree(decoder_dir)
    return {"manifest": str(manifest_path), "vision_quantization": "clip_gptq_v1"}
