"""Sequential native vision/decoder GPTQ using the isolated GPTQModel solver.

Calibration preserves native tiles and the physical parent. Each projection
group observes the already quantized preceding groups; each block propagates
its quantized result to the next. Portable GPTQ-v1 tensors are independently
unpacked before saving. No RTN fallback or borrowed model statistics are used.
"""
import json
from pathlib import Path


def unpack_gptq(tensors):
    """Return [out, in] weights from the checked portable 4-bit format."""
    import torch

    shifts = torch.arange(8, device=tensors["qweight"].device, dtype=torch.int64) * 4
    codes = ((tensors["qweight"].long()[:, None, :] >> shifts[None, :, None]) & 15).flatten(0, 1)
    zeros = ((tensors["qzeros"].long()[:, :, None] >> shifts[None, None, :]) & 15).flatten(1) + 1
    groups = tensors["g_idx"].long()
    if not bool((zeros == 8).all()):
        raise ValueError("InternVL GPTQ requires the symmetric GPTQ-v1 zero convention")
    return (tensors["scales"].float()[groups] * (codes - zeros[groups])).T.contiguous()


def _move(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if value is None or type(value) in (bool, int, float):
        return value
    raise ValueError(f"Unsupported native calibration input: {type(value).__name__}")


def capture_inputs(layer, samples, invoke):
    """Capture the real first-block inputs and stop before decoder allocation."""
    class Captured(Exception):
        pass

    rows = []
    def capture(module, args, kwargs):
        if len(args) != 1:
            raise ValueError("Expected native hidden states as the sole positional input")
        if "past_key_values" in kwargs:
            kwargs = {**kwargs, "past_key_values": None, "use_cache": False}
        rows.append((_move(args[0], "cpu"), _move(kwargs, "cpu")))
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
                raise ValueError("Native calibration did not visit the first block exactly once")
    finally:
        handle.remove()
    return rows


def quantize_blocks(layers, prefix, rows, groups, packed, reports):
    import torch
    from skills.llava_quantization import new_gptq_task, pack_gptq_task

    for index, layer in enumerate(layers):
        device = next(layer.parameters()).device
        for names in groups:
            tasks, handles = {}, []
            try:
                for name in names:
                    linear = layer.get_submodule(name)
                    if not isinstance(linear, torch.nn.Linear):
                        raise ValueError("GPTQ calibration requires a fresh physical Linear")
                    task = new_gptq_task(linear, bits=4, group_size=128)
                    tasks[name] = task
                    def collect(module, args, output, task=task):
                        task.add_batch(args[0].detach(), output.detach())
                    handles.append(linear.register_forward_hook(collect))
                for hidden, kwargs in rows:
                    layer(hidden.to(device), **_move(kwargs, device))
            finally:
                for handle in handles:
                    handle.remove()
            for name, task in tasks.items():
                full_name = f"{prefix}.{index}.{name}"
                tensors, report = pack_gptq_task(task)
                linear = layer.get_submodule(name)
                tensors = {key: value.to(linear.weight.dtype) if value.is_floating_point() else value
                           for key, value in tensors.items()}
                # Calibration propagation uses the exact portable quantized
                # weights. Native reload separately proves actual Marlin use.
                linear.weight.copy_(unpack_gptq(tensors).to(device=device, dtype=linear.weight.dtype))
                packed[full_name] = tensors
                reports.append({"module": full_name, **report, "scale_dtype": str(tensors["scales"].dtype)})
        rows = [(layer(hidden.to(device), **_move(kwargs, device)).detach().cpu(), kwargs)
                for hidden, kwargs in rows]
        print(f"GPTQ {prefix} block {index + 1}/{len(layers)}", flush=True)
    return rows


def quantize_native_modules(native, samples, inputs):
    """Run both native towers, including their actual block-call contracts."""
    import torch

    packed, reports = {}, []
    with torch.inference_mode():
        vision_rows = capture_inputs(native.vision_model.encoder.layers[0], samples,
                                     lambda row: native.extract_feature(inputs(row)["pixel_values"]))
        quantize_blocks(native.vision_model.encoder.layers, "vision_model.encoder.layers", vision_rows,
                        (("attn.qkv",), ("attn.proj",), ("mlp.fc1",), ("mlp.fc2",)), packed, reports)
        del vision_rows
        decoder_rows = capture_inputs(native.language_model.model.layers[0], samples,
                                      lambda row: native.generate(**inputs(row), max_new_tokens=1, do_sample=False))
        quantize_blocks(native.language_model.model.layers, "language_model.model.layers", decoder_rows,
                        (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                         ("self_attn.o_proj",), ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
                        packed, reports)
    return packed, reports


def export_multimodal_gptq(*, candidate, model, processor, calibration, output_dir,
                          module_names, loaded_model=None):
    import torch
    import shutil
    from search.model import PROJECT_ROOT, prepare_inputs
    from search.evaluate import load_samples
    from adapters.internvl35_dense import load_model as load_dense
    from skills.internvl35_artifact import (
        QUANT_METHOD, artifact_config_dict, load_artifact, native_adapter_id, save_tensors,
    )
    from skills.quantization import make_manifest, digest, validate_module_inventory
    from skills.quantization_evidence import audit_gptq_export

    if (candidate.get("route") != "gptqmodel_w4a16" or candidate.get("bits") != 4
            or candidate.get("group_size") != 128
            or candidate.get("vision_quantization") != QUANT_METHOD):
        raise ValueError("InternVL requires the explicit multimodal GPTQ W4A16 Group-128 route")
    source = (PROJECT_ROOT / model["path"]).resolve()
    header = json.loads((source / "config.json").read_text())
    if header.get("quantization_config"):
        raise ValueError("GPTQ requires a fresh Dense or physically pruned BF16 parent")
    adapter_id = native_adapter_id(header)
    if model.get("adapter") != adapter_id:
        raise ValueError("InternVL GPTQ descriptor differs from the source decoder family")
    language_count = header["llm_config"]["num_hidden_layers"] * 7
    vision_count = header["vision_config"]["num_hidden_layers"] * 4
    expected_count = language_count + vision_count
    inventory = validate_module_inventory(module_names, model_adapter=adapter_id)
    if (not inventory["valid"] or inventory["int4_count"] != expected_count or inventory["bf16_count"] != 3
            or inventory["language_linear_count"] != language_count
            or inventory["vision_linear_count"] != vision_count):
        raise ValueError(f"InternVL export requires all {language_count} decoder and {vision_count} vision linears")
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Existing InternVL export must be inspected before replacement")
    rows = load_samples(PROJECT_ROOT / calibration["data"],
                        media_root=PROJECT_ROOT / calibration["media_root"])[:calibration["limit"]]
    if len(rows) != calibration["limit"] or len({row["stable_row_id"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or repeated native calibration inputs")
    own_model = loaded_model is None
    if own_model:
        loader = load_artifact if header.get("a4fvlm_stack_widths") is not None else load_dense
        loaded_model, processor = loader(model)
    native = loaded_model.eval()
    actual_names = [name for name, value in native.named_modules() if isinstance(value, torch.nn.Linear)]
    if set(actual_names) != set(module_names):
        raise ValueError("InternVL loaded parent does not match the declared module inventory")
    if native.config.model_type != "internvl_chat" or native.dtype != torch.bfloat16:
        raise ValueError("InternVL GPTQ requires its native BF16 parent")
    def inputs(sample):
        return prepare_inputs(processor, sample, calibration["media_root"]).to(native.device)

    packed, reports = quantize_native_modules(native, rows, inputs)
    if len(packed) != expected_count or any(row["fallback"] is not None for row in reports):
        raise ValueError("Incomplete native GPTQ calibration or unsupported fallback")
    state = {key: value for key, value in native.state_dict().items()
             if key.rsplit(".", 1)[0] not in packed}
    for name, tensors in packed.items():
        state.update({name + "." + key: value for key, value in tensors.items()})
    config = artifact_config_dict(native.config)
    config["a4fvlm_internvl_gptq"] = {"method": QUANT_METHOD, "format": "gptq_v1"}
    config["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False,
        "sym": True, "checkpoint_format": "gptq", "lm_head": False,
        "modules_in_block_to_quantize": [sorted(packed)],
    }
    destination.mkdir(parents=True, exist_ok=False)
    save_tensors(state, destination)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    processor.save_pretrained(destination)
    receipt = {"method": QUANT_METHOD, "calibration": calibration,
               "sample_ids": [row["stable_row_id"] for row in rows], "sample_count": len(rows),
               "sequential_projection_groups": True, "vision_before_decoder": True,
               "modules": reports, "source_config_digest": digest(header)}
    (destination / "vision-quantization.json").write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = make_manifest(model=model, config=candidate, calibration=calibration,
                             module_names=module_names, status="exported")
    manifest["visual_quantization"] = {"method": QUANT_METHOD,
                                      "artifact_config_digest": digest(config),
                                      "calibration_receipt_digest": digest(receipt)}
    if header.get("a4fvlm_stack_widths") is not None:
        parent = json.loads((source / "stack-parent.json").read_text())
        manifest["stack"] = {"parent_recipe_digest": parent["recipe_digest"],
                             "parent_manifest_digest": digest(parent), "fresh_calibration": True}
        shutil.copyfile(source / "stack-parent.json", destination / "stack-parent.json")
    manifest["artifact_digest"] = digest({key: value for key, value in manifest.items() if key != "artifact_digest"})
    audit = audit_gptq_export(destination, manifest)
    (destination / "precision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "passed":
        raise ValueError("Native InternVL export does not implement the declared precision policy")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(destination / "manifest.json"), "vision_quantization": QUANT_METHOD}
