"""Kimi's exact Linear inventory and native sequential calibration interfaces."""

import json
from pathlib import Path

from skills.kimi_vl_pruning import dimensions, units
from skills.quantization import assignment, digest


def expected_inventory(header):
    if header.get("model_type") != "kimi_vl" or header["text_config"].get("q_lora_rank") is not None:
        raise ValueError("Kimi GPTQ requires its native direct-q MLA topology")
    names = ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2", "language_model.lm_head"]
    dims = dimensions(header["text_config"])
    names += [f"language_model.model.layers.{i}.self_attn.{suffix}"
              for i in range(dims["num_hidden_layers"])
              for suffix in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")]
    for unit in units(dims):
        root = f"language_model.model.layers.{unit['layer']}.mlp"
        branches = ([root] if unit["kind"] == "dense" else [root + ".shared_experts"]
                    if unit["kind"] == "shared" else
                    [f"{root}.experts.{i}" for i in range(unit["experts"])])
        names += [f"{branch}.{suffix}" for branch in branches
                  for suffix in ("gate_proj", "up_proj", "down_proj")]
    names += [f"vision_tower.encoder.blocks.{i}.{suffix}"
              for i in range(header["vision_config"]["num_hidden_layers"])
              for suffix in ("wqkv", "wo", "mlp.fc0", "mlp.fc1")]
    return assignment(names, model_adapter="kimi_vl")


def high_precision_exceptions(header):
    result = {name: {"dtype": "bfloat16", "reason": "preserved_multimodal_interface_or_output_head"}
              for name, precision in expected_inventory(header).items() if precision == "bf16"}
    for unit in units(dimensions(header["text_config"])):
        if unit["kind"] == "routed":
            root = f"language_model.model.layers.{unit['layer']}.mlp.gate"
            result[root + ".weight"] = {"dtype": "bfloat16", "reason": "preserved_native_routing"}
            result[root + ".e_score_correction_bias"] = {
                "dtype": "bfloat16", "reason": "preserved_original_routing_correction"}
    return result


def vision_padding(header):
    original = header["vision_config"]["intermediate_size"]
    padded = ((original + 127) // 128) * 128
    return {"original_width": original, "padded_width": padded, "added_channels": padded - original,
            "layers": header["vision_config"]["num_hidden_layers"],
            "method": "zero_extend_internal_mlp_channels"}


def pad_vision(native):
    """Group alignment affects only internal channels; native image grids stay intact."""
    import torch

    padding = vision_padding(native.config.to_dict())
    original, padded = padding["original_width"], padding["padded_width"]
    blocks = native.vision_tower.encoder.blocks
    for block in blocks:
        if block.mlp.fc0.out_features != original or block.mlp.fc1.in_features != original:
            raise ValueError("Kimi vision padding requires fresh native MLP dimensions")
    if padded == original:
        return padding
    for block in blocks:
        first, second = block.mlp.fc0, block.mlp.fc1
        expand = torch.nn.Linear(first.in_features, padded, bias=first.bias is not None,
                                 device="meta", dtype=first.weight.dtype)
        contract = torch.nn.Linear(padded, second.out_features, bias=second.bias is not None,
                                   device="meta", dtype=second.weight.dtype)
        # Meta reload needs only the padded module shapes; the saved packed or
        # BF16 weights will be strictly assigned afterwards.
        if not first.weight.is_meta:
            with torch.no_grad():
                expand.weight = torch.nn.Parameter(torch.nn.functional.pad(first.weight, (0, 0, 0, padded - original)))
                contract.weight = torch.nn.Parameter(torch.nn.functional.pad(second.weight, (0, padded - original)))
                if first.bias is not None:
                    expand.bias = torch.nn.Parameter(torch.nn.functional.pad(first.bias, (0, padded - original)))
                if second.bias is not None:
                    contract.bias = torch.nn.Parameter(second.bias.detach().clone())
        block.mlp.fc0, block.mlp.fc1 = expand, contract
    return padding


def capture_inputs(layer, samples, invoke, *, vision=False):
    """Capture actual packed MoonViT sequence offsets or cache-free MLA inputs."""
    from skills.internvl35_quantization import _move

    class Captured(Exception):
        pass

    rows = []
    def capture(module, args, kwargs):
        kwargs = dict(kwargs)
        if len(args) != (2 if vision else 1):
            raise ValueError("Unexpected Kimi native block positional arguments")
        if vision:
            if "cu_seqlens" in kwargs:
                raise ValueError("Duplicate Kimi native packed sequence offsets")
            kwargs["cu_seqlens"] = args[1]
        for name in ("past_key_value", "past_key_values"):
            if kwargs.get(name) is not None:
                raise ValueError("Kimi calibration must start without a cache")
        if kwargs.get("use_cache"):
            raise ValueError("Kimi sequential calibration requires use_cache=False")
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
                raise ValueError("Native Kimi calibration must visit the first block exactly once")
    finally:
        handle.remove()
    return rows


def decoder_groups(layer):
    mlp = layer.mlp
    branches = ([f"mlp.experts.{i}" for i in range(len(mlp.experts))]
                + ["mlp.shared_experts"] if hasattr(mlp, "experts") else ["mlp"])
    # Kimi calls kv_b_proj normally, so it observes real sequential activations
    # and is quantized before the output projection.
    return [("self_attn.q_proj", "self_attn.kv_a_proj_with_mqa"), ("self_attn.kv_b_proj",),
            ("self_attn.o_proj",),
            tuple(f"{branch}.{suffix}" for branch in branches for suffix in ("gate_proj", "up_proj")),
            tuple(f"{branch}.down_proj" for branch in branches)]


def quantize_native_modules(native, samples, inputs):
    """Run real Hessian GPTQ with Kimi activations, vision before the decoder."""
    import torch
    from skills.deepseek_vl2_quantization import quantize_blocks

    if native.training:
        raise ValueError("Kimi calibration requires native eval routing")
    policy = expected_inventory(native.config.to_dict())
    actual = {name for name, module in native.named_modules() if isinstance(module, torch.nn.Linear)}
    if actual != set(policy):
        raise ValueError("Kimi GPTQ Linear topology differs from its declared native inventory")
    packed, reports = {}, []
    padding = pad_vision(native)
    with torch.inference_mode():
        def invoke_vision(row):
            batch = inputs(row)
            # The original outer forward casts processor FP32 pixels to the
            # vision tower's dtype before calling this extraction method.
            # Calibration enters extraction directly and must preserve that step.
            native._extract_image_features(batch["pixel_values"].to(native.vision_tower.dtype),
                                           batch["image_grid_hws"])
        blocks = native.vision_tower.encoder.blocks
        vision_rows = capture_inputs(blocks[0], samples, invoke_vision, vision=True)
        # Reuse the low-level Hessian solver/replay, with Kimi's own names,
        # sequence offsets, real activations and padded physical dimensions.
        quantize_blocks(blocks, "vision_tower.encoder.blocks", vision_rows,
                        (("wqkv",), ("wo",), ("mlp.fc0",), ("mlp.fc1",)), packed, reports)
        del vision_rows
        def invoke_decoder(row):
            native(**inputs(row), use_cache=False, output_attentions=False)
        layers = native.language_model.model.layers
        decoder_rows = capture_inputs(layers[0], samples, invoke_decoder)
        quantize_blocks(layers, "language_model.model.layers", decoder_rows,
                        decoder_groups, packed, reports)
    expected = {name for name, precision in policy.items() if precision == "int4_w4a16"}
    if set(packed) != expected or any(row.get("fallback") is not None or row["observed_tokens"] <= 0 for row in reports):
        raise ValueError("Incomplete native Kimi GPTQ calibration; unobserved modules cannot fall back")
    return packed, reports, padding


def validate_loaded_parent(native, header, descriptor, recipe=None):
    """Check actual physical geometry and native BF16 routing before calibration."""
    import torch
    from skills.kimi_vl_artifact import construct_native, router_inventory
    from skills.kimi_vl_pruning import audit_recipe, modules

    dims = dimensions(header["text_config"])
    if (dimensions(native.config.text_config) != dims
            or getattr(native, "structured_pruning_recipe", None) != recipe):
        raise ValueError("Kimi loaded Parent does not match its source topology/recipe")
    if (descriptor.get("attn_implementation") != "eager"
            or any(block.attn_implementation != "eager" for block in native.vision_tower.encoder.blocks)):
        raise ValueError("Kimi GPTQ requires the declared native eager vision attention")
    reference = construct_native(header, descriptor, recipe)
    def geometry(model):
        return {name: {"weight": list(module.weight.shape),
                       "bias": list(module.bias.shape) if module.bias is not None else None}
                for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    actual = geometry(native)
    if actual != geometry(reference) or set(actual) != set(expected_inventory(header)):
        raise ValueError("Kimi loaded Parent Linear geometry differs from its native source")
    if any(value.dtype != torch.bfloat16 for value in native.parameters() if value.is_floating_point()):
        raise ValueError("Kimi Parent parameters and routing corrections must retain native BF16")
    modules(native, dims)
    if recipe is not None:
        audit_recipe(native, recipe)
    return {"linear_count": len(actual), "linear_shapes_digest": digest(actual),
            "recipe_digest": digest(recipe) if recipe is not None else None,
            "routers": router_inventory(native), "router_and_expert_counts_preserved": True}


def export_multimodal_gptq(*, candidate, model, processor, calibration, output_dir,
                          module_names, loaded_model=None):
    """Fresh native Hessian calibration and portable packed export for this Parent."""
    import shutil
    import gptqmodel  # noqa: F401 -- import before loading the complete checkpoint
    from adapters.kimi_vl_dense import PROJECT_ROOT, load_model, native_path, prepare_inputs
    from search.evaluate import load_samples
    from skills.internvl35_artifact import save_tensors
    from skills.kimi_vl_artifact import (
        QUANT_ARCHITECTURE, QUANT_METHOD, read_parent, router_fingerprints, rotary_inventory,
        audit_quantized_artifact, validate_quantized_bindings,
    )
    from skills.quantization import make_manifest

    if (model.get("adapter") != "kimi_vl" or candidate.get("route") != "gptqmodel_w4a16"
            or candidate.get("bits") != 4 or candidate.get("group_size") != 128
            or candidate.get("vision_quantization") != QUANT_METHOD):
        raise ValueError("Kimi requires its explicit native multimodal W4A16 Group-128 GPTQ route")
    source = native_path(model)
    header = json.loads((source / "config.json").read_text())
    if header.get("quantization_config"):
        raise ValueError("Kimi GPTQ requires a fresh Dense or physically pruned BF16 Parent")
    policy = expected_inventory(header)
    if set(module_names) != set(policy):
        raise ValueError("Kimi export requires its complete native Linear inventory")
    parent_data = read_parent(source) if header.get("a4fvlm_stack_widths") is not None else None
    recipe, parent = (parent_data[1], parent_data[2]) if parent_data else (None, None)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Existing Kimi export must be inspected before replacement")
    rows = load_samples(PROJECT_ROOT / calibration["data"],
                        media_root=PROJECT_ROOT / calibration["media_root"])[:calibration["limit"]]
    if not rows or len(rows) != calibration["limit"] or len({row["stable_row_id"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or repeated native Kimi calibration inputs")
    if loaded_model is None:
        loaded_model, processor = load_model(model)
    if (processor is None or not hasattr(processor, "_agent4vlm_model_config")
            or native_path(processor._agent4vlm_model_config) != source):
        raise ValueError("Kimi quantization processor must authenticate the actual source Parent")
    native = loaded_model.eval()
    topology = validate_loaded_parent(native, header, model, recipe)
    routers = router_fingerprints(native)
    rotary_buffers = rotary_inventory(native)
    def inputs(row):
        batch = prepare_inputs(processor, row, calibration["media_root"]).to(native.device)
        if batch["input_ids"].shape[1] > header["text_config"]["max_position_embeddings"]:
            raise ValueError("Kimi calibration exceeds native context; truncation is forbidden")
        return batch
    packed, reports, padding = quantize_native_modules(native, rows, inputs)
    if router_fingerprints(native) != routers or rotary_inventory(native) != rotary_buffers:
        raise ValueError("Kimi GPTQ changed native router parameters or rotary frequencies")
    state = {key: value for key, value in native.state_dict().items() if key.rsplit(".", 1)[0] not in packed}
    for name, tensors in packed.items():
        state.update({name + "." + key: value for key, value in tensors.items()})
    # Preserve source topology verbatim; module-only vision padding is explicit.
    config = dict(header)
    if parent is None:
        config["architectures"] = [QUANT_ARCHITECTURE]
    config["a4fvlm_kimi_gptq"] = {"method": QUANT_METHOD, "format": "gptq_v1", "vision_padding": padding}
    config["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False,
        "sym": True, "checkpoint_format": "gptq", "lm_head": False,
        "modules_in_block_to_quantize": [sorted(packed)],
    }
    destination.mkdir(parents=True, exist_ok=False)
    save_tensors(state, destination)
    processor.save_pretrained(destination)
    for name in (*model["native_source_sha256"], "tiktoken.model"):
        shutil.copyfile(source / name, destination / name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "source-config.json").write_text(json.dumps(header, indent=2) + "\n")
    receipt = {"method": QUANT_METHOD, "calibration": calibration,
               "sample_ids": [row["stable_row_id"] for row in rows], "sample_count": len(rows),
               "sequential_projection_groups": True, "vision_before_decoder": True,
               "native_routing": True, "modules": reports, "vision_padding": padding,
               "source_config_digest": digest(header), "source_topology": topology,
               "routers": routers, "rotary_buffers": rotary_buffers}
    (destination / "vision-quantization.json").write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = make_manifest(model=model, config={**candidate, "calibration": calibration}, calibration=calibration,
                             module_names=module_names, status="exported")
    manifest["visual_quantization"] = {"method": QUANT_METHOD, "artifact_config_digest": digest(config),
                                       "source_config_digest": digest(header),
                                       "calibration_receipt_digest": digest(receipt)}
    manifest["high_precision_exceptions"] = high_precision_exceptions(header)
    if parent is not None:
        manifest["stack"] = {"parent_manifest_digest": digest(parent), "parent_recipe_digest": digest(recipe),
                             "fresh_calibration": True, "fresh_shape_reparse": True}
        for name in ("stack-parent.json", "sp-recipe.json"):
            shutil.copyfile(source / name, destination / name)
    manifest["artifact_digest"] = digest({key: value for key, value in manifest.items() if key != "artifact_digest"})
    validate_quantized_bindings(destination, config, manifest)
    audit = audit_quantized_artifact(destination, manifest)
    (destination / "precision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "passed":
        raise ValueError("Kimi GPTQ export failed its actual packed precision audit")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(destination / "manifest.json"), "vision_quantization": QUANT_METHOD}
