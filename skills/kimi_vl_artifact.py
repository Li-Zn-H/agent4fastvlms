"""Export and strictly reload a physical Kimi MoE Parent with native sources."""

import copy
import json
from pathlib import Path
import shutil

from adapters.kimi_vl_dense import PROJECT_ROOT, native_path
from skills.kimi_vl_pruning import (
    apply_recipe, audit_recipe, dimensions, units, validate_recipe,
)
from skills.quantization import digest

STACK_ARCHITECTURE = "A4FVLMPhysicalKimiVLForConditionalGeneration"
QUANT_ARCHITECTURE = "A4FVLMQuantizedKimiVLForConditionalGeneration"
QUANT_METHOD = "kimi_vl_multimodal_gptq_v1"


def validate_physical_config(header, recipe):
    dims = dimensions(header["text_config"])
    validate_recipe(recipe, dims)
    if (header.get("model_type") != "kimi_vl"
            or header.get("architectures") != [STACK_ARCHITECTURE]
            or header.get("a4fvlm_stack_physical_compaction") is not True
            or header.get("a4fvlm_stack_recipe_digest") != digest(recipe)
            or header.get("a4fvlm_stack_widths") != recipe["widths"]
            or header.get("a4fvlm_stack_units") != units(dims)):
        raise ValueError("Kimi physical topology is not bound to its recipe")


def router_inventory(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.named_parameters() if ".mlp.gate." in name}


def router_fingerprints(model):
    """Bind exact native routing values as well as their shapes and precision."""
    import hashlib
    import torch
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in model.named_parameters() if ".mlp.gate." in name}


def rotary_inventory(model):
    """Nonpersistent frequencies need their original precision on recipe replay."""
    import hashlib
    import torch
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in model.named_buffers() if name.endswith(".rotary_emb.inv_freq")}


def replay_rotary_precision(model, inventory):
    import torch
    names = {name for name, _ in model.named_buffers() if name.endswith(".rotary_emb.inv_freq")}
    if names != set(inventory):
        raise ValueError("Kimi derived rotary inventory differs from its native topology")
    dtypes = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
    for name, row in inventory.items():
        module_name, attribute = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        if row.get("dtype") not in dtypes:
            raise ValueError("Unsupported Kimi native rotary precision")
        setattr(module, attribute, getattr(module, attribute).to(dtype=dtypes[row["dtype"]]))
        # Previously cached cos/sin tables must be rebuilt from these exact
        # native frequencies, at the normal next-forward sequence length.
        module.max_seq_len_cached = None
    if rotary_inventory(model) != inventory:
        raise ValueError("Kimi derived frequencies differ from the authenticated native constructor")


def construct_native(header, descriptor, recipe=None, *, padded=False):
    """Derive shapes from the authenticated native class without allocating weights."""
    from accelerate import init_empty_weights
    from adapters.kimi_vl_dense import native_model_class
    cls = native_model_class(descriptor)
    config = cls.config_class(**copy.deepcopy(header))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    with init_empty_weights():
        model = cls(config)
        if recipe is not None:
            apply_recipe(model, recipe)
        if padded:
            from skills.kimi_vl_quantization import pad_vision
            pad_vision(model)
    return model


def write_parent(model, processor, recipe, output_dir):
    from skills.internvl35_artifact import save_tensors

    if processor is None or not hasattr(processor, "_agent4vlm_model_config"):
        raise ValueError("Kimi Parent requires its authenticated native processor")
    source_config = processor._agent4vlm_model_config
    source = native_path(source_config)
    if recipe != getattr(model, "structured_pruning_recipe", None):
        raise ValueError("Kimi Parent must contain the actually applied physical recipe")
    audit = audit_recipe(model, recipe)
    config = copy.deepcopy(model.config)
    config.architectures = [STACK_ARCHITECTURE]
    config.a4fvlm_stack_widths = recipe["widths"]
    config.a4fvlm_stack_units = units(recipe["dimensions"])
    config.a4fvlm_stack_recipe_digest = digest(recipe)
    config.a4fvlm_stack_physical_compaction = True
    header = json.loads(config.to_json_string(use_diff=False))
    validate_physical_config(header, recipe)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_tensors(model.state_dict(), output)
    processor.save_pretrained(output)
    # Keep the original code, Instruct template, image settings, tokenizer and
    # EOS configuration byte-identical. Processor serialization may rewrite them.
    for name in (*source_config["native_source_sha256"], "tiktoken.model"):
        shutil.copyfile(source / name, output / name)
    (output / "config.json").write_text(json.dumps(header, indent=2) + "\n")
    (output / "sp-recipe.json").write_text(json.dumps(recipe) + "\n")
    manifest = {"schema": "a4fvlm.stack-parent.v1", "architecture": STACK_ARCHITECTURE,
                "recipe_digest": digest(recipe), "widths": recipe["widths"],
                "units": units(recipe["dimensions"]), "physical_compaction": True,
                "model_config_digest": digest(header), "mechanism": audit,
                "native_source_sha256": source_config["native_source_sha256"],
                "source_model": source_config, "source_model_digest": digest(source_config),
                "routers": router_inventory(model), "rotary_buffers": rotary_inventory(model)}
    (output / "stack-parent.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_parent(path):
    path = Path(path)
    header = json.loads((path / "config.json").read_text())
    recipe = json.loads((path / "sp-recipe.json").read_text())
    validate_physical_config(header, recipe)
    manifest = json.loads((path / "stack-parent.json").read_text())
    source_header = header
    if header.get("quantization_config"):
        source_header = json.loads((path / "source-config.json").read_text())
        validate_physical_config(source_header, recipe)
        if (source_header.get("quantization_config")
                or header != {**source_header, "quantization_config": header["quantization_config"],
                              "a4fvlm_kimi_gptq": header.get("a4fvlm_kimi_gptq")}):
            raise ValueError("Kimi quantized descendant changed its physical Parent configuration")
    if (manifest.get("schema") != "a4fvlm.stack-parent.v1"
            or manifest.get("architecture") != STACK_ARCHITECTURE
            or manifest.get("recipe_digest") != digest(recipe)
            or manifest.get("widths") != recipe["widths"]
            or manifest.get("units") != units(recipe["dimensions"])
            or manifest.get("physical_compaction") is not True
            or manifest.get("model_config_digest") != digest(source_header)
            or manifest.get("source_model_digest") != digest(manifest.get("source_model"))
            or manifest.get("native_source_sha256") != manifest.get("source_model", {}).get("native_source_sha256")):
        raise ValueError("Kimi Parent manifest does not match its physical/source bindings")
    native_path({**manifest["source_model"], "path": str(path)})
    return header, recipe, manifest


def load_physical_model(descriptor):
    """Construct original modules, replay physical widths, then assign exact tensors."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, GenerationConfig
    from adapters.kimi_vl_dense import native_model_class

    if descriptor.get("dtype") != "bfloat16" or descriptor.get("attn_implementation") != "eager":
        raise ValueError("Kimi physical reload requires native BF16 eager MLA")
    path = native_path(descriptor)
    header, recipe, manifest = read_parent(path)
    if header.get("quantization_config"):
        raise ValueError("Quantized Kimi requires a separately verified packed-weight loader")
    if descriptor["native_source_sha256"] != manifest["native_source_sha256"]:
        raise ValueError("Kimi Parent source differs from its descriptor")
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    cls = native_model_class(descriptor)
    with init_empty_weights():
        model = cls(config)
        apply_recipe(model, recipe)
    state = {}
    for filename in sorted(path.glob("*.safetensors")):
        shard = load_file(str(filename), device="cpu")
        if state.keys() & shard.keys():
            raise ValueError("Duplicate Kimi Parent tensor keys")
        state.update(shard)
    model.load_state_dict(state, strict=True, assign=True)
    if manifest.get("rotary_buffers") is not None:
        replay_rotary_precision(model, manifest["rotary_buffers"])
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Kimi Parent left unmaterialized tensors")
    if router_inventory(model) != manifest["routers"]:
        raise ValueError("Kimi Parent changed router shape or native precision")
    if any(value.is_floating_point() and value.dtype != torch.bfloat16 for value in model.parameters()):
        raise ValueError("Kimi Parent changed native BF16 parameter precision")
    model.generation_config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
    model._agent4vlm_router_dtypes = {name: row["dtype"] for name, row in manifest["routers"].items()
                                    if name.endswith("e_score_correction_bias")}
    model.to(descriptor["device"]).eval()
    audit_recipe(model, recipe)
    return model


def validate_quantized_bindings(path, header, manifest):
    """Reject missing calibration, modified native topology or stale SP ancestry."""
    from skills.quantization import validate_manifest
    from skills.kimi_vl_quantization import expected_inventory, high_precision_exceptions, vision_padding
    path = Path(path)
    validate_manifest(manifest)
    policy = expected_inventory(header)
    selected = {name for name, precision in policy.items() if precision == "int4_w4a16"}
    source = json.loads((path / "source-config.json").read_text())
    quant = {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False,
             "sym": True, "checkpoint_format": "gptq", "lm_head": False,
             "modules_in_block_to_quantize": [sorted(selected)]}
    extra = {"method": QUANT_METHOD, "format": "gptq_v1", "vision_padding": vision_padding(source)}
    physical = source.get("a4fvlm_stack_widths") is not None
    expected_header = {**source, "quantization_config": quant, "a4fvlm_kimi_gptq": extra}
    if not physical:
        expected_header["architectures"] = [QUANT_ARCHITECTURE]
    if (source.get("quantization_config") or source.get("model_type") != "kimi_vl"
            or header != expected_header or manifest.get("module_assignment") != policy
            or manifest.get("high_precision_exceptions") != high_precision_exceptions(source)
            or manifest.get("status") != "exported"
            or manifest.get("model", {}).get("adapter") != "kimi_vl"
            or manifest.get("model_digest") != digest(manifest.get("model"))
            or manifest.get("calibration_digest") != digest(manifest.get("calibration"))):
        raise ValueError("Kimi quantization policy/source topology binding mismatch")
    native_path({**manifest["model"], "path": str(path)})
    receipt = json.loads((path / "vision-quantization.json").read_text())
    visual = manifest.get("visual_quantization", {})
    if (visual.get("method") != QUANT_METHOD or visual.get("artifact_config_digest") != digest(header)
            or visual.get("source_config_digest") != digest(source)
            or visual.get("calibration_receipt_digest") != digest(receipt)
            or receipt.get("source_config_digest") != digest(source)):
        raise ValueError("Kimi artifact/calibration receipt binding mismatch")
    samples, reports = receipt.get("sample_ids", []), receipt.get("modules", [])
    topology = receipt.get("source_topology", {})
    routers = receipt.get("routers", {})
    router_names = {f"language_model.model.layers.{u['layer']}.mlp.gate.{suffix}"
                    for u in units(dimensions(source["text_config"])) if u["kind"] == "routed"
                    for suffix in ("weight", "e_score_correction_bias")}
    rotary_names = {f"language_model.model.layers.{i}.self_attn.rotary_emb.inv_freq"
                    for i in range(source["text_config"]["num_hidden_layers"])}
    if (receipt.get("method") != QUANT_METHOD or receipt.get("calibration") != manifest["calibration"]
            or receipt.get("vision_padding") != extra["vision_padding"]
            or receipt.get("sequential_projection_groups") is not True
            or receipt.get("vision_before_decoder") is not True or receipt.get("native_routing") is not True
            or receipt.get("sample_count", 0) <= 0
            or receipt.get("sample_count") != manifest["calibration"].get("limit")
            or len(samples) != receipt["sample_count"] or len(set(samples)) != len(samples)
            or len(reports) != len(selected) or {r.get("module") for r in reports} != selected
            or any(r.get("method") != "gptq_hessian" or r.get("fallback") is not None
                   or r.get("samples", 0) <= 0 or r.get("observed_tokens", 0) <= 0
                   or r.get("bits") != 4 or r.get("group_size") != 128
                   or r.get("packing_validation") != "exact_codes_zeros_groups_and_fp32_dequantization"
                   for r in reports)
            or topology.get("linear_count") != len(policy)
            or topology.get("router_and_expert_counts_preserved") is not True
            or set(routers) != router_names
            or set(receipt.get("rotary_buffers", {})) != rotary_names
            or {name: {k: row[k] for k in ("shape", "dtype")} for name, row in routers.items()}
               != topology.get("routers")):
        raise ValueError("Kimi requires complete observed native GPTQ calibration and unchanged routing")
    if physical:
        _, recipe, parent = read_parent(path)
        stack = manifest.get("stack", {})
        if (stack.get("parent_manifest_digest") != digest(parent)
                or stack.get("parent_recipe_digest") != digest(recipe)
                or stack.get("fresh_calibration") is not True or stack.get("fresh_shape_reparse") is not True
                or receipt["source_config_digest"] != parent["model_config_digest"]
                or topology.get("recipe_digest") != digest(recipe)
                or parent.get("rotary_buffers") is not None
                   and parent["rotary_buffers"] != receipt["rotary_buffers"]):
            raise ValueError("Kimi quantization is not bound to its freshly calibrated physical Parent")
    elif (manifest.get("stack") or topology.get("recipe_digest") is not None
          or any((path / name).exists() for name in ("stack-parent.json", "sp-recipe.json"))):
        raise ValueError("Independent Kimi quantization has unexpected SP ancestry")


def audit_quantized_artifact(path, manifest):
    """Audit actual packed geometry, Group-128 contents and exact BF16 routers."""
    import hashlib
    import math
    import torch
    from safetensors import safe_open
    from skills.quantization_evidence import audit_gptq_export

    path = Path(path)
    audit = audit_gptq_export(path, manifest)
    source = json.loads((path / "source-config.json").read_text())
    recipe = read_parent(path)[1] if source.get("a4fvlm_stack_widths") is not None else None
    descriptor = {**manifest["model"], "path": str(path)}
    model = construct_native(source, descriptor, recipe, padded=True)
    receipt = json.loads((path / "vision-quantization.json").read_text())
    replay_rotary_precision(model, receipt["rotary_buffers"])
    metadata, locations = {}, {}
    for filename in sorted(path.glob("*.safetensors")):
        with safe_open(str(filename), framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in metadata:
                    audit["errors"].append({"error": "duplicate_tensor", "tensor": name})
                value = shard.get_slice(name)
                metadata[name] = {"dtype": value.get_dtype(), "shape": value.get_shape()}
                locations[name] = filename
    expected = {name: {"shape": list(value.shape), "dtype": "BF16" if value.is_floating_point() else None}
                for name, value in model.state_dict().items()}
    verified = 0
    for name, module in model.named_modules():
        if manifest["module_assignment"].get(name) != "int4_w4a16":
            continue
        expected.pop(name + ".weight")
        size, width = module.in_features, module.out_features
        groups = math.ceil(size / 128)
        for key, shape, dtype in (("qweight", [size // 8, width], "I32"),
                                  ("qzeros", [groups, width // 8], "I32"),
                                  ("scales", [groups, width], "BF16"), ("g_idx", [size], "I32")):
            expected[name + "." + key] = {"shape": shape, "dtype": dtype}
        for suffix in ("qzeros", "g_idx", "scales"):
            key = name + "." + suffix
            if key not in locations:
                continue
            with safe_open(str(locations[key]), framework="pt", device="cpu") as shard:
                value = shard.get_tensor(key)
            # A symmetric GPTQ group with an all-zero weight slice can have a
            # zero scale.  That is a valid packed group (all codes dequantize
            # to zero); requiring strictly positive scales incorrectly rejects
            # such groups, which occur in padded/vision channels.  Negative or
            # non-finite scales remain invalid because they change the signed
            # dequantization grid.
            valid = (bool((value == 0x77777777).all()) if suffix == "qzeros" else
                     torch.equal(value, torch.arange(size, dtype=torch.int32, device=value.device) // 128)
                     if suffix == "g_idx" else
                     bool(torch.isfinite(value).all() and (value >= 0).all()))
            if not valid:
                audit["errors"].append({"tensor": key, "error": "invalid_symmetric_group128_packing"})
        verified += 1
    for name in expected.keys() | metadata.keys():
        want, actual = expected.get(name), metadata.get(name)
        if want is None or actual is None or want["shape"] != actual["shape"] or (
                want["dtype"] is not None and want["dtype"] != actual["dtype"]):
            audit["errors"].append({"tensor": name, "expected": want, "observed": actual})
    for name, row in receipt.get("routers", {}).items():
        if name not in locations:
            continue
        with safe_open(str(locations[name]), framework="pt", device="cpu") as shard:
            value = shard.get_tensor(name)
        fingerprint = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if (row.get("sha256") != fingerprint or row.get("shape") != list(value.shape)
                or row.get("dtype") != str(value.dtype)):
            audit["errors"].append({"tensor": name, "error": "changed_native_router"})
    audit["verified_packed_geometries"] = verified
    audit["verified_router_tensors"] = len(receipt.get("routers", {}))
    audit["status"] = "passed" if not audit["errors"] else "blocked_preflight"
    return audit


def _load_packed_model(descriptor, kernel_factory, initialize):
    """Strict shared replay; the public loader always chooses native CUDA Marlin."""
    import torch
    from safetensors.torch import load_file
    from transformers import GenerationConfig
    path = native_path(descriptor)
    header = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    validate_quantized_bindings(path, header, manifest)
    if descriptor["native_source_sha256"] != manifest["model"]["native_source_sha256"]:
        raise ValueError("Kimi quantized source differs from its authenticated descriptor")
    audit = audit_quantized_artifact(path, manifest)
    if audit["status"] != "passed":
        raise ValueError(f"Kimi artifact precision/geometry audit failed: {audit['errors'][:3]}")
    source = json.loads((path / "source-config.json").read_text())
    recipe = read_parent(path)[1] if source.get("a4fvlm_stack_widths") is not None else None
    model = construct_native(source, descriptor, recipe, padded=True)
    kernels = []
    for name, module in list(model.named_modules()):
        if manifest["module_assignment"].get(name) != "int4_w4a16":
            continue
        kernel = kernel_factory(bits=4, group_size=128, desc_act=False, sym=True,
                                in_features=module.in_features, out_features=module.out_features,
                                bias=module.bias is not None, dtype=torch.bfloat16)
        owner, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(owner), attribute, kernel)
        kernels.append((name, kernel))
    state = {}
    for filename in sorted(path.glob("*.safetensors")):
        shard = load_file(str(filename), device="cpu")
        if state.keys() & shard.keys():
            raise ValueError("Duplicate Kimi packed artifact tensor keys")
        state.update(shard)
    model.load_state_dict(state, strict=True, assign=True)
    receipt = json.loads((path / "vision-quantization.json").read_text())
    replay_rotary_precision(model, receipt["rotary_buffers"])
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Kimi quantized reload left unmaterialized tensors")
    for name, value in state.items():
        if value.is_floating_point() and value.dtype != torch.bfloat16:
            raise ValueError(f"Kimi packed artifact changed BF16 precision: {name}")
    if router_fingerprints(model) != json.loads((path / "vision-quantization.json").read_text())["routers"]:
        raise ValueError("Kimi quantized reload changed native routing values")
    model.generation_config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
    model.to(descriptor["device"]).eval()
    for name, kernel in kernels:
        initialize(name, kernel)
    model._agent4vlm_quantized_modules = [name for name, _ in kernels]
    model._agent4vlm_vision_padding = header["a4fvlm_kimi_gptq"]["vision_padding"]
    return model


def load_quantized_model(descriptor):
    """Load portable GPTQ weights with CUDA Marlin; never silently fall back."""
    if (descriptor.get("dtype") != "bfloat16" or not descriptor.get("device", "").startswith("cuda:")
            or descriptor.get("attn_implementation") != "eager"):
        raise ValueError("Kimi quantized reload requires native BF16 CUDA Marlin and eager MLA")
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    def initialize(name, kernel):
        kernel.post_init()
        if kernel._marlin_tile_padding is not None:
            raise ValueError(f"Unexpected Kimi runtime Marlin padding: {name}")
    return _load_packed_model(descriptor, MarlinLinear, initialize)


def load_artifact(descriptor):
    from adapters.kimi_vl_dense import load_processor
    header = json.loads((native_path(descriptor) / "config.json").read_text())
    model = load_quantized_model(descriptor) if header.get("quantization_config") else load_physical_model(descriptor)
    return model, load_processor(descriptor)
