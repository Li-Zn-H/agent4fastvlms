"""Native InternVL physical topology and portable GPTQ artifact loading.

No optional framework is imported by catalog/planning callers. Quantized CUDA
reload uses GPTQModel Marlin only in its isolated interpreter. CPU reference
loading is explicit and never supplies a CUDA mechanism or performance pass.
"""
import copy
import json
from pathlib import Path

from skills.quantization import digest

STACK_ARCHITECTURE = "A4FVLMSPQuantizedInternVLChatModel"
QUANT_METHOD = "internvl_multimodal_gptq_v1"


def native_adapter_id(config):
    """Validate the supported native decoder without importing model backends."""
    if config.get("llm_config", {}).get("model_type") != "qwen3":
        raise ValueError("Unsupported native InternVL decoder family")
    return "internvl35"


def artifact_config_dict(config):
    # The checkpoint's legacy composite to_dict omits Transformers 5's dtype
    # normalization; restore the normal serializer step without editing it.
    value = config.to_dict()
    config.dict_dtype_to_str(value)
    return value


def validate_artifact_config(config):
    if config.get("model_type") != "internvl_chat":
        raise ValueError("Expected a native InternVL artifact")
    widths = config.get("a4fvlm_stack_widths")
    quant = config.get("a4fvlm_internvl_gptq")
    architecture = config.get("architectures")
    if widths is None and quant is None:
        raise ValueError("InternVL artifact has neither physical SP nor quantization metadata")
    if widths is not None:
        text = config["llm_config"]
        if (architecture != [STACK_ARCHITECTURE]
                or config.get("a4fvlm_stack_physical_compaction") is not True
                or not config.get("a4fvlm_stack_recipe_digest")
                or not isinstance(widths, list) or len(widths) != text["num_hidden_layers"]
                or any(type(width) is not int or width <= 0 or width > text["intermediate_size"]
                       or width % 128 for width in widths)):
            raise ValueError("Invalid InternVL physical width metadata")
    elif architecture != ["InternVLChatModel"]:
        raise ValueError("Independent InternVL quantization must retain the native architecture")
    if quant is not None:
        cfg = config.get("quantization_config", {})
        if (quant != {"method": QUANT_METHOD, "format": "gptq_v1"}
                or any(cfg.get(key) != value for key, value in {
                    "quant_method": "gptq", "bits": 4, "group_size": 128,
                    "desc_act": False, "sym": True,
                }.items())):
            raise ValueError("Unsupported InternVL quantization metadata")
    elif config.get("quantization_config"):
        raise ValueError("Unbound InternVL quantization configuration")
    return widths


def artifact_model_class(path):
    from adapters.internvl35_dense import native_model_class

    class InternVLArtifactModel(native_model_class(path)):
        def __init__(self, config, **kwargs):
            widths = validate_artifact_config(config.to_dict())
            super().__init__(config, **kwargs)
            if widths is not None:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as MLP
                for layer, width in zip(self.language_model.model.layers, widths, strict=True):
                    layer_config = copy.deepcopy(config.llm_config)
                    layer_config.intermediate_size = width
                    layer.mlp = MLP(layer_config)

    return InternVLArtifactModel


def save_tensors(tensors, destination):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    destination = Path(destination)
    shards = split_torch_state_dict_into_shards(tensors, max_shard_size="2GB")
    for filename, names in shards.filename_to_tensors.items():
        save_file({key: tensors[key].detach().cpu().contiguous() for key in names},
                  str(destination / filename), metadata={"format": "pt"})
    if shards.is_sharded:
        (destination / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": shards.metadata, "weight_map": shards.tensor_to_filename,
        }, indent=2) + "\n")


def write_parent(model, processor, recipe, output_dir):
    from skills.structured_pruning import audit_recipe, validate_recipe

    text = model.config.llm_config
    dimensions = {key: getattr(text, key) for key in
                  ("num_hidden_layers", "hidden_size", "intermediate_size")}
    validate_recipe(recipe, dimensions)
    if recipe != getattr(model, "structured_pruning_recipe", None):
        raise ValueError("InternVL parent requires the actually applied SP recipe")
    audit_recipe(model, recipe)
    destination = Path(output_dir)
    config = copy.deepcopy(model.config)
    config.architectures = [STACK_ARCHITECTURE]
    config.a4fvlm_stack_widths = recipe["widths"]
    config.a4fvlm_stack_recipe_digest = digest(recipe)
    config.a4fvlm_stack_physical_compaction = True
    validate_artifact_config(config.to_dict())
    destination.mkdir(parents=True, exist_ok=False)
    save_tensors(model.state_dict(), destination)
    (destination / "config.json").write_text(json.dumps(artifact_config_dict(config), indent=2) + "\n")
    processor.save_pretrained(destination)
    manifest = {"schema": "a4fvlm.stack-parent.v1", "architecture": STACK_ARCHITECTURE,
                "recipe_digest": digest(recipe), "widths": recipe["widths"],
                "physical_compaction": True,
                "model_config_digest": digest(json.loads((destination / "config.json").read_text()))}
    (destination / "stack-parent.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _validate_parent(path, config):
    if config.get("a4fvlm_stack_widths") is None:
        if (path / "stack-parent.json").exists():
            raise ValueError("Independent artifact has unexpected SP parent metadata")
        return
    parent = json.loads((path / "stack-parent.json").read_text())
    if (parent.get("schema") != "a4fvlm.stack-parent.v1"
            or parent.get("architecture") != STACK_ARCHITECTURE
            or parent.get("physical_compaction") is not True
            or parent.get("widths") != config["a4fvlm_stack_widths"]
            or parent.get("recipe_digest") != config["a4fvlm_stack_recipe_digest"]):
        raise ValueError("InternVL parent manifest does not match the artifact")
    if not config.get("a4fvlm_internvl_gptq") and parent.get("model_config_digest") != digest(config):
        raise ValueError("InternVL physical parent config changed")


def validate_quantized_bindings(path, config, manifest):
    """Bind the reload to its config, full calibration and physical Parent."""
    from skills.quantization import validate_manifest

    path = Path(path)
    validate_artifact_config(config)
    validate_manifest(manifest)
    _validate_parent(path, config)
    visual = manifest.get("visual_quantization", {})
    receipt = json.loads((path / "vision-quantization.json").read_text())
    if (manifest.get("status") != "exported"
            or manifest.get("model", {}).get("adapter") != native_adapter_id(config)
            or visual.get("method") != QUANT_METHOD
            or visual.get("artifact_config_digest") != digest(config)
            or visual.get("calibration_receipt_digest") != digest(receipt)
            or manifest.get("model_digest") != digest(manifest["model"])
            or manifest.get("calibration_digest") != digest(manifest["calibration"])):
        raise ValueError("InternVL quantization config/calibration binding mismatch")
    samples = receipt.get("sample_ids", [])
    modules = receipt.get("modules", [])
    assigned = {name for name, precision in manifest["module_assignment"].items()
                if precision == "int4_w4a16"}
    if (receipt.get("method") != QUANT_METHOD
            or receipt.get("calibration") != manifest["calibration"]
            or receipt.get("sequential_projection_groups") is not True
            or receipt.get("vision_before_decoder") is not True
            or receipt.get("sample_count") != manifest["calibration"].get("limit")
            or len(samples) != receipt.get("sample_count") or len(set(samples)) != len(samples)
            or not modules or len(modules) != len(assigned)
            or {row.get("module") for row in modules} != assigned
            or any(row.get("method") != "gptq_hessian" or row.get("fallback") is not None
                   or row.get("samples", 0) <= 0 or row.get("bits") != 4
                   or row.get("group_size") != 128 for row in modules)):
        raise ValueError("InternVL requires complete sequential calibration without fallback")
    if config.get("a4fvlm_stack_widths") is not None:
        parent = json.loads((path / "stack-parent.json").read_text())
        stack = manifest.get("stack", {})
        if (stack.get("parent_manifest_digest") != digest(parent)
                or stack.get("parent_recipe_digest") != parent["recipe_digest"]
                or stack.get("fresh_calibration") is not True
                or receipt.get("source_config_digest") != parent.get("model_config_digest")):
            raise ValueError("InternVL quantization is not bound to its freshly calibrated Parent")
    elif manifest.get("stack"):
        raise ValueError("Independent InternVL quantization has unexpected Stack provenance")


def load_artifact(config):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from adapters.internvl35_dense import PROJECT_ROOT, load_processor
    from skills.quantization import assignment
    from skills.quantization_evidence import audit_gptq_export

    path = (PROJECT_ROOT / config["path"]).resolve()
    header = json.loads((path / "config.json").read_text())
    validate_artifact_config(header)
    _validate_parent(path, header)
    if config.get("attn_implementation", "sdpa") not in {"eager", "sdpa"}:
        raise ValueError("Native InternVL artifact requires eager/SDPA decoder attention")
    processor = load_processor(config, artifact=True)
    native = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    quantized = header.get("a4fvlm_internvl_gptq") is not None
    if quantized:
        manifest = json.loads((path / "manifest.json").read_text())
        validate_quantized_bindings(path, header, manifest)
        if audit_gptq_export(path, manifest)["status"] != "passed":
            raise ValueError("InternVL artifact precision audit failed")
    with init_empty_weights():
        model = artifact_model_class(path)(native, use_flash_attn=False)
    dtype = getattr(torch, config["dtype"])
    model.to(dtype=dtype)
    kernels = []
    if quantized:
        if not config.get("device", "cuda:0").startswith("cuda:"):
            raise ValueError("InternVL quantized reload requires its CUDA Marlin worker")
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
        modules = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)]
        policy = assignment([name for name, _ in modules], model_adapter=native_adapter_id(header))
        if policy != manifest["module_assignment"]:
            raise ValueError("InternVL artifact precision policy does not match its topology")
        for name, module in modules:
            if policy[name] != "int4_w4a16":
                continue
            kernel = MarlinLinear(bits=4, group_size=128, desc_act=False, sym=True,
                                  in_features=module.in_features, out_features=module.out_features,
                                  bias=module.bias is not None, dtype=dtype)
            owner, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(owner), attribute, kernel)
            kernels.append((name, kernel))
    state = {}
    for filename in sorted(path.glob("*.safetensors")):
        shard = load_file(str(filename), device="cpu")
        if set(state) & set(shard):
            raise ValueError("Duplicate InternVL artifact tensor keys")
        state.update(shard)
    model.load_state_dict(state, strict=True, assign=True)
    if any(value.is_meta for value in model.parameters()):
        raise ValueError("Unmaterialized InternVL artifact parameters")
    model.to(device=config["device"], dtype=dtype).eval()
    for name, kernel in kernels:
        if not bool((kernel.qzeros == 0x77777777).all()):
            raise ValueError(f"Expected symmetric GPTQ-v1 zero convention: {name}")
        kernel.post_init()
        if kernel._marlin_tile_padding is not None:
            raise ValueError(f"Unexpected runtime padding in native InternVL: {name}")
    model.language_model.set_attn_implementation(config.get("attn_implementation", "sdpa"))
    model.img_context_token_id = processor.image_token_id
    separator = processor.conversation_factory(processor.config.template).sep.strip()
    model.language_model.generation_config.eos_token_id = processor.tokenizer.convert_tokens_to_ids(separator)
    model._agent4vlm_quantized_modules = [name for name, _ in kernels]
    return model, processor
