"""Framework-independent quantization candidate contracts.

The quantization skill intentionally starts from the Dense model.  It compares
several backend candidates before any structured-pruning or Stack integration.
An SP adapter may be added later, but no recipe is part of this contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


ROUTE_GPTQMODEL_W4A16 = "gptqmodel_w4a16"
ROUTE_LLM_COMPRESSOR_W4A16 = "llm_compressor_w4a16"
ROUTE_LLM_COMPRESSOR_FP8 = "llm_compressor_fp8"
ROUTE_TORCHAO_INT4 = "torchao_int4"
ROUTE_AUTOROUND_W4A16 = "autoround_w4a16"
SUPPORTED_ROUTES = (
    ROUTE_GPTQMODEL_W4A16,
    ROUTE_LLM_COMPRESSOR_W4A16,
    ROUTE_LLM_COMPRESSOR_FP8,
    ROUTE_TORCHAO_INT4,
    ROUTE_AUTOROUND_W4A16,
)
MODULE_PRECISION_INT4 = "int4_w4a16"
MODULE_PRECISION_BF16 = "bf16"
QUANTIZATION_STAGE_ORDER = ("prepare", "calibrate", "export", "manifest", "reload")

# These names intentionally describe the model topology, rather than a
# particular quantizer's module-tree implementation.  The runtime probe checks
# that the selected backend really honours this policy before accepting it.
LANGUAGE_LINEAR = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
VISION_LINEAR = re.compile(
    r"(?:^|\.)visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)
MAPPER = re.compile(
    r"(?:^|\.)visual\.(?:merger|deepstack_merger_list)(?:\.|$)"
)
LLAVA_VISION_LINEAR = re.compile(
    r"(?:^|\.)vision_tower\.(?:vision_model\.)?encoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))$"
)
LLAVA_MAPPER = re.compile(
    r"(?:^|\.)multi_modal_projector\.(?:linear_[12]|linear_[12]\.\w+)$"
)
INTERNVL_LANGUAGE_LINEAR = re.compile(
    r"^language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
INTERNVL_VISION_LINEAR = re.compile(
    r"^vision_model\.encoder\.layers\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))$"
)
INTERNVL_MAPPER = re.compile(r"^mlp1\.(?:1|3)$")
DEEPSEEK_LANGUAGE_LINEAR = re.compile(
    r"^language\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|kv_a_proj_with_mqa|o_proj)|"
    r"mlp\.(?:(?:experts\.\d+|shared_experts)\.)?(?:gate_proj|up_proj|down_proj))$"
)
DEEPSEEK_VISION_LINEAR = re.compile(
    r"^vision\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))$"
)
DEEPSEEK_MAPPER = re.compile(r"^projector\.layers\.(?:0|2)$")
PRESERVE = re.compile(
    r"(?:^|\.)"
    r"(?:lm_head|embed_tokens|visual\.patch_embed|visual\.pos_embed|"
    r"visual\.rotary_pos_emb|visual\.merger|visual\.deepstack_merger_list)"
    r"(?:\.|$)"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    """Return a stable short digest for manifests and cross-stage matching."""

    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:20]


FRAMEWORK_CANDIDATES: dict[str, dict[str, Any]] = {
    ROUTE_GPTQMODEL_W4A16: {
        "framework": "GPTQModel",
        "imports": ["gptqmodel"],
        "runtime_imports": ["vllm"],
        "runtime": "vllm_gptq_marlin",
        "format": "gptq",
        "calibration_required": True,
        "requirements": "requirements-quantization-gptqmodel.txt",
    },
    ROUTE_LLM_COMPRESSOR_W4A16: {
        "framework": "LLM Compressor",
        "imports": ["llmcompressor"],
        "runtime_imports": ["vllm"],
        "runtime": "vllm_compressed_tensors",
        "format": "compressed-tensors",
        "calibration_required": True,
        "requirements": "requirements-quantization-llmcompressor.txt",
    },
    ROUTE_LLM_COMPRESSOR_FP8: {
        "framework": "LLM Compressor",
        "imports": ["llmcompressor"],
        "runtime_imports": ["vllm"],
        "runtime": "vllm_fp8",
        "format": "compressed-tensors",
        "calibration_required": False,
        "requirements": "requirements-quantization-llmcompressor.txt",
    },
    ROUTE_TORCHAO_INT4: {
        "framework": "torchao",
        "imports": ["torchao"],
        "runtime_imports": ["vllm"],
        "runtime": "vllm_torchao",
        "format": "torchao",
        "calibration_required": False,
        "requirements": "requirements-quantization-torchao.txt",
    },
    ROUTE_AUTOROUND_W4A16: {
        "framework": "AutoRound",
        "imports": ["auto_round"],
        "runtime_imports": ["vllm"],
        "runtime": "vllm_awq_or_gptq",
        "format": "auto-round",
        "calibration_required": True,
        "requirements": "requirements-quantization-autoround.txt",
    },
}


def candidate_defaults(route: str) -> dict[str, Any]:
    """Return a copy of the static metadata for a candidate route."""

    if route not in FRAMEWORK_CANDIDATES:
        raise ValueError(f"Unsupported quantization route: {route!r}")
    return {**FRAMEWORK_CANDIDATES[route], "route": route}


def stage_template() -> dict[str, dict[str, Any]]:
    """Return the durable lifecycle contract for one candidate attempt."""

    return {
        stage: {"status": "not_run"} for stage in QUANTIZATION_STAGE_ORDER
    }


def validate_stage_trace(stages: dict[str, dict[str, Any]]) -> None:
    """Reject a manifest whose required lifecycle stages were skipped."""

    if tuple(stages) != QUANTIZATION_STAGE_ORDER:
        raise ValueError("Quantization stages must follow the declared lifecycle")
    required = {"passed", "reused"}
    missing = [
        stage
        for stage in QUANTIZATION_STAGE_ORDER
        if stages[stage].get("status") not in required
    ]
    if missing:
        raise ValueError(f"Quantization lifecycle is incomplete: {missing}")


def adapter_contract(route: str) -> dict[str, Any]:
    """Describe what a route adapter must provide without importing its backend."""

    candidate = candidate_defaults(route)
    return {
        "route": route,
        "framework": candidate["framework"],
        "format": candidate["format"],
        "requirements": candidate["requirements"],
        "stages": list(QUANTIZATION_STAGE_ORDER),
        "module_policy": expected_module_families(),
    }


def validate_config(config: dict[str, Any]) -> None:
    """Validate one JSON candidate without importing Torch or a quantizer."""

    route = config.get("route")
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"Unsupported quantization route: {route!r}")
    if config.get("bits") not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if config.get("group_size") not in (32, 64, 128, 256):
        raise ValueError("group_size must be one of 32, 64, 128 or 256")
    if config.get("route") == ROUTE_LLM_COMPRESSOR_FP8:
        if config.get("scheme") != "FP8_DYNAMIC":
            raise ValueError("LLM Compressor FP8 requires scheme=FP8_DYNAMIC")
    elif config.get("route") == ROUTE_TORCHAO_INT4:
        if config.get("bits") != 4:
            raise ValueError("torchao_int4 requires bits=4")
    elif config.get("bits") != 4 or config.get("group_size") != 128:
        raise ValueError("GPTQ/AWQ-style candidates require INT4 Group-128")
    if config.get("mapper_precision", MODULE_PRECISION_BF16) not in {
        MODULE_PRECISION_BF16,
        "fp8",
    }:
        raise ValueError("mapper_precision must be bf16 or fp8")
    calibration = config.get("calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    if FRAMEWORK_CANDIDATES[route]["calibration_required"]:
        if not calibration.get("data"):
            raise ValueError("calibration.data is required for this candidate")
        if type(calibration.get("limit", 0)) is not int or calibration["limit"] < 1:
            raise ValueError("calibration.limit must be a positive integer")
    runtime = config.get("runtime_config", config.get("runtime", {}))
    # ``candidate_defaults`` uses ``runtime`` for the deployment route name;
    # runtime control flags live in ``runtime_config`` when present.
    if isinstance(runtime, str):
        runtime = {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    if "runtime_python" in runtime and (not isinstance(runtime["runtime_python"], str)
                                        or not runtime["runtime_python"].strip()):
        raise ValueError("runtime.runtime_python must be a nonempty interpreter path")
    for key in ("clean_reload", "vllm_reload", "quantize", "vllm_benchmark"):
        if not isinstance(runtime.get(key, False), bool):
            raise ValueError(f"runtime.{key} must be boolean")
    utilization = runtime.get("vllm_gpu_memory_utilization")
    if utilization is not None and (
        not isinstance(utilization, (int, float)) or not 0 < float(utilization) <= 1
    ):
        raise ValueError("runtime.vllm_gpu_memory_utilization must be in (0, 1]")
    max_model_len = runtime.get("vllm_max_model_len")
    if max_model_len is not None and (
        type(max_model_len) is not int or max_model_len < 1
    ):
        raise ValueError("runtime.vllm_max_model_len must be a positive integer")
    for key in ("vllm_benchmark_limit", "vllm_benchmark_repeats", "vllm_benchmark_batch_size"):
        value = runtime.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"runtime.{key} must be a positive integer")
    warmup = runtime.get("vllm_benchmark_warmup")
    if warmup is not None and (type(warmup) is not int or warmup < 0):
        raise ValueError("runtime.vllm_benchmark_warmup must be a nonnegative integer")


def _module_patterns(model_adapter):
    if model_adapter == "deepseek_vl2":
        return DEEPSEEK_LANGUAGE_LINEAR, DEEPSEEK_VISION_LINEAR, DEEPSEEK_MAPPER
    if model_adapter == "internvl35":
        return INTERNVL_LANGUAGE_LINEAR, INTERNVL_VISION_LINEAR, INTERNVL_MAPPER
    if model_adapter in {"llava_onevision", "llava15"}:
        return LANGUAGE_LINEAR, LLAVA_VISION_LINEAR, LLAVA_MAPPER
    return LANGUAGE_LINEAR, VISION_LINEAR, MAPPER


def classify_module(name: str, *, model_adapter: str = "qwen3_vl") -> str:
    """Return the intended precision for a model module name."""

    # Explicit preservation wins over broad linear matching.  This protects
    # the visual-to-language interface even if a backend exposes it as Linear.
    language, vision, mapper = _module_patterns(model_adapter)
    if PRESERVE.search(name) or mapper.search(name):
        return MODULE_PRECISION_BF16
    if language.search(name) or vision.search(name):
        return MODULE_PRECISION_INT4
    return MODULE_PRECISION_BF16


def assignment(module_names: Iterable[str], *, model_adapter: str = "qwen3_vl") -> dict[str, str]:
    """Build a deterministic module-to-precision assignment."""

    return {
        name: classify_module(name, model_adapter=model_adapter)
        for name in sorted(set(module_names))
    }


def expected_module_families() -> dict[str, str]:
    """Expose policy families for reports and backend adapters."""

    return {
        "language_linear": MODULE_PRECISION_INT4,
        "vision_linear": MODULE_PRECISION_INT4,
        "vision_patch_embed": MODULE_PRECISION_BF16,
        "vision_mapper": MODULE_PRECISION_BF16,
        "embedding_norm_lm_head": MODULE_PRECISION_BF16,
    }


def make_manifest(
    *,
    model: dict[str, Any],
    config: dict[str, Any],
    calibration: dict[str, Any],
    module_names: Iterable[str] = (),
    environment: dict[str, Any] | None = None,
    status: str = "planned",
) -> dict[str, Any]:
    """Create the durable metadata for a framework probe or final artifact."""

    validate_config(config)
    modules = assignment(module_names, model_adapter=model.get("adapter", "qwen3_vl"))
    manifest = {
        "schema": "a4fvlm.quantization-manifest.v1",
        "status": status,
        "route": config["route"],
        "model": model,
        "model_digest": digest(model),
        "framework": candidate_defaults(config["route"]),
        "quantization": {
            "bits": config["bits"],
            "group_size": config["group_size"],
            "weight_activation": config.get("scheme", "W4A16"),
            "mapper_precision": config.get("mapper_precision", MODULE_PRECISION_BF16),
            "module_families": expected_module_families(),
        },
        "calibration": calibration,
        "calibration_digest": digest(calibration),
        "module_assignment": modules,
        "module_assignment_digest": digest(modules),
        "environment": environment or {},
    }
    manifest["artifact_digest"] = digest(
        {key: value for key, value in manifest.items() if key != "artifact_digest"}
    )
    return manifest


def validate_module_inventory(
    module_names: Iterable[str], *, require_vision: bool = True, model_adapter: str = "qwen3_vl"
) -> dict[str, Any]:
    """Check that a loaded Dense model exposes the intended target families."""

    names = sorted(set(module_names))
    selected = assignment(names, model_adapter=model_adapter)
    language_pattern, vision_pattern, mapper_pattern = _module_patterns(model_adapter)
    language = [name for name in names if language_pattern.search(name)]
    vision = [name for name in names if vision_pattern.search(name)]
    mapper = [name for name in names if mapper_pattern.search(name)]
    missing = []
    if not language:
        missing.append("language_linear")
    if require_vision and not vision:
        missing.append("vision_linear")
    if require_vision and not mapper:
        missing.append("vision_mapper")
    return {
        "module_count": len(names),
        "int4_count": sum(value == MODULE_PRECISION_INT4 for value in selected.values()),
        "bf16_count": sum(value == MODULE_PRECISION_BF16 for value in selected.values()),
        "language_linear_count": len(language),
        "vision_linear_count": len(vision),
        "mapper_count": len(mapper),
        "missing_families": missing,
        "assignment_digest": digest(selected),
        "valid": not missing,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject stale or internally inconsistent quantization metadata."""

    if manifest.get("schema") != "a4fvlm.quantization-manifest.v1":
        raise ValueError("Unsupported quantization manifest schema")
    config = {
        "route": manifest.get("route"),
        "bits": manifest.get("quantization", {}).get("bits"),
        "group_size": manifest.get("quantization", {}).get("group_size"),
        "mapper_precision": manifest.get("quantization", {}).get("mapper_precision"),
        "calibration": manifest.get("calibration", {}),
        "runtime": {},
    }
    validate_config(config)
    modules = manifest.get("module_assignment", {})
    model_adapter = (manifest.get("model") or {}).get("adapter", "qwen3_vl")
    if digest(modules) != manifest.get("module_assignment_digest"):
        raise ValueError("Quantization module assignment digest mismatch")
    if modules != assignment(modules, model_adapter=model_adapter):
        raise ValueError("Quantization module assignment uses the wrong model topology")
    expected = digest(
        {key: value for key, value in manifest.items() if key != "artifact_digest"}
    )
    if expected != manifest.get("artifact_digest"):
        raise ValueError("Quantization artifact digest mismatch")
