"""Contracts for runtime optimizations layered on a frozen Stack artifact.

This module is deliberately importable in the Dense environment.  It only
contains JSON validation and vLLM argument translation; vLLM itself is loaded
inside the isolated Worker.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from skills.quantization import digest


RUNTIME_SCHEMA = "a4fvlm.stack-runtime.v1"
RUNTIME_FREEZE_SCHEMA = "a4fvlm.stack-runtime-freeze.v1"
ATTENTION_BACKENDS = ("auto", "FLASH_ATTN", "FLASHINFER")
OPTIMIZATION_LEVELS = (0, 1, 2, 3)
KV_CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_runtime_settings(runtime: Mapping[str, Any]) -> None:
    """Validate settings without importing Torch or vLLM."""

    if not isinstance(runtime, Mapping):
        raise ValueError("runtime settings must be an object")
    backend = runtime.get("attention_backend", "auto")
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unsupported attention_backend: {backend!r}")
    level = runtime.get("optimization_level")
    if level is not None and (type(level) is not int or level not in OPTIMIZATION_LEVELS):
        raise ValueError("optimization_level must be one of 0, 1, 2 or 3")
    kv_dtype = runtime.get("kv_cache_dtype")
    if kv_dtype is not None and kv_dtype not in KV_CACHE_DTYPES:
        raise ValueError(f"unsupported kv_cache_dtype: {kv_dtype!r}")
    for key in ("compile_mm_encoder", "cudagraph_mm_encoder", "qk_norm_rope_fusion"):
        if key in runtime and type(runtime[key]) is not bool:
            raise ValueError(f"runtime.{key} must be boolean")
    for key in ("vllm_gpu_memory_utilization",):
        value = runtime.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not 0 < float(value) <= 1
        ):
            raise ValueError(f"runtime.{key} must be in (0, 1]")
    for key in ("vllm_max_model_len", "vllm_benchmark_limit", "vllm_benchmark_repeats", "vllm_benchmark_batch_size"):
        if key in runtime and runtime[key] is not None:
            _positive_int(runtime[key], f"runtime.{key}")
    if "vllm_benchmark_warmup" in runtime:
        value = runtime["vllm_benchmark_warmup"]
        if type(value) is not int or value < 0:
            raise ValueError("runtime.vllm_benchmark_warmup must be nonnegative")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate one runtime-skill JSON job."""

    for key in ("parent_freeze", "output", "evaluation_config", "vision_token_config"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"runtime config requires {key}")
    if type(config.get("device")) is not int or config["device"] not in (0, 1):
        raise ValueError("device must be 0 or 1")
    limits = config.get("limits", {})
    if not isinstance(limits, Mapping):
        raise ValueError("limits must be an object")
    _positive_int(limits.get("total_candidates", 0), "limits.total_candidates")
    benchmark = config.get("benchmark", {})
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark must be an object")
    for key in ("search_limit", "warmup", "repeats", "batch_size"):
        if key in benchmark:
            value = benchmark[key]
            if key == "warmup":
                if type(value) is not int or value < 0:
                    raise ValueError("benchmark.warmup must be nonnegative")
            else:
                _positive_int(value, f"benchmark.{key}")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("runtime config requires candidates")
    ids = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str):
            raise ValueError("each runtime candidate requires a string id")
        if candidate["id"] in ids:
            raise ValueError("runtime candidate ids must be unique")
        ids.add(candidate["id"])
        validate_runtime_settings(candidate.get("runtime", {}))


def candidate_key(parent_digest: str, candidate: Mapping[str, Any]) -> str:
    return digest({"schema": RUNTIME_SCHEMA, "parent": parent_digest, "candidate": dict(candidate)})


def build_vllm_runtime_kwargs(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Translate portable JSON settings to the pinned vLLM 0.26 API.

    No vLLM import is needed here.  Unsupported fields are never silently
    passed to the engine; this keeps the preflight result meaningful.
    """

    validate_runtime_settings(runtime)
    kwargs: dict[str, Any] = {}
    backend = runtime.get("attention_backend", "auto")
    if backend != "auto":
        kwargs["attention_backend"] = backend
    if runtime.get("optimization_level") is not None:
        kwargs["optimization_level"] = int(runtime["optimization_level"])
    if runtime.get("kv_cache_dtype") is not None:
        kwargs["kv_cache_dtype"] = runtime["kv_cache_dtype"]
    if runtime.get("vllm_gpu_memory_utilization") is not None:
        kwargs["gpu_memory_utilization"] = float(runtime["vllm_gpu_memory_utilization"])
    if runtime.get("vllm_max_model_len") is not None:
        kwargs["max_model_len"] = int(runtime["vllm_max_model_len"])
    if runtime.get("enable_prefix_caching") is not None:
        kwargs["enable_prefix_caching"] = bool(runtime["enable_prefix_caching"])

    compilation: dict[str, Any] = {}
    for key in ("compile_mm_encoder", "cudagraph_mm_encoder"):
        if key in runtime:
            compilation[key] = bool(runtime[key])
    pass_config: dict[str, Any] = {}
    if "qk_norm_rope_fusion" in runtime:
        pass_config["enable_qk_norm_rope_fusion"] = bool(runtime["qk_norm_rope_fusion"])
    if pass_config:
        compilation["pass_config"] = pass_config
    if compilation:
        kwargs["compilation_config"] = compilation
    return kwargs


def performance_ready(probe: Mapping[str, Any]) -> bool:
    """Require completed measurements, including for legacy qualified receipts."""
    if probe.get("status") not in {"performance_ready", "qualified", "quality_rejected", "runtime_frozen", "kv_cache_frozen"}:
        return False
    benchmarks = probe.get("benchmarks", {})
    if not isinstance(benchmarks, Mapping) or not benchmarks:
        return False
    return all(
        isinstance(row, Mapping) and row.get("status") == "passed"
        and isinstance(row.get("summary"), Mapping)
        and all(isinstance(row.get("summary", {}).get(key), (int, float))
                and not isinstance(row["summary"][key], bool)
                and math.isfinite(row["summary"][key]) and row["summary"][key] > 0
                for key in ("elapsed_seconds", "request_count", "output_token_throughput"))
        for row in benchmarks.values()
    )


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the fastest measured runtime; quality awaits final full LMMS."""

    qualified = []
    for row in rows:
        if not performance_ready(row):
            continue
        benchmark = row.get("benchmarks", {}).get("search", {})
        summary = benchmark.get("summary", {}) if isinstance(benchmark, Mapping) else {}
        throughput = summary.get("output_token_throughput")
        if isinstance(throughput, (int, float)):
            qualified.append((float(throughput), row))
    if not qualified:
        return None
    qualified.sort(key=lambda item: (-item[0], item[1].get("candidate_id", "")))
    return qualified[0][1]
