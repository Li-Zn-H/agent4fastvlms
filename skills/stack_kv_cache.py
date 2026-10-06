"""Contracts for KV-cache settings layered on a runtime-frozen Stack."""

from __future__ import annotations

from typing import Any, Mapping

from skills.quantization import digest
from skills.stack_runtime import KV_CACHE_DTYPES, performance_ready, validate_runtime_settings


KV_SCHEMA = "a4fvlm.stack-kv-cache.v1"


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("parent_freeze", "output", "evaluation_config", "vision_token_config"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"KV cache config requires {key}")
    if type(config.get("device")) is not int or config["device"] not in (0, 1):
        raise ValueError("device must be 0 or 1")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("KV cache config requires candidates")
    ids = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str):
            raise ValueError("each KV cache candidate requires a string id")
        if candidate["id"] in ids:
            raise ValueError("KV cache candidate ids must be unique")
        ids.add(candidate["id"])
        dtype = candidate.get("kv_cache_dtype", "auto")
        if dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"unsupported kv_cache_dtype: {dtype!r}")
        runtime = candidate.get("runtime", {})
        validate_runtime_settings({**runtime, "kv_cache_dtype": dtype})


def candidate_key(parent_digest: str, candidate: Mapping[str, Any]) -> str:
    return digest({"schema": KV_SCHEMA, "parent": parent_digest, "candidate": dict(candidate)})


def select_candidate(rows: list[dict[str, Any]], *, profile: str = "batch1") -> dict[str, Any] | None:
    qualified = []
    for row in rows:
        if not performance_ready(row):
            continue
        benchmark = row.get("benchmarks", {}).get(profile, {})
        summary = benchmark.get("summary", {}) if isinstance(benchmark, Mapping) else {}
        throughput = summary.get("output_token_throughput")
        if isinstance(throughput, (int, float)):
            qualified.append((float(throughput), row))
    if not qualified:
        return None
    qualified.sort(key=lambda item: (-item[0], item[1].get("candidate_id", "")))
    return qualified[0][1]
