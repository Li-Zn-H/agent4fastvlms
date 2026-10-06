"""Pure-Python LMMS aggregation and quality-gate helpers for Stack."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TASKS: tuple[dict[str, str], ...] = (
    {"task": "chartqa", "metric": "relaxed_overall,none"},
    {"task": "mmstar", "metric": "average,none"},
    {"task": "textvqa_val", "metric": "exact_match,none"},
    {"task": "docvqa_val", "metric": "anls,none"},
    {"task": "pope", "metric": "pope_f1_score,none"},
)


def selection_policy(config):
    """Strict remains the default for existing jobs and qualified receipts."""
    policy = config.get("selection_policy", "strict")
    if policy not in {"strict", "best_effort"}:
        raise ValueError("selection_policy must be strict or best_effort")
    return policy


def complete_lmms(receipt):
    """A below-target handoff needs a complete, finite five-task decision.

    Callers still validate raw tasks/parents before creating or reusing a freeze.
    This compact contract never converts partial results or errors into quality.
    """
    if not isinstance(receipt, dict):
        return False
    rows = receipt.get("tasks", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return False
    retention = receipt.get("retention")
    minimum = receipt.get("minimum_retention")
    return (
        receipt.get("status") == "completed"
        and receipt.get("task_count") == 5
        and receipt.get("all_task_hashes_match") is True
        and receipt.get("token_mechanism_verified") is True
        and receipt.get("aggregation") == "mean_per_task_retention"
        and [{"task": r.get("task"), "metric": r.get("metric")} for r in rows] == list(DEFAULT_TASKS)
        and all(type(r.get("samples")) is int and r["samples"] > 0
                and r.get("task_hash_match") is True and r.get("token_mechanism")
                and type(r.get("retention")) in (int, float) and math.isfinite(r["retention"])
                for r in rows)
        and type(retention) in (int, float) and math.isfinite(retention)
        and type(minimum) in (int, float) and .95 <= minimum <= 1
        and math.isclose(retention, sum(r["retention"] for r in rows) / 5, abs_tol=1e-12)
        and receipt.get("quality_gate") == ("passed" if retention >= minimum else "quality_rejected")
    )


def stack_handoff_eligible(freeze):
    receipt = freeze.get("lmms", {})
    return receipt.get("quality_gate") == "passed" or (
        freeze.get("status") == "frozen_best_available"
        and selection_policy(freeze) == "best_effort"
        and freeze.get("quality_target_met") is False
        and freeze.get("search_stop_reason") in {"candidate_budget_exhausted", "recovery_ladder_exhausted"}
        and complete_lmms(receipt)
    )


def quality_speed_frontier(rows):
    """Keep measured quality/latency tradeoffs, without a weighted score.

    A caller supplies only validated candidates measured under the same scope.
    """
    valid = [r for r in rows if all(type(r.get(k)) in (int, float)
                                   and math.isfinite(r[k]) for k in ("retention", "mean_seconds"))
             and r["mean_seconds"] > 0]
    return sorted([r for r in valid if not any(
        o["retention"] >= r["retention"] and o["mean_seconds"] <= r["mean_seconds"]
        and (o["retention"] > r["retention"] or o["mean_seconds"] < r["mean_seconds"])
        for o in valid)], key=lambda r: (-r["retention"], r["mean_seconds"], r["candidate_id"]))


def select_measured(rows, minimum=.95):
    rows = quality_speed_frontier(rows)
    passing = [r for r in rows if r["retention"] >= minimum]
    return (min(passing, key=lambda r: (r["mean_seconds"], -r["retention"], r["candidate_id"]))
            if passing else rows[0] if rows else None)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _task_result(root: Path, task: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*_results.json")):
        try:
            result = _read(path)
        except (OSError, ValueError, TypeError):
            continue
        if task in result.get("results", {}):
            matches.append((path, result))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one complete LMMS result for {task}, found {len(matches)}")
    return matches[0]


def _sample_file(result_path: Path, task: str) -> Path:
    prefix = result_path.name.removesuffix("_results.json")
    matches = sorted(result_path.parent.glob(f"{prefix}_samples_{task}.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one native sample file for {task}, found {len(matches)}")
    return matches[0]


def _serving_receipt(root: Path, task: str, descriptor: dict[str, Any]) -> dict[str, Any]:
    receipt = _read(root / f"{task}-token-mechanism.runtime.json")
    if receipt.get("backend") != "vllm" or receipt.get("model") != descriptor:
        raise ValueError(f"Effective serving config mismatch for {task}")
    observed = receipt.get("observed_model", {})
    runtime = descriptor["runtime"]
    if (observed.get("token_pruning_enabled") is not True
            or observed.get("token_runtime", {}).get("keep_ratio") != descriptor["token_keep_ratio"]
            or observed.get("kv_cache_dtypes") != [runtime.get("kv_cache_dtype", "auto")]
            or not observed.get("attention_backends") or not observed.get("physical_widths")):
        raise ValueError(f"Missing actual vLLM model/settings evidence for {task}")
    backend = runtime.get("attention_backend", "auto")
    if backend != "auto" and observed["attention_backends"] != [backend]:
        raise ValueError(f"Observed attention backend mismatch for {task}")
    return receipt


def _mechanism_file(root: Path, task: str, expected: int, ratio: float | None, *,
                    backend: str = "native_transformers") -> Path | None:
    if ratio is None:
        return None
    path = root / f"{task}-token-mechanism.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing Token mechanism evidence for {task}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"Token mechanism evidence count mismatch for {task}: {len(rows)} != {expected}")
    reduced = 0
    for index, row in enumerate(rows):
        if float(row.get("token_keep_ratio")) != float(ratio):
            raise ValueError(f"Token mechanism ratio mismatch for {task}")
        before = int(row.get("image_tokens_before", 0))
        after = int(row.get("image_tokens_after", 0))
        # Coverage floors can legitimately retain all visual tokens for a
        # small image.  The controller must never increase the count, while
        # at least one sample in the task must demonstrate actual reduction.
        if before < 1 or after < 1 or after > before:
            raise ValueError(f"Token mechanism increased visual tokens for {task}")
        reduced += int(after < before)
        if backend == "vllm":
            images = row.get("images", [])
            if (row.get("backend") != "vllm" or row.get("controller") != "vllm_score_based_post_encoder"
                    or row.get("sample_index") != index or row.get("prompt_verified") is not True
                    or not row.get("native_prompt_sha256") or not images
                    or sum(i["image_tokens_before"] for i in images) != before
                    or sum(i["image_tokens_after"] for i in images) != after
                    or any((i.get("mrope_positions") != i["image_tokens_after"]
                            and not (i.get("position_type") == "1d_compacted"
                                     and i.get("positions_after") == i["image_tokens_after"]))
                           or not i.get("selected_indices_sha256")
                           or i.get("token_keep_ratio") != ratio for i in images)
                    or row.get("decoder_tokens_after") != row.get("native_prompt_tokens", 0) - before + after):
                raise ValueError(f"Invalid vLLM Token mechanism evidence for {task}")
        elif backend != "native_transformers":
            raise ValueError(f"Unsupported LMMS mechanism backend: {backend}")
        elif int(row.get("kv_tokens_after_last_forward")) != int(row.get("decoder_tokens_after")) + int(row.get("decode_steps")):
            raise ValueError(f"Token KV evidence mismatch for {task}")
    if reduced == 0:
        raise ValueError(f"Token mechanism did not reduce visual tokens for {task}")
    return path


def aggregate_lmms(
    candidate_root: str | Path,
    dense_root: str | Path,
    *,
    tasks: Iterable[dict[str, str]] = DEFAULT_TASKS,
    minimum_retention: float = 0.95,
    token_keep_ratio: float | None = None,
    candidate_id: str | None = None,
    model_config: str | None = None,
    execution_backend: str = "native_transformers",
) -> dict[str, Any]:
    """Validate complete native results and compute equal-task retention."""

    candidate_root = Path(candidate_root)
    dense_root = Path(dense_root)
    task_rows: list[dict[str, Any]] = []
    for spec in tasks:
        task = str(spec["task"])
        metric = str(spec["metric"])
        candidate_path, candidate = _task_result(candidate_root, task)
        dense_path, dense = _task_result(dense_root, task)
        candidate_count = candidate.get("n-samples", {}).get(task, {}).get("effective")
        dense_count = dense.get("n-samples", {}).get(task, {}).get("effective")
        if candidate_count != dense_count or not isinstance(candidate_count, int):
            raise ValueError(f"Sample count mismatch for {task}")
        candidate_hash = candidate.get("task_hashes", {}).get(task)
        dense_hash = dense.get("task_hashes", {}).get(task)
        if not candidate_hash or candidate_hash != dense_hash:
            raise ValueError(f"Task hash mismatch for {task}")
        dense_score = dense.get("results", {}).get(task, {}).get(metric)
        candidate_score = candidate.get("results", {}).get(task, {}).get(metric)
        if not isinstance(dense_score, (int, float)) or dense_score <= 0:
            raise ValueError(f"Invalid Dense denominator for {task}")
        if not isinstance(candidate_score, (int, float)):
            raise ValueError(f"Missing candidate metric {metric} for {task}")
        mechanism = _mechanism_file(candidate_root, task, candidate_count, token_keep_ratio,
                                    backend=execution_backend)
        if execution_backend == "vllm":
            descriptor = _read(Path(model_config)) if model_config else None
            if descriptor is None:
                raise ValueError(f"Missing serving model descriptor for {task}")
            _serving_receipt(candidate_root, task, descriptor)
            sample_sets = []
            for result_path in (candidate_path, dense_path):
                samples = [json.loads(line) for line in _sample_file(result_path, task).read_text().splitlines() if line.strip()]
                aligned = {row["doc_id"]: row.get("doc_hash") for row in samples}
                if len(samples) != candidate_count or len(aligned) != candidate_count:
                    raise ValueError(f"Sample IDs/count mismatch for {task}")
                sample_sets.append(aligned)
            if sample_sets[0] != sample_sets[1]:
                raise ValueError(f"Dense/candidate sample alignment mismatch for {task}")
        task_rows.append(
            {
                "task": task,
                "metric": metric,
                "stack": float(candidate_score),
                "dense": float(dense_score),
                "retention": float(candidate_score) / float(dense_score),
                "samples": candidate_count,
                "task_hash_match": True,
                "result": str(candidate_path.resolve()),
                "samples_jsonl": str(_sample_file(candidate_path, task).resolve()),
                "dense_result": str(dense_path.resolve()),
                "token_mechanism": str(mechanism.resolve()) if mechanism else None,
            }
        )
    retention = sum(row["retention"] for row in task_rows) / len(task_rows)
    return {
        "schema": "a4fvlm.lmms-summary.v1",
        "status": "completed",
        "candidate_id": candidate_id,
        "model_config": model_config,
        "token_keep_ratio": token_keep_ratio,
        "token_mechanism_verified": token_keep_ratio is None or all(row["token_mechanism"] for row in task_rows),
        "runtime_scope": ("vllm_stack_runtime_kv_lmms" if execution_backend == "vllm" else
                          "native_transformers_lmms; vllm_runtime_and_kv_settings_not_applied"),
        "tasks": task_rows,
        "task_count": len(task_rows),
        "all_task_hashes_match": True,
        "aggregation": "mean_per_task_retention",
        "retention": retention,
        "minimum_retention": float(minimum_retention),
        "quality_gate": "passed" if retention >= float(minimum_retention) else "quality_rejected",
    }


__all__ = ["DEFAULT_TASKS", "aggregate_lmms"]
