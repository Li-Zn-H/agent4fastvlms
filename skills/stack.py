"""Contracts for the bounded multi-skill Stack stage.

The individual skills deliberately keep their own search state.  This module
only joins durable, independently validated receipts and never derives a
Stack result by multiplying skill-level speedups.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from skills.quantization import digest


STACK_SCHEMA = "a4fvlm.stack.v1"
TRANSFORMATION_ORDER = ("structured_pruning", "quantization", "vision_token")
RECOVERY_AXES = (
    "paired_structured_token_recovery",
    "wider_structured_pruning",
    "higher_token_keep_ratio",
    "disable_structured_pruning",
    "disable_quantization",
    "disable_vision_token",
)


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the small JSON control surface without importing Torch."""

    from search.lmms_gate import selection_policy
    policy = selection_policy(config)
    if policy == "best_effort" and (not isinstance(config.get("lmms"), dict)
                                    or config["lmms"].get("enabled") is not True):
        raise ValueError("best_effort requires complete native LMMS")
    if config.get("transformation_order") != list(TRANSFORMATION_ORDER):
        raise ValueError(
            "Stack transformation_order must be structured_pruning, quantization, vision_token"
        )
    for key in (
        "model_config",
        "evaluation_config",
        "structured_pruning_root",
        "quantization_root",
        "vision_token_root",
        "output",
    ):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Stack config requires a non-empty {key}")
    if isinstance(config.get("device"), bool) or config.get("device") not in (0, 1):
        raise ValueError("Stack device must be 0 or 1")
    limits = config.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("Stack limits must be an object")
    _positive_int(limits.get("initial_candidates", 0), "limits.initial_candidates")
    _positive_int(limits.get("total_candidates", 0), "limits.total_candidates")
    if limits["initial_candidates"] > limits["total_candidates"]:
        raise ValueError("initial_candidates cannot exceed total_candidates")
    initial_sources = config.get("initial_sources", {})
    if not isinstance(initial_sources, dict) or any(
        key not in TRANSFORMATION_ORDER or not isinstance(value, str) or not value
        for key, value in initial_sources.items()
    ):
        raise ValueError("Stack initial_sources must map skill names to source IDs")
    recovery = config.get("recovery_axes", list(RECOVERY_AXES))
    if not isinstance(recovery, list) or any(axis not in RECOVERY_AXES for axis in recovery):
        raise ValueError("Stack recovery_axes contains an unsupported axis")
    if len(set(recovery)) != len(recovery):
        raise ValueError("Stack recovery_axes must not contain duplicates")
    token_exhaustion = config.get("token_recovery_exhaustion", "stop")
    if token_exhaustion not in ("stop", "reuse_last_qualified"):
        raise ValueError("Stack token_recovery_exhaustion must be stop or reuse_last_qualified")
    if token_exhaustion == "reuse_last_qualified" and (
        "paired_structured_token_recovery" not in recovery
        or not isinstance(config.get("lmms"), dict)
        or config["lmms"].get("enabled") is not True
    ):
        raise ValueError("Token exhaustion fallback requires paired recovery and complete native LMMS")
    if not isinstance(config.get("require_fresh_evidence", True), bool):
        raise ValueError("require_fresh_evidence must be boolean")
    lmms = config.get("lmms", {})
    if lmms is True:
        lmms = {}
    if not isinstance(lmms, dict):
        raise ValueError("Stack lmms must be an object")
    if "enabled" in lmms and not isinstance(lmms["enabled"], bool):
        raise ValueError("Stack lmms.enabled must be boolean")
    if "smoke_samples_per_task" in lmms:
        samples = lmms["smoke_samples_per_task"]
        if type(samples) is not int or samples < 1:
            raise ValueError("Stack lmms.smoke_samples_per_task must be a positive integer")
    if "minimum_retention" in lmms:
        threshold = lmms["minimum_retention"]
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.95 <= float(threshold) <= 1:
            raise ValueError("Stack lmms.minimum_retention must be in [0.95, 1]")
    if lmms.get("enabled", False) is True:
        from search.lmms_gate import DEFAULT_TASKS
        if lmms.get("tasks", list(DEFAULT_TASKS)) != list(DEFAULT_TASKS):
            raise ValueError("Stack full LMMS requires all five native tasks")
        for key in ("worker_python", "baseline_root", "output_subdir"):
            if not isinstance(lmms.get(key), str) or not lmms[key]:
                raise ValueError(f"Stack lmms.{key} must be non-empty when enabled")
        tasks = lmms.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("Stack lmms.tasks must be a non-empty list")
            for task in tasks:
                if (
                    not isinstance(task, dict)
                    or not isinstance(task.get("task"), str)
                    or not task["task"]
                    or not isinstance(task.get("metric"), str)
                    or not task["metric"]
                ):
                    raise ValueError("Each Stack lmms task needs task and metric")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("Stack runtime must be an object")
    for key in ("quantize", "transformers_reload", "vllm_reload", "benchmark"):
        if key in runtime and not isinstance(runtime[key], bool):
            raise ValueError(f"Stack runtime.{key} must be boolean")
    quantization = config.get("quantization", {})
    if not isinstance(quantization, dict):
        raise ValueError("Stack quantization must be an object")
    if quantization:
        if not isinstance(quantization.get("route"), str) or not quantization["route"]:
            raise ValueError("Stack quantization.route must be non-empty")
        worker_python = quantization.get("worker_python")
        if not isinstance(worker_python, str) or not worker_python:
            raise ValueError("Stack quantization.worker_python must be non-empty")
        calibration = quantization.get("calibration")
        if not isinstance(calibration, dict) or not calibration.get("data"):
            raise ValueError("Stack quantization.calibration.data is required")
        runtime = quantization.get("runtime", {})
        if not isinstance(runtime, dict) or runtime.get("quantize") is not True:
            raise ValueError("Stack quantization.runtime.quantize must be true")


def evidence_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compare legacy jobs without obsolete local gates; never mutate receipts.

    Only fields removed by the LMMS-only policy are ignored. Artifact, runtime,
    benchmark workload, LMMS and independent-source settings stay binding.
    """
    value = dict(config)
    # Selection policy does not change measurements; handoff eligibility is
    # checked separately before a below-target freeze can be reused.
    value.pop("selection_policy", None)
    quality = dict(value.get("quality", {}))
    quality.pop("minimum_dev_retention", None)
    if quality:
        value["quality"] = quality
    else:
        value.pop("quality", None)
    if "benchmark" in value:
        value["benchmark"] = dict(value["benchmark"])
        value["benchmark"].pop("dev_limit", None)
    return value


def _status(value: dict[str, Any] | None) -> str | None:
    if not isinstance(value, dict):
        return None
    return value.get("status")


def _dev_pass(value: dict[str, Any] | None, minimum: float) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("action") == "pass":
        return True
    stages = value.get("stages", {})
    dev = stages.get("dev", {}) if isinstance(stages, dict) else {}
    return dev.get("action") == "pass" and float(dev.get("retention", 0)) >= minimum


def validate_source_receipt(
    source: dict[str, Any], *, kind: str, minimum_dev_retention: float
) -> None:
    """Fail closed unless a source has durable quality/runtime evidence."""

    if not isinstance(source, dict):
        raise ValueError(f"{kind} source must be an object")
    if not source.get("candidate_id"):
        raise ValueError(f"{kind} source has no candidate_id")
    if source.get("status") not in {"qualified", "completed", "pass"}:
        raise ValueError(f"{kind} source is not qualified")
    quality = source.get("quality")
    if not _dev_pass(quality, minimum_dev_retention):
        raise ValueError(f"{kind} source lacks passing Dev evidence")
    if source.get("fresh_evidence", True) is not True:
        raise ValueError(f"{kind} source does not declare fresh evidence")


def candidate_key(
    *,
    model_digest: str,
    sp_recipe_digest: str | None,
    quant_manifest_digest: str | None,
    token_config_digest: str | None,
    token_keep_ratio: float | None,
) -> str:
    """Return a stable physical-composition identity."""

    if token_keep_ratio is not None:
        if not isinstance(token_keep_ratio, (int, float)) or not 0 < float(token_keep_ratio) <= 1:
            raise ValueError("token_keep_ratio must be in (0, 1]")
    return digest(
        {
            "schema": STACK_SCHEMA,
            "model_digest": model_digest,
            "sp_recipe_digest": sp_recipe_digest,
            "quant_manifest_digest": quant_manifest_digest,
            "token_config_digest": token_config_digest,
            "token_keep_ratio": token_keep_ratio,
        }
    )


def make_candidate(
    *,
    model_digest: str,
    sp: dict[str, Any] | None,
    quantization: dict[str, Any] | None,
    token: dict[str, Any] | None,
    token_keep_ratio: float | None,
    source_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Create one composition request without fabricating performance metrics."""

    if sp is None and quantization is None and token is None:
        raise ValueError("A Stack candidate needs at least two enabled skills")
    if quantization is not None and sp is None:
        # Dense quantization plus Token is a supported smoke composition; a
        # physical SP+quant candidate must carry its SP source explicitly.
        pass
    if token is None and token_keep_ratio is not None:
        raise ValueError("token_keep_ratio requires a Token source")
    sp_digest = None if sp is None else sp.get("recipe_digest")
    quant_digest = None if quantization is None else quantization.get("manifest_digest")
    if quantization is not None and quantization.get("export_identity_digest"):
        quant_digest = digest({"manifest": quant_digest, "export": quantization["export_identity_digest"]})
    token_digest = None if token is None else token.get("config_digest")
    key = candidate_key(
        model_digest=model_digest,
        sp_recipe_digest=sp_digest,
        quant_manifest_digest=quant_digest,
        token_config_digest=token_digest,
        token_keep_ratio=token_keep_ratio,
    )
    kinds = [name for name, value in (("structured_pruning", sp), ("quantization", quantization), ("vision_token", token)) if value is not None]
    return {
        "schema": STACK_SCHEMA,
        "candidate_id": f"stack-{key}",
        "candidate_key": key,
        "model_digest": model_digest,
        "skills": kinds,
        "structured_pruning": sp,
        "quantization": quantization,
        "vision_token": token,
        "token_keep_ratio": token_keep_ratio,
        "source_ids": sorted(set(source_ids)),
        "metrics_transferable": False,
        "fresh_evidence": False,
        "status": "planned",
    }


def select_initial_candidates(candidates: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Choose a deterministic bounded set, preferring deeper valid SP cuts."""

    _positive_int(limit, "initial candidate limit")
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate.get("candidate_key")
        if not key:
            raise ValueError("Stack candidate is missing candidate_key")
        unique.setdefault(key, candidate)
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            -float((row.get("structured_pruning") or {}).get("actual_depth", 0.0)),
            float(row.get("token_keep_ratio") or 1.0),
            row["candidate_id"],
        ),
    )
    return ordered[:limit]


def recovery_candidates(
    candidate: dict[str, Any], *, wider_sp: dict[str, Any] | None = None,
    higher_token_keep_ratio: float | None = None,
    paired_sp: dict[str, Any] | None = None,
    paired_token_keep_ratio: float | None = None,
    paired_token: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate bounded recovery neighbors.

    The paired form is the fast Stack recovery path: one wider SP recipe and
    one higher Token keep ratio are changed in the same candidate.  The
    one-axis form remains available for older callers and focused contracts.
    """

    result = []
    if paired_sp is not None or paired_token_keep_ratio is not None:
        if paired_sp is None or paired_token_keep_ratio is None:
            raise ValueError("Paired recovery requires both SP and Token values")
        current = candidate.get("token_keep_ratio")
        if current is None or paired_token_keep_ratio <= current:
            raise ValueError("Recovery Token ratio must be higher than the current ratio")
        token = paired_token or candidate.get("vision_token")
        if token is None:
            raise ValueError("Paired recovery requires a Token source")
        result.append(
            make_candidate(
                model_digest=candidate["model_digest"],
                sp=paired_sp,
                quantization=candidate.get("quantization"),
                token=token,
                token_keep_ratio=paired_token_keep_ratio,
                source_ids=(
                    paired_sp.get("candidate_id"),
                    (candidate.get("quantization") or {}).get("candidate_id"),
                    token.get("candidate_id"),
                ),
            )
        )
        return result
    if wider_sp is not None:
        result.append(
            make_candidate(
                model_digest=candidate["model_digest"],
                sp=wider_sp,
                quantization=candidate.get("quantization"),
                token=candidate.get("vision_token"),
                token_keep_ratio=candidate.get("token_keep_ratio"),
                source_ids=candidate.get("source_ids", ()),
            )
        )
    if higher_token_keep_ratio is not None:
        current = candidate.get("token_keep_ratio")
        if current is None or higher_token_keep_ratio <= current:
            raise ValueError("Recovery Token ratio must be higher than the current ratio")
        result.append(
            make_candidate(
                model_digest=candidate["model_digest"],
                sp=candidate.get("structured_pruning"),
                quantization=candidate.get("quantization"),
                token=candidate.get("vision_token"),
                token_keep_ratio=higher_token_keep_ratio,
                source_ids=candidate.get("source_ids", ()),
            )
        )
    return result


def validate_result(result: dict[str, Any]) -> None:
    """Validate fresh mechanism and timing evidence before full LMMS."""

    if result.get("schema") != "a4fvlm.stack-result.v1":
        raise ValueError("Unexpected Stack result schema")
    if result.get("candidate_id") is None:
        raise ValueError("Stack result has no candidate_id")
    if result.get("fresh_evidence") is not True:
        raise ValueError("Stack result is missing fresh evidence")
    skills = set(result.get("skills", ()))
    if {"structured_pruning", "quantization"}.issubset(skills):
        mechanism = result.get("mechanism", {})
        if mechanism.get("fresh_shape_reparse") is not True:
            raise ValueError("SP+quantization result lacks a fresh shape reparse")
        if mechanism.get("fresh_calibration") is not True:
            raise ValueError("SP+quantization result lacks fresh calibration")
    for stage in ("mechanism", "reload", "benchmark"):
        if _status(result.get(stage)) not in {"passed", "qualified"}:
            raise ValueError(f"Stack result is missing {stage} evidence")


def compare_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic non-scalar Pareto frontier over fresh results."""

    valid = [row for row in rows if row.get("status") in {"ready_for_lmms", "qualified", "passed"}
             and row.get("lmms", {}).get("quality_gate") == "passed"]
    frontier = []
    for row in valid:
        metrics = row.get("benchmark", {})
        current = metrics.get("mean_seconds")
        memory = metrics.get("peak_allocated_bytes")
        quality = row.get("lmms", {}).get("retention")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (current, memory, quality)):
            continue
        dominated = False
        for other in valid:
            om = other.get("benchmark", {})
            oq = other.get("lmms", {}).get("retention")
            if other is row or not isinstance(oq, (int, float)):
                continue
            if (
                isinstance(om.get("mean_seconds"), (int, float))
                and isinstance(om.get("peak_allocated_bytes"), (int, float))
                and om["mean_seconds"] <= current
                and om["peak_allocated_bytes"] <= memory
                and oq >= quality
                and (
                    om["mean_seconds"] < current
                    or om["peak_allocated_bytes"] < memory
                    or oq > quality
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: row.get("candidate_id", ""))


def select_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a deterministic member of the non-scalar frontier.

    Pareto filtering remains the decision boundary.  This tie-break only makes
    a single artifact reproducible when several frontier rows are equivalent:
    higher full LMMS retention, then lower latency, then lower memory, then ID.
    """

    frontier = compare_candidates(rows)
    if not frontier:
        return None
    return max(
        frontier,
        key=lambda row: (
            float(row["lmms"]["retention"]),
            -float(row["benchmark"]["mean_seconds"]),
            -float(row["benchmark"]["peak_allocated_bytes"]),
            row.get("candidate_id", ""),
        ),
    )


__all__ = [
    "RECOVERY_AXES",
    "STACK_SCHEMA",
    "TRANSFORMATION_ORDER",
    "candidate_key",
    "compare_candidates",
    "make_candidate",
    "recovery_candidates",
    "select_initial_candidates",
    "select_candidate",
    "validate_config",
    "validate_result",
    "validate_source_receipt",
]
