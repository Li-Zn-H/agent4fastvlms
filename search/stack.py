"""Plan and execute the bounded Qwen3-VL multi-skill Stack stage."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import os
from pathlib import Path
from typing import Any

from search.model import PROJECT_ROOT
from search.context import result_root
from search.optimize import hardware, launch, lock, read
from search.worker import save_json
from search.vllm_benchmark import compare_reports
from search.lmms_gate import (DEFAULT_TASKS, aggregate_lmms, complete_lmms,
                              selection_policy, stack_handoff_eligible,
                              quality_speed_frontier, select_measured)
from skills.quantization import digest, validate_manifest
from skills.stack import (
    STACK_SCHEMA,
    evidence_config,
    make_candidate,
    select_candidate,
    select_initial_candidates,
    validate_config,
    validate_result,
)


_VLLM_SCORE_TOKEN_METHOD = "vllm_score_based_post_encoder"


def _score_token_benchmark_ready(probe: dict[str, Any]) -> bool:
    """Return whether a Token Stack receipt uses feature-score pruning.

    Older Stack receipts marked the Token path as enabled while only changing
    the image processor's pixel budget.  Those reports are useful historical
    measurements, but they do not prove the current post-encoder mechanism and
    must be revalidated before they can be reused as Stack evidence.
    """

    token = probe.get("vllm_benchmark", {}).get("vision_token", {})
    return (
        isinstance(token, dict)
        and token.get("status") == "enabled"
        and token.get("method") == _VLLM_SCORE_TOKEN_METHOD
    )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _quality(path: Path) -> dict[str, Any] | None:
    try:
        value = read(path)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _quality_pass(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if "action" in value:
        return value.get("action") == "pass"
    stages = value.get("stages", {})
    return isinstance(stages, dict) and any(
        isinstance(stage, dict) and stage.get("action") == "pass"
        for stage in stages.values()
    )


def _source(
    *, candidate_id: str, status: str, quality: dict[str, Any], path: Path, **facts: Any
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "quality": quality,
        "source": str(path.resolve()),
        "fresh_evidence": True,
        **facts,
    }


def discover_structured_pruning(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Find quality-passing physical recipes; no GPU query or model load."""

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for recipe_path in sorted(root.glob("candidates/step-*/recipe.json")):
        try:
            recipe = read(recipe_path)
            # Recovery Dev decisions supersede earlier Search-only receipts,
            # including when the attempt number grows past a single digit.
            attempts = sorted(
                recipe_path.parent.glob("attempt-*/result.json"),
                key=lambda path: int(path.parent.name.removeprefix("attempt-")),
            )
            result_path = next(
                (path for path in reversed(attempts) if _quality(path)), None
            )
            if result_path is None:
                continue
            result = read(result_path)
            quality = result.get("quality")
            if result.get("status") != "completed" or not _quality_pass(quality):
                continue
            index = recipe_path.parent.name.split("-", 1)[-1]
            rows.append(
                _source(
                    candidate_id=f"sp-{index}",
                    status="completed",
                    quality=quality,
                    path=recipe_path,
                    recipe_path=str(recipe_path.resolve()),
                    recipe_digest=digest(recipe),
                    widths=recipe.get("widths"),
                    actual_depth=float(recipe.get("actual_depth", 0.0)),
                    result_path=str(result_path.resolve()),
                )
            )
        except (OSError, ValueError, TypeError, KeyError):
            blockers.append(f"invalid_sp_source:{recipe_path}")
    rows.sort(key=lambda row: (-row["actual_depth"], row["candidate_id"]))
    if not rows:
        blockers.append("no_passing_structured_pruning_recipe")
    return rows, blockers


def discover_quantization(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Find a qualified exported quantization artifact and its manifest."""

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    summary_path = root / "summary.json"
    summary = _quality(summary_path) if summary_path.is_file() else None
    config = _quality(root / "config.json")
    config_candidates = {}
    if isinstance(config, dict):
        for item in config.get("candidates", []):
            if isinstance(item, dict) and item.get("id"):
                config_candidates[str(item["id"])] = item
    selected = summary.get("selected_candidate") if summary else None
    candidate_ids = [selected] if selected else []
    if not candidate_ids:
        candidate_ids = [path.parent.name for path in sorted((root / "candidates").glob("*/probe.json"))]
    for candidate_id in candidate_ids:
        if not candidate_id:
            continue
        probe_path = root / "candidates" / candidate_id / "probe.json"
        result_path = root / "candidates" / candidate_id / "result.json"
        artifact = root / "final" / "artifact"
        if not probe_path.is_file() or not result_path.is_file():
            continue
        try:
            probe = read(probe_path)
            result = read(result_path)
            manifest_path = artifact / "manifest.json"
            manifest = read(manifest_path)
            validate_manifest(manifest)
            candidate_config = config_candidates.get(str(candidate_id), {})
            calibration = candidate_config.get("calibration", {})
            runtime = candidate_config.get("runtime_config", {})
            worker_python = candidate_config.get("worker_python")
            if (
                probe.get("status") != "qualified"
                or result.get("status") != "completed"
                or not artifact.is_dir()
                or not manifest_path.is_file()
            ):
                continue
            export_identity = {}
            if candidate_config.get("vision_quantization") == "siglip_gptq_zero_pad_v1":
                # Legacy manifests hash declared precision, so a decoder-only
                # export and a completed visual export can share that hash.
                # Bind the new route to its actual padding/calibration receipts.
                visual_receipt = read(artifact / "vision-quantization.json")
                cache_identity = read(artifact / "cache-identity.json")
                if visual_receipt.get("method") != candidate_config["vision_quantization"]:
                    raise ValueError("Visual quantization receipt does not match selected route")
                export_identity = {
                    "manifest_digest": manifest.get("artifact_digest"),
                    "artifact_config": read(artifact / "config.json"),
                    "visual_calibration": visual_receipt,
                    "implementation_sha256": cache_identity["implementation_sha256"],
                }
            rows.append(
                _source(
                    candidate_id=str(candidate_id),
                    status="qualified",
                    quality=probe.get("quality", {}),
                    path=probe_path,
                    artifact=str(artifact.resolve()),
                    manifest_path=str(manifest_path.resolve()),
                    manifest_digest=manifest.get("artifact_digest") or digest(manifest),
                    route=probe.get("route"),
                    bits=candidate_config.get("bits", manifest.get("bits")),
                    group_size=candidate_config.get("group_size", manifest.get("group_size")),
                    scheme=candidate_config.get("scheme", manifest.get("scheme")),
                    mapper_precision=candidate_config.get("mapper_precision", "bf16"),
                    worker_python=(
                        str(_resolve(worker_python)) if worker_python else None
                    ),
                    calibration=calibration,
                    runtime_config=runtime,
                    candidate_config=candidate_config,
                    **({"export_identity_digest": digest(export_identity)} if export_identity else {}),
                    result_path=str(result_path.resolve()),
                )
            )
        except (OSError, ValueError, TypeError, KeyError):
            blockers.append(f"invalid_quantization_source:{candidate_id}")
    if not rows:
        blockers.append("no_qualified_quantization_artifact")
    return rows, blockers


def discover_tokens(
    root: Path, config: dict[str, Any], allowed_ratios: list[float] | None = None,
    *, model=None, evaluation=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find existing Token ratios with quality and benchmark receipts."""

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    config_digest = digest(config)
    for directory in sorted(root.glob("keep-*")):
        match = re.fullmatch(r"keep-(\d+)", directory.name)
        if not match:
            continue
        ratio = int(match.group(1)) / 100
        if allowed_ratios is not None and not any(
            abs(ratio - float(value)) < 1e-9 for value in allowed_ratios
        ):
            continue
        quality_path = directory / "quality" / "summary.json"
        benchmark_path = directory / "benchmark" / "summary.json"
        quality = _quality(quality_path)
        benchmark = _quality(benchmark_path)
        if not _quality_pass(quality):
            continue
        if not isinstance(benchmark, dict) or not isinstance(benchmark.get("mean_round_seconds"), (int, float)):
            blockers.append(f"token_missing_benchmark:{ratio:.2f}")
            continue
        rows.append(
            _source(
                candidate_id=f"token-{int(ratio * 100):02d}",
                status="completed",
                quality=quality,
                path=quality_path,
                ratio=ratio,
                config_digest=config_digest,
                benchmark=str(benchmark_path.resolve()),
            )
        )
    # Unified Token runs keep attempts under candidates/keep-*/ rather than
    # the historical root/keep-* layout. Consume their original evidence;
    # never export duplicate/fabricated legacy summaries just for Stack.
    from search.optimize import evidence_paths, inspect_benchmark, inspect_quality

    known = {row["ratio"] for row in rows}
    for state_path in sorted(root.rglob("state.json")):
        saved = _quality(state_path.parent / "config.json")
        state = _quality(state_path)
        if not saved or not state or "vision" not in saved:
            continue
        if (saved["vision"].get("selection") != config.get("selection")
                or (model is not None and saved.get("model") != model)
                or (evaluation is not None and saved.get("evaluation") != evaluation)):
            continue
        paths = evidence_paths([state_path.parent])
        for candidate in state.get("candidates", []):
            ratio = candidate.get("keep_ratio")
            if ratio is None or ratio in known or (allowed_ratios is not None
                    and not any(abs(ratio - value) < 1e-9 for value in allowed_ratios)):
                continue
            # A new runner may have reused every answer/timing from another
            # directory and consequently have no local Worker attempts.
            references = list(candidate.get("quality_sources", {}).values())
            if (candidate.get("benchmark") or {}).get("source"):
                references.append(candidate["benchmark"]["source"])
            candidate_paths = paths + [Path(source).parent.parent / "request.json" for source in references]
            sources, quality = inspect_quality(saved, ratio, candidate_paths)
            benchmark = inspect_benchmark(saved, ratio, candidate_paths)
            if not _quality_pass(quality) or benchmark is None:
                continue
            rows.append(_source(
                candidate_id=f"token-{round(ratio * 100):02d}", status="completed",
                quality=quality, path=state_path, ratio=ratio,
                config_digest=config_digest, benchmark=benchmark["source"],
                quality_sources=sources,
            ))
            known.add(ratio)
    rows.sort(key=lambda row: (row["ratio"], row["candidate_id"]))
    if not rows:
        blockers.append("no_passing_token_benchmark")
    return rows, blockers


def snapshot(config_path: str | Path) -> dict[str, Any]:
    config_path = _resolve(config_path)
    config = read(config_path)
    validate_config(config)
    model = read(_resolve(config["model_config"]))
    evaluation = read(_resolve(config["evaluation_config"]))
    from search.context import require_stages, validate_result_scope
    require_stages(model, ("stack",), device=config["device"])
    validate_result_scope(model, evaluation)
    token_config = read(_resolve(config.get("vision_token_config", "configs/vision_token.json")))
    sp, sp_blockers = discover_structured_pruning(_resolve(config["structured_pruning_root"]))
    quant, quant_blockers = discover_quantization(_resolve(config["quantization_root"]))
    requested_route = config.get("quantization", {}).get("route")
    if requested_route:
        quant = [row for row in quant if row.get("route") == requested_route]
        if not quant:
            quant_blockers.append(f"no_qualified_quantization_route:{requested_route}")
    tokens, token_blockers = discover_tokens(
        _resolve(config["vision_token_root"]),
        token_config,
        config.get("token_keep_ratios"),
        model=model, evaluation=evaluation,
    )
    candidates: list[dict[str, Any]] = []
    if sp and quant and tokens:
        for sp_source in sp:
            for quant_source in quant:
                for token_source in tokens:
                    candidates.append(
                        make_candidate(
                            model_digest=digest(model),
                            sp=sp_source,
                            quantization=quant_source,
                            token=token_source,
                            token_keep_ratio=token_source["ratio"],
                            source_ids=(
                                sp_source["candidate_id"],
                                quant_source["candidate_id"],
                                token_source["candidate_id"],
                            ),
                        )
                    )
    initial_sources = config.get("initial_sources", {})
    initial_pool = [
        row for row in candidates
        if all((row.get(skill) or {}).get("candidate_id") == source_id
               for skill, source_id in initial_sources.items())
    ]
    if initial_sources and not initial_pool:
        sp_blockers.append("initial_sources_not_available")
    initial = select_initial_candidates(
        initial_pool, config["limits"]["initial_candidates"]
    ) if initial_pool else []
    return {
        "schema": STACK_SCHEMA,
        "config_path": str(config_path.resolve()),
        "config": config,
        "model": model,
        "evaluation": evaluation,
        "token_config": token_config,
        "model_digest": digest(model),
        "sources": {
            "structured_pruning": sp,
            "quantization": quant,
            "vision_token": tokens,
        },
        "blockers": sorted(set(sp_blockers + quant_blockers + token_blockers)),
        "candidate_count": len(candidates),
        "initial_candidates": initial,
        "candidates": candidates,
        "status": "ready" if initial else "blocked",
    }


def plan(config_path: str | Path = "configs/stack.json") -> dict[str, Any]:
    snap = snapshot(config_path)
    compact = [
        {
            "candidate_id": row["candidate_id"],
            "candidate_key": row["candidate_key"],
            "skills": row["skills"],
            "source_ids": row["source_ids"],
            "token_keep_ratio": row.get("token_keep_ratio"),
            "sp_candidate": (row.get("structured_pruning") or {}).get("candidate_id"),
            "quantization_candidate": (row.get("quantization") or {}).get("candidate_id"),
        }
        for row in snap["initial_candidates"]
    ]
    return {
        "schema": snap["schema"],
        "status": snap["status"],
        "model_digest": snap["model_digest"],
        "source_counts": {key: len(value) for key, value in snap["sources"].items()},
        "candidate_count": snap["candidate_count"],
        "initial_candidates": compact,
        "token_recovery_exhaustion": snap["config"].get("token_recovery_exhaustion", "stop"),
        "blockers": snap["blockers"],
        "output": str(_resolve(snap["config"]["output"])),
        "metrics_transferable": False,
    }


def _write_vllm_comparison(destination: Path, selected: dict[str, Any]) -> None:
    """Persist a compact same-runtime receipt for the selected Stack."""

    dense_path = destination / "dense-vllm-benchmark-8192" / "benchmark.json"
    stack_report = (selected.get("vllm_benchmark") or {}).get("report")
    stack_path = Path(stack_report) if stack_report else None
    if not dense_path.is_file() or stack_path is None or not stack_path.is_file():
        return
    try:
        dense = read(dense_path)
        stack = read(stack_path)
        token_runtime = stack.get("vision_token")
        receipt = {
            "schema": "a4fvlm.stack-vllm-comparison.v2",
            "status": "passed",
            "runtime": stack.get("runtime", {}),
            "dense_report": str(dense_path.resolve()),
            "stack_report": str(stack_path.resolve()),
            "scope": (
                "physical_sp_plus_gptq_marlin_plus_vllm_score_token_pruning"
                if isinstance(token_runtime, dict)
                and token_runtime.get("status") == "enabled"
                else "physical_sp_plus_gptq_marlin"
            ),
            "token_controller": (
                token_runtime.get("method")
                if isinstance(token_runtime, dict)
                else "not_applied"
            ),
            "token_runtime": token_runtime,
            "metrics_transferable": False,
            "comparison": compare_reports(dense, stack),
        }
        save_json(destination / "vllm-comparison.json", receipt)
    except (OSError, TypeError, ValueError, KeyError):
        # The Stack receipt remains authoritative even when an optional Dense
        # control is absent or an older benchmark schema cannot be compared.
        return


def _recovery_neighbors(candidate: dict[str, Any], snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Return bounded recovery neighbors after a Stack quality rejection.

    Paired recovery changes both quality axes together. An explicit Token
    exhaustion policy may retain the last qualified Token source while SP
    widens; it never skips fresh composition or the full LMMS gate.
    """

    config = snap["config"]
    axes = config.get("recovery_axes", [])
    sp_sources = snap["sources"].get("structured_pruning", [])
    token_sources = snap["sources"].get("vision_token", [])
    current_sp = (candidate.get("structured_pruning") or {}).get("candidate_id")
    current_token = candidate.get("token_keep_ratio")
    result: list[dict[str, Any]] = []
    if "paired_structured_token_recovery" in axes:
        if not current_sp or current_token is None:
            return result
        ordered_sp = sorted(
            [row for row in sp_sources if row.get("candidate_id") != current_sp],
            key=lambda row: (-float(row.get("actual_depth", 0.0)), row.get("candidate_id", "")),
        )
        current_depth = float(
            (candidate.get("structured_pruning") or {}).get("actual_depth", 0.0)
        )
        wider = [
            row
            for row in ordered_sp
            if float(row.get("actual_depth", 0.0)) < current_depth
        ]
        higher = [
            row
            for row in token_sources
            if float(row.get("ratio", 0.0)) > float(current_token)
        ]
        higher.sort(key=lambda row: (float(row["ratio"]), row.get("candidate_id", "")))
        if wider and higher:
            from skills.stack import recovery_candidates

            paired = recovery_candidates(
                candidate,
                paired_sp=wider[0],
                paired_token_keep_ratio=higher[0]["ratio"],
                paired_token=higher[0],
            )[0]
            paired["recovery"] = {
                "strategy": "paired_structured_token_recovery",
                "parent_candidate": candidate.get("candidate_id"),
                "from_sp": current_sp,
                "to_sp": wider[0].get("candidate_id"),
                "from_token_keep_ratio": current_token,
                "to_token_keep_ratio": higher[0].get("ratio"),
            }
            return [paired]
        if wider and not higher and config.get("token_recovery_exhaustion") == "reuse_last_qualified":
            current_source = candidate.get("vision_token") or {}
            last_token = next((row for row in token_sources
                if row.get("ratio") == current_token
                and row.get("candidate_id") == current_source.get("candidate_id")
                and row.get("config_digest") == current_source.get("config_digest")
                and row.get("status") in {"completed", "qualified", "pass"}
                and row.get("fresh_evidence", True) is True
                and _quality_pass(row.get("quality"))), None)
            if last_token is not None:
                fallback = make_candidate(
                    model_digest=candidate["model_digest"],
                    sp=wider[0], quantization=candidate.get("quantization"),
                    token=last_token, token_keep_ratio=current_token,
                    source_ids=(wider[0]["candidate_id"],
                                (candidate.get("quantization") or {})["candidate_id"],
                                last_token["candidate_id"]),
                )
                fallback["recovery"] = {
                    "strategy": "wider_sp_reuse_last_token",
                    "reason": "token_recovery_ladder_exhausted",
                    "parent_candidate": candidate.get("candidate_id"),
                    "from_sp": current_sp, "to_sp": wider[0]["candidate_id"],
                    "from_token_keep_ratio": current_token, "to_token_keep_ratio": current_token,
                    "reused_token_source": last_token["candidate_id"],
                }
                return [fallback]
        return result
    for axis in axes:
        if axis == "wider_structured_pruning" and current_sp:
            ordered = [row for row in sp_sources if row.get("candidate_id") != current_sp]
            current_depth = float((candidate.get("structured_pruning") or {}).get("actual_depth", 0.0))
            wider = [row for row in ordered if float(row.get("actual_depth", 0.0)) < current_depth]
            if wider:
                result.append(
                    make_candidate(
                        model_digest=candidate["model_digest"],
                        sp=wider[0],
                        quantization=candidate.get("quantization"),
                        token=candidate.get("vision_token"),
                        token_keep_ratio=current_token,
                        source_ids=(
                            wider[0].get("candidate_id"),
                            *candidate.get("source_ids", ()),
                        ),
                    )
                )
        elif axis == "higher_token_keep_ratio" and current_token is not None:
            higher = [
                row for row in token_sources
                if float(row.get("ratio", 0.0)) > float(current_token)
            ]
            if higher:
                higher.sort(key=lambda row: (float(row["ratio"]), row.get("candidate_id", "")))
                result.append(
                    make_candidate(
                        model_digest=candidate["model_digest"],
                        sp=candidate.get("structured_pruning"),
                        quantization=candidate.get("quantization"),
                        token=higher[0],
                        token_keep_ratio=higher[0]["ratio"],
                        source_ids=(
                            higher[0].get("candidate_id"),
                            *candidate.get("source_ids", ()),
                        ),
                    )
                )
        elif axis == "disable_structured_pruning" and candidate.get("structured_pruning") is not None:
            result.append(
                make_candidate(
                    model_digest=candidate["model_digest"],
                    sp=None,
                    quantization=candidate.get("quantization"),
                    token=candidate.get("vision_token"),
                    token_keep_ratio=current_token,
                    source_ids=candidate.get("source_ids", ()),
                )
            )
        elif axis == "disable_quantization" and candidate.get("quantization") is not None:
            result.append(
                make_candidate(
                    model_digest=candidate["model_digest"],
                    sp=candidate.get("structured_pruning"),
                    quantization=None,
                    token=candidate.get("vision_token"),
                    token_keep_ratio=current_token,
                    source_ids=candidate.get("source_ids", ()),
                )
            )
        # Vision Token is required by the current Stack Worker contract. A
        # disable_vision_token neighbor is therefore recorded in the policy
        # but not scheduled until a no-Token Worker path exists.
    return result


def _retain_attempt_diagnostics(attempt: Path, candidate_dir: Path) -> None:
    """Keep each retry's diagnostics before disposable Stack work is removed."""
    history = candidate_dir / "evidence"
    number = 1 + max((int(path.name.removeprefix("attempt-"))
                      for path in history.glob("attempt-*")
                      if path.name.removeprefix("attempt-").isdigit()), default=0)
    retained = history / f"attempt-{number}"
    for name in ("request.json", "worker.log", "probe.json", "result.json",
                 "quality", "benchmark", "vllm-benchmark", "evidence"):
        source = attempt / name
        if not source.exists():
            continue
        retained.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, retained / name)
        else:
            shutil.copy2(source, retained / name)
    for directory in ("parent", "artifact"):
        for name in ("config.json", "manifest.json", "stack-parent.json",
                     "precision-audit.json", "vision-quantization.json"):
            source = attempt / directory / name
            if source.is_file():
                destination = retained / (directory + "-metadata")
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / name)


def _retain_evaluation_outputs(attempt: Path, candidate_dir: Path, probe: dict[str, Any]) -> None:
    """Retain raw answers/timings before disposable Worker files are removed."""
    for name in ("quality", "benchmark"):
        source = attempt / name
        if not source.is_dir():
            continue
        destination = candidate_dir / name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        receipt = probe.get(name)
        if isinstance(receipt, dict):
            receipt["report"] = str((destination / "summary.json").resolve())
            if name == "quality":
                receipt["outputs"] = {
                    stage: str((destination / f"{stage}.jsonl").resolve())
                        for stage in ("search", "dev")
                        if (destination / f"{stage}.jsonl").is_file()
                }


def _lmms_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("lmms", {})
    if settings is True:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("Stack lmms must be an object")
    return settings


def _run_lmms_gate(
    *, candidate: dict[str, Any], candidate_dir: Path,
    probe: dict[str, Any], snap: dict[str, Any]
) -> dict[str, Any]:
    """Run native LMMS after mechanism/performance validation and return a durable receipt."""
    settings = _lmms_settings(snap["config"])
    if settings.get("enabled", False) is not True:
        return {"status": "disabled"}
    artifact = probe.get("artifact") or str((candidate_dir / "artifact").resolve())
    if not Path(artifact).is_dir():
        return {"status": "blocked", "reason": "lmms_artifact_missing"}
    output = candidate_dir / settings.get("output_subdir", "lmms")
    summary_path = output / "summary.json"
    descriptor = dict(snap["model"])
    descriptor.update({
        "path": str(Path(artifact).resolve()),
        "device": "cuda:0",
        "stack_adapter": True,
        "token_keep_ratio": candidate.get("token_keep_ratio"),
        "token_selection": snap["token_config"]["selection"],
    })
    if settings.get("native_worker_python"):
        descriptor["native_worker_python"] = str(_resolve(settings["native_worker_python"]))
    descriptor_path = candidate_dir / "lmms-model.json"
    worker_python = _resolve(settings.get("worker_python", ".venv-quant-gptqmodel/bin/python"))
    command = [
        str(worker_python), "-B", "-m", "search.lmms",
        "--model-config", str(descriptor_path), "--output", str(output),
        "--tasks", ",".join(row["task"] for row in settings.get("tasks", DEFAULT_TASKS)),
    ]
    if output.exists():
        command.append("--resume")
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = environment.get(variable)
        try:
            valid = value is not None and int(value) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            environment.pop(variable, None)
    environment["CUDA_VISIBLE_DEVICES"] = str(snap["config"]["device"])
    try:
        from search.lmms import _task_complete

        if descriptor_path.is_file() and read(descriptor_path) != descriptor:
            raise ValueError("LMMS descriptor changed; use a matching/new Stack job")
        if not descriptor_path.is_file():
            save_json(descriptor_path, descriptor)
        if summary_path.is_file():
            for row in settings.get("tasks", DEFAULT_TASKS):
                if not _task_complete(output, row["task"], descriptor):
                    raise ValueError("Saved LMMS summary lacks complete native evidence")
            return aggregate_lmms(
                output, _resolve(settings.get("baseline_root", result_root(snap.get("evaluation", {})) / "lmms/dense")),
                tasks=settings.get("tasks", DEFAULT_TASKS),
                minimum_retention=float(settings.get("minimum_retention", .95)),
                token_keep_ratio=candidate.get("token_keep_ratio"),
                candidate_id=candidate.get("candidate_id"),
                model_config=str(descriptor_path.resolve()),
            )
        if settings.get("smoke_samples_per_task") is not None:
            smoke_output = candidate_dir / "lmms-smoke"
            smoke_command = [arg for arg in command if arg != "--resume"]
            smoke_command[smoke_command.index("--output") + 1] = str(smoke_output)
            smoke_command.extend(["--limit", str(settings["smoke_samples_per_task"])])
            if smoke_output.exists():
                smoke_command.append("--resume")
            subprocess.run(smoke_command, cwd=PROJECT_ROOT, env=environment, check=True)
            for row in settings.get("tasks", DEFAULT_TASKS):
                if not _task_complete(smoke_output, row["task"], descriptor):
                    raise ValueError(f"Native Stack smoke lacks complete mechanism evidence: {row['task']}")
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        summary = aggregate_lmms(
            output,
            _resolve(settings.get("baseline_root", result_root(snap.get("evaluation", {})) / "lmms/dense")),
            tasks=settings.get("tasks", DEFAULT_TASKS),
            minimum_retention=float(settings.get("minimum_retention", 0.95)),
            token_keep_ratio=candidate.get("token_keep_ratio"),
            candidate_id=candidate.get("candidate_id"),
            model_config=str(descriptor_path.resolve()),
        )
        save_json(summary_path, summary)
        return summary
    except (OSError, subprocess.CalledProcessError, ValueError, TypeError, KeyError) as error:
        receipt = {
            "schema": "a4fvlm.lmms-gate.v1", "status": "blocked",
            "reason": f"{type(error).__name__}: {error}",
            "candidate_id": candidate.get("candidate_id"),
            "model_config": str(descriptor_path.resolve()),
        }
        save_json(candidate_dir / "lmms-blocked.json", receipt)
        return receipt


def _lmms_quality_rejected(receipt: dict[str, Any] | None) -> bool:
    return (isinstance(receipt, dict) and receipt.get("status") == "completed"
            and receipt.get("quality_gate") == "quality_rejected"
            and isinstance(receipt.get("retention"), (int, float))
            and math.isfinite(receipt["retention"]))


def _lmms_blocked(receipt: dict[str, Any] | None) -> bool:
    """Return whether LMMS could not produce a valid quality decision.

    A preflight or execution block is deliberately different from a quality
    rejection: it must be surfaced to the operator and must not silently move
    the quality ladder.
    """

    if not isinstance(receipt, dict):
        return False
    if receipt.get("status") == "blocked":
        return True
    # A completed-looking receipt without a gate is still unusable evidence;
    # classify it as an execution/preflight block rather than a quality miss.
    return (receipt.get("status") != "completed"
            or receipt.get("quality_gate") not in {"passed", "quality_rejected"}
            or not isinstance(receipt.get("retention"), (int, float))
            or not math.isfinite(receipt["retention"]))


def _lmms_required(config: dict[str, Any]) -> bool:
    return _lmms_settings(config).get("enabled", False) is True


def _completed_handoff(destination: Path, snap: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse an accepted Stack before traversing obsolete local rejections."""
    paths = [destination / name for name in ("config.json", "summary.json", "final/stack-freeze.json")]
    if not all(path.is_file() for path in paths):
        return None
    saved, summary, freeze = (read(path) for path in paths)
    if (saved.get("model_digest") != snap["model_digest"]
            or evidence_config(saved.get("config", {})) != evidence_config(snap["config"])):
        return None
    if not stack_handoff_eligible(freeze) or (
        freeze.get("status") == "frozen_best_available" and selection_policy(snap["config"]) != "best_effort"
    ):
        return None
    candidate = next((row for row in snap["candidates"]
                      if row["candidate_id"] == freeze.get("candidate_id")
                      and row["candidate_key"] == freeze.get("candidate_key")), None)
    if candidate is None or summary.get("selected_candidate") != candidate["candidate_id"]:
        return None
    directory = destination / "candidates" / candidate["candidate_id"]
    probe = read(directory / "probe.json")
    validate_result(probe)
    from skills.stack_vllm import validate_stack_artifact
    validate_stack_artifact(Path(freeze["artifact"]))
    receipt = _run_lmms_gate(candidate=candidate, candidate_dir=directory, probe=probe, snap=snap)
    if receipt != freeze["lmms"] or not stack_handoff_eligible(freeze | {"lmms": receipt}):
        raise ValueError("Saved Stack handoff no longer has applicable full LMMS evidence")
    return summary


def execute(config_path: str | Path = "configs/stack.json", *, output: str | Path | None = None) -> dict[str, Any]:
    snap = snapshot(config_path)
    policy = selection_policy(snap["config"])
    destination = _resolve(output or snap["config"]["output"])
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".run.lock"
    with lock(lock_path):
        completed = _completed_handoff(destination, snap)
        if completed is not None:
            lock_path.unlink(missing_ok=True)
            return completed
        # Keep old local-gate search history intact when continuing in-place.
        old_state = destination / "state.json"
        if old_state.is_file() and read(old_state).get("quality_policy") != "full_lmms":
            history = destination / "legacy-local-quality"
            history.mkdir(exist_ok=True)
            for name in ("config.json", "state.json", "summary.json"):
                source = destination / name
                if source.is_file() and not (history / name).exists():
                    shutil.copy2(source, history / name)
        save_json(
            destination / "config.json",
            {
                "schema": snap["schema"],
                "config": snap["config"],
                "model_digest": snap["model_digest"],
                "blockers": snap["blockers"],
                "candidate_ids": [row["candidate_id"] for row in snap["candidates"]],
                "initial_candidate_ids": [
                    row["candidate_id"] for row in snap["initial_candidates"]
                ],
            },
        )
        state_path = destination / "state.json"
        state = read(state_path) if state_path.is_file() else {"status": "running", "candidates": {}, "attempts": []}
        state["status"] = "running"
        state["stop_reason"] = None
        state["selected_candidate"] = None
        state["config_path"] = snap["config_path"]
        state["quality_policy"] = "full_lmms"
        state.setdefault("candidates", {})
        state.setdefault("attempts", [])
        save_json(state_path, state)
        save_json(destination / "summary.json", {
            "status": "running", "checkpoint": "state.json",
            "selected_candidate": None, "stop_reason": None,
        })
        if snap["status"] != "ready":
            state["status"] = "blocked"
            state["blockers"] = snap["blockers"]
            state["stop_reason"] = "blocked_preflight"
            save_json(state_path, state)
            summary = {
                "status": "blocked",
                "checkpoint": "state.json",
                "blockers": snap["blockers"],
                "stop_reason": "blocked_preflight",
            }
            save_json(destination / "summary.json", summary)
            lock_path.unlink(missing_ok=True)
            return summary
        machine = hardware(snap["config"]["device"], snap["model"])
        work = destination / ".work"
        for interrupted in sorted(work.glob("*")):
            if interrupted.is_dir():
                _retain_attempt_diagnostics(interrupted, destination / "candidates" / interrupted.name)
        shutil.rmtree(work, ignore_errors=True)
        rows = []
        pending = list(snap["initial_candidates"])
        seen_candidates: set[str] = set()
        candidate_records: dict[str, dict[str, Any]] = {}
        lmms_quality_rejected = False
        blocked = False
        execution_failed = False
        while pending and len(rows) < int(snap["config"]["limits"]["total_candidates"]):
            candidate = pending.pop(0)
            if candidate["candidate_key"] in seen_candidates:
                continue
            seen_candidates.add(candidate["candidate_key"])
            candidate_records[candidate["candidate_id"]] = candidate
            candidate_dir = destination / "candidates" / candidate["candidate_id"]
            prior_result = candidate_dir / "result.json"
            prior_probe = candidate_dir / "probe.json"
            runtime_validation = None
            export_validation = None
            if prior_result.is_file() and prior_probe.is_file():
                try:
                    prior = read(prior_result)
                    probe = read(prior_probe)
                    reusable = (
                        prior.get("candidate_key") == candidate["candidate_key"]
                        and probe.get("candidate_key") == candidate["candidate_key"]
                        and prior.get("probe_status") in {"qualified", "development_qualified", "stack_bridge_blocked", "stack_runtime_blocked", "ready_for_lmms", "mechanism_ready"}
                    )
                    needs_vllm = (
                        snap["config"].get("runtime", {}).get("vllm_reload", False)
                        and (
                            probe.get("reload", {}).get("vllm", {}).get("status")
                            != "passed"
                            or (
                                snap["config"].get("runtime", {}).get(
                                    "vllm_benchmark", False
                                )
                                and probe.get("vllm_benchmark", {}).get("status")
                                != "passed"
                            )
                            or (
                                candidate.get("token_keep_ratio") is not None
                                and not _score_token_benchmark_ready(probe)
                            )
                        )
                    )
                    if reusable and prior.get("probe_status") not in {"stack_bridge_blocked", "stack_runtime_blocked"}:
                        try:
                            validate_result(probe)
                        except (ValueError, TypeError):
                            reusable = False
                    if reusable and not needs_vllm:
                        lmms = probe.get("lmms")
                        if (
                            probe.get("status") in {"qualified", "ready_for_lmms"}
                            and _lmms_required(snap["config"])
                        ):
                            state["active_candidate"] = candidate["candidate_id"]
                            state["active_stage"] = "lmms"
                            state["candidates"][candidate["candidate_id"]] = prior
                            save_json(state_path, state)
                            lmms = _run_lmms_gate(
                                candidate=candidate,
                                candidate_dir=candidate_dir,
                                probe=probe,
                                snap=snap,
                            )
                            probe["lmms"] = lmms
                            save_json(candidate_dir / "probe.json", probe)
                            prior["lmms"] = lmms
                            save_json(candidate_dir / "result.json", prior)
                        rows.append({
                            "candidate_id": candidate["candidate_id"],
                            "source_ids": candidate.get("source_ids", []),
                            "token_keep_ratio": candidate.get("token_keep_ratio"),
                            **prior,
                            "reused": True,
                        })
                        state["candidates"][candidate["candidate_id"]] = prior
                        save_json(state_path, state)
                        if policy == "best_effort" and not complete_lmms(lmms):
                            blocked = True
                        elif _lmms_quality_rejected(lmms):
                            lmms_quality_rejected = True
                            neighbors = _recovery_neighbors(candidate, snap)
                            if neighbors:
                                pending.extend(neighbors)
                        elif (_lmms_blocked(lmms) or prior.get("probe_status") in
                              {"stack_bridge_blocked", "stack_runtime_blocked"}):
                            blocked = True
                        if blocked or (lmms or {}).get("quality_gate") == "passed":
                            break
                        continue
                    if reusable and needs_vllm and probe.get("artifact"):
                        runtime_validation = (prior, probe)
                    if (prior.get("candidate_key") == candidate["candidate_key"]
                            and probe.get("candidate_key") == candidate["candidate_key"]
                            and prior.get("probe_status") in {"candidate_error", "quality_rejected"}
                            and probe.get("mechanism", {}).get("status") == "passed"
                            and (candidate_dir / "artifact/manifest.json").is_file()
                            and (candidate_dir / "export-request.json").is_file()):
                        export_validation = probe
                except (OSError, ValueError, TypeError):
                    pass
            if prior_probe.is_file() and read(prior_probe).get("status") == "quality_rejected":
                history = candidate_dir / "legacy-local-quality"
                history.mkdir(exist_ok=True)
                for source in (prior_probe, prior_result):
                    if source.is_file() and not (history / source.name).exists():
                        shutil.copy2(source, history / source.name)
            attempt = work / candidate["candidate_id"]
            state["attempts"].append(str(attempt))
            request = {
                "operation": (
                    "stack_runtime_validation"
                    if runtime_validation is not None
                    else "stack_candidate"
                ),
                "device": snap["config"]["device"],
                "hardware": machine,
                "worker_python": (
                    (candidate.get("quantization") or {}).get("worker_python")
                    or snap["config"].get("quantization", {}).get("worker_python")
                ),
                "model": {**snap["model"], "device": "cuda:0"},
                "evaluation": snap["evaluation"],
                "selection": snap["token_config"]["selection"],
                "candidate": candidate,
                "runtime": snap["config"].get("runtime", {}),
                "calibration": snap["config"].get("quantization", {}).get("calibration"),
                "quantization_runtime": snap["config"].get("quantization", {}).get("runtime", {}),
                "probe_output": str(attempt),
            }
            if runtime_validation is not None:
                prior, probe = runtime_validation
                request["artifact"] = probe["artifact"]
                request["prior_probe"] = probe
            elif export_validation is not None:
                request["resume_artifact"] = str((candidate_dir / "artifact").resolve())
                request["export_request"] = str((candidate_dir / "export-request.json").resolve())
                request["prior_probe"] = export_validation
            state["active_candidate"] = candidate["candidate_id"]
            state["active_stage"] = request["operation"]
            save_json(state_path, state)
            try:
                result = launch(request, attempt)
            except Exception as error:  # noqa: BLE001
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _retain_attempt_diagnostics(attempt, candidate_dir)
            probe_source = attempt / "probe.json"
            result_source = attempt / "result.json"
            durable_artifact = None
            if isinstance(result, dict):
                raw_artifact = result.get("artifact")
                if raw_artifact and Path(raw_artifact).is_dir():
                    durable_artifact = candidate_dir / "artifact"
                    if Path(raw_artifact).resolve() != durable_artifact.resolve():
                        shutil.rmtree(durable_artifact, ignore_errors=True)
                        shutil.copytree(raw_artifact, durable_artifact)
                    else:
                        durable_artifact = Path(raw_artifact)
                    if runtime_validation is None and export_validation is None:
                        save_json(candidate_dir / "export-request.json", request)
            if probe_source.is_file():
                probe = read(probe_source)
                if candidate.get("recovery") is not None:
                    probe.setdefault("recovery", candidate["recovery"])
                _retain_evaluation_outputs(attempt, candidate_dir, probe)
                if durable_artifact is not None:
                    probe["artifact"] = str(durable_artifact.resolve())
                    mechanism = probe.get("mechanism")
                    if isinstance(mechanism, dict):
                        mechanism["artifact"] = str(durable_artifact.resolve())
                        if mechanism.get("manifest"):
                            mechanism["manifest"] = str(
                                (durable_artifact / "manifest.json").resolve()
                            )
                benchmark_source = attempt / "vllm-benchmark"
                if benchmark_source.is_dir():
                    benchmark_destination = candidate_dir / "vllm-benchmark"
                    shutil.rmtree(benchmark_destination, ignore_errors=True)
                    shutil.copytree(benchmark_source, benchmark_destination)
                    vllm_benchmark = probe.get("vllm_benchmark")
                    if isinstance(vllm_benchmark, dict):
                        vllm_benchmark["report"] = str(
                            (benchmark_destination / "benchmark.json").resolve()
                        )
                save_json(candidate_dir / "probe.json", probe)
            if result_source.is_file():
                receipt = read(result_source)
                receipt["candidate_key"] = candidate["candidate_key"]
                if candidate.get("recovery") is not None:
                    receipt.setdefault("recovery", candidate["recovery"])
                if probe_source.is_file():
                    receipt["report"] = str((candidate_dir / "probe.json").resolve())
                if durable_artifact is not None:
                    receipt["artifact"] = str(durable_artifact.resolve())
                save_json(candidate_dir / "result.json", receipt)
                result = receipt
            else:
                result["candidate_key"] = candidate["candidate_key"]
                if probe_source.is_file():
                    result["report"] = str((candidate_dir / "probe.json").resolve())
                if durable_artifact is not None:
                    result["artifact"] = str(durable_artifact.resolve())
            # Persist mechanism/performance readiness before a long LMMS subprocess. A
            # killed coordinator can resume without rebuilding/requantizing.
            save_json(candidate_dir / "result.json", result)
            state["candidates"][candidate["candidate_id"]] = result
            if probe_source.is_file() and probe.get("status") in {"qualified", "ready_for_lmms"} and _lmms_required(snap["config"]):
                state["active_candidate"] = candidate["candidate_id"]
                state["active_stage"] = "lmms"
                save_json(state_path, state)
                probe["lmms"] = _run_lmms_gate(
                    candidate=candidate, candidate_dir=candidate_dir,
                    probe=probe, snap=snap,
                )
                save_json(candidate_dir / "probe.json", probe)
            if probe_source.is_file() and isinstance(probe.get("lmms"), dict):
                result["lmms"] = probe["lmms"]
                save_json(candidate_dir / "result.json", result)
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "source_ids": candidate.get("source_ids", []),
                "token_keep_ratio": candidate.get("token_keep_ratio"),
                **result,
            })
            state["candidates"][candidate["candidate_id"]] = result
            lmms = probe.get("lmms") if probe_source.is_file() else None
            if policy == "best_effort" and not complete_lmms(lmms):
                blocked = True
            elif _lmms_quality_rejected(lmms):
                lmms_quality_rejected = True
                neighbors = _recovery_neighbors(candidate, snap)
                if neighbors:
                    pending.extend(neighbors)
            elif _lmms_blocked(lmms):
                blocked = True
            elif result.get("probe_status") in {"stack_runtime_blocked", "stack_bridge_blocked", "blocked_preflight"}:
                blocked = True
            elif result.get("probe_status") not in {"qualified", "development_qualified", "ready_for_lmms", "mechanism_ready"}:
                execution_failed = True
            if blocked or execution_failed or (lmms or {}).get("quality_gate") == "passed":
                break
        shutil.rmtree(work, ignore_errors=True)
        state["status"] = "completed"
        state["active_candidate"] = None
        state["active_stage"] = None
        # ``pending`` may contain duplicate recovery requests that were
        # already seen (for example after a resumed run).  Only an unseen
        # request means that the configured budget, rather than the recovery
        # ladder, stopped the loop.
        pending_unseen = any(
            isinstance(item, dict)
            and item.get("candidate_key") not in seen_candidates
            for item in pending
        )
        candidate_budget = int(snap["config"]["limits"]["total_candidates"])
        if blocked:
            stop_reason = "blocked_preflight"
        elif execution_failed:
            stop_reason = "execution_error"
        elif len(rows) >= candidate_budget and pending_unseen:
            stop_reason = "candidate_budget_exhausted"
        elif lmms_quality_rejected:
            stop_reason = "recovery_ladder_exhausted"
        else:
            stop_reason = "candidate_ladder_exhausted"
        state["stop_reason"] = stop_reason
        state["quality_rejections"] = {
            "lmms": lmms_quality_rejected,
        }
        qualified = []
        measured = []
        for row in rows:
            candidate_dir = destination / "candidates" / row["candidate_id"]
            probe_path = candidate_dir / "probe.json"
            if probe_path.is_file():
                try:
                    probe = read(probe_path)
                except (OSError, ValueError, TypeError):
                    continue
                if probe.get("status") not in {"qualified", "ready_for_lmms"}:
                    continue
                try:
                    validate_result(probe)
                except (OSError, TypeError, ValueError):
                    continue
                if (
                    snap["config"].get("runtime", {}).get("vllm_reload", False)
                    and probe.get("reload", {}).get("vllm", {}).get("status") != "passed"
                ):
                    continue
                if (
                    snap["config"].get("runtime", {}).get("vllm_benchmark", False)
                    and probe.get("vllm_benchmark", {}).get("status") != "passed"
                ):
                    continue
                if _lmms_required(snap["config"]):
                    lmms = probe.get("lmms")
                    if (complete_lmms(lmms) and probe.get("artifact") and Path(probe["artifact"]).is_dir()):
                        measured.append({"candidate_id": row["candidate_id"], **probe})
                    if not isinstance(lmms, dict) or lmms.get("quality_gate") != "passed":
                        continue
                qualified.append({"candidate_id": row["candidate_id"], **probe})
        selected = select_candidate(qualified)
        tradeoffs = quality_speed_frontier([
            {"candidate_id": p["candidate_id"], "retention": p["lmms"]["retention"],
             "mean_seconds": p["benchmark"].get("mean_seconds"),
             "timing_scope": "stack_transformers_benchmark_round",
             "probe": str(destination / "candidates" / p["candidate_id"] / "probe.json")}
            for p in measured
        ])
        exhausted = stop_reason in {"candidate_budget_exhausted", "recovery_ladder_exhausted"}
        if policy == "best_effort" and not (blocked or execution_failed):
            choice = select_measured(tradeoffs, _lmms_settings(snap["config"]).get("minimum_retention", .95))
            if choice and (choice["retention"] >= _lmms_settings(snap["config"]).get("minimum_retention", .95) or exhausted):
                selected = next(p for p in measured if p["candidate_id"] == choice["candidate_id"])
            else:
                selected = None
        target_met = bool(selected and selected.get("lmms", {}).get("quality_gate") == "passed")
        if target_met:
            stop_reason = "lmms_quality_passed"
        selected_id = selected.get("candidate_id") if selected else None
        selected_source = candidate_records.get(selected_id) if selected_id else None
        if selected and selected_source:
            for key in (
                "source_ids", "skills", "token_keep_ratio", "structured_pruning",
                "quantization", "vision_token", "recovery",
            ):
                if key in selected_source:
                    selected[key] = selected_source[key]
        if selected and selected.get("artifact"):
            artifact = Path(selected["artifact"])
            if artifact.is_dir():
                final_root = destination / "final"
                final = final_root / "artifact"
                shutil.rmtree(final, ignore_errors=True)
                shutil.copytree(artifact, final)
                save_json(
                    final_root / "stack-freeze.json",
                    {
                        "schema": "a4fvlm.stack-freeze.v1",
                        "status": "frozen_for_lmms" if target_met else "frozen_best_available",
                        "selection_policy": policy,
                        "quality_target_met": target_met,
                        "search_stop_reason": stop_reason,
                        "candidate_id": selected.get("candidate_id"),
                        "candidate_key": selected.get("candidate_key"),
                        "source_ids": selected.get("source_ids", []),
                        "skills": selected.get("skills", []),
                        "token_keep_ratio": selected.get("token_keep_ratio"),
                        "artifact": str(final.resolve()),
                        "quality": selected.get("quality"),
                        "benchmark": selected.get("benchmark"),
                        "lmms": selected.get("lmms"),
                        "metrics_transferable": False,
                    },
                )
        if selected:
            _write_vllm_comparison(destination, selected)
        final_stop_reason = (
            "best_available_selected" if selected and not target_met and _lmms_required(snap["config"])
            else "lmms_quality_passed" if selected and _lmms_required(snap["config"])
            else "quality_passed" if selected else stop_reason
        )
        state["selected_candidate"] = selected_id
        state["stop_reason"] = final_stop_reason
        state["search_stop_reason"] = stop_reason
        state["quality_target_met"] = target_met
        save_json(state_path, state)
        summary = {
            "status": "completed",
            "checkpoint": "state.json",
            "candidate_count": len(rows),
            "results": rows,
            "selected_candidate": selected_id,
            "metrics_transferable": False,
            "lmms_enabled": _lmms_required(snap["config"]),
            "stop_reason": final_stop_reason,
            "search_stop_reason": stop_reason,
            "selection_policy": policy,
            "quality_target_met": target_met,
            "tradeoffs": tradeoffs,
        }
        save_json(destination / "summary.json", summary)
        lock_path.unlink(missing_ok=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stack.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = execute(args.config, output=args.output) if args.execute else plan(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
