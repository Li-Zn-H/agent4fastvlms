"""Shared planner/runner for runtime layers on a frozen Stack artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from search.optimize import hardware, launch, lock, read
from search.model import PROJECT_ROOT
from search.context import require_stages, dense_root
from search.vllm_benchmark import compare_reports
from search.worker import save_json
from skills.quantization import digest
from skills.stack_vllm import validate_stack_artifact
from search.lmms_gate import stack_handoff_eligible


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parent(config: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    path = _resolve(config["parent_freeze"])
    freeze = read(path)
    if not isinstance(freeze, dict):
        raise ValueError("parent freeze must be a JSON object")
    if freeze.get("status") not in {"frozen_for_lmms", "frozen_best_available", "runtime_frozen", "kv_cache_frozen"}:
        raise ValueError("parent freeze is not a completed handoff")
    artifact = Path(str(freeze.get("artifact", "")))
    if not artifact.is_absolute():
        artifact = _resolve(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(f"parent Stack artifact is missing: {artifact}")
    validate_stack_artifact(artifact)
    if kind == "runtime" and not stack_handoff_eligible(freeze):
        raise ValueError("Runtime requires an eligible complete Stack LMMS handoff")
    if kind == "kv_cache" and freeze.get("status") != "runtime_frozen":
        raise ValueError("KV Cache requires a runtime-frozen parent")
    if kind == "kv_cache" and freeze.get("token_keep_ratio") is None:
        raise ValueError("runtime freeze is missing the Stack token_keep_ratio handoff")
    return {
        "path": str(path.resolve()),
        "digest": digest(freeze),
        "freeze": freeze,
        "artifact": str(artifact.resolve()),
    }


def snapshot(config_path: str | Path, *, kind: str) -> dict[str, Any]:
    config_path = _resolve(config_path)
    config = read(config_path)
    if kind == "runtime":
        from skills.stack_runtime import validate_config
    elif kind == "kv_cache":
        from skills.stack_kv_cache import validate_config
    else:
        raise ValueError(f"unknown acceleration kind: {kind}")
    validate_config(config)
    parent = _parent(config, kind=kind)
    evaluation = read(_resolve(config["evaluation_config"]))
    require_stages({"path": parent["artifact"], "hardware": config.get("hardware", "cuda"),
                    "device": "cuda:0"}, (kind,), device=config["device"])
    if "result_root" in evaluation and _resolve(config["dense_benchmark"]).resolve() != (
        dense_root(evaluation) / "benchmark/summary.json"
    ).resolve():
        raise ValueError("Runtime/KV must use evaluation.result_root's Dense denominator")
    token_config = read(_resolve(config["vision_token_config"]))
    # KV Cache is a layer on top of the selected Runtime profile.  Carry the
    # frozen Runtime settings into every KV candidate, then let the KV config
    # override only fields it owns.  Without this merge, a KV benchmark would
    # silently fall back to vLLM defaults and would not measure the composed
    # Runtime + KV configuration.
    base_runtime: dict[str, Any] = {}
    if kind == "kv_cache":
        base_runtime.update(parent["freeze"].get("runtime", {}))
    base_runtime.update(config.get("runtime", {}))
    benchmark = dict(config.get("benchmark", {}))
    candidates = []
    for raw in config["candidates"]:
        candidate = dict(raw)
        candidate_runtime = dict(base_runtime)
        candidate_runtime.update(candidate.get("runtime", {}))
        if kind == "kv_cache":
            candidate_runtime["kv_cache_dtype"] = candidate.get("kv_cache_dtype", "auto")
        candidate["runtime"] = candidate_runtime
        candidate["candidate_key"] = (
            __import__(
                "skills.stack_runtime" if kind == "runtime" else "skills.stack_kv_cache",
                fromlist=["candidate_key"],
            ).candidate_key(parent["digest"], candidate)
        )
        candidates.append(candidate)
    return {
        "kind": kind,
        "config_path": str(config_path.resolve()),
        "config": config,
        "parent": parent,
        "evaluation": evaluation,
        "token_config": token_config,
        "benchmark": benchmark,
        "candidates": candidates,
    }


def plan(config_path: str | Path, *, kind: str) -> dict[str, Any]:
    try:
        snap = snapshot(config_path, kind=kind)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {
            "schema": "a4fvlm.stack-acceleration-plan.v1",
            "kind": kind,
            "status": "blocked",
            "blockers": [f"{type(error).__name__}: {error}"],
            "config": str(_resolve(config_path)),
        }
    return {
        "schema": "a4fvlm.stack-acceleration-plan.v1",
        "kind": kind,
        "status": "ready",
        "parent_freeze": snap["parent"]["path"],
        "parent_digest": snap["parent"]["digest"],
        "artifact": snap["parent"]["artifact"],
        "candidate_count": len(snap["candidates"]),
        "candidates": [
            {
                "id": row["id"],
                "candidate_key": row["candidate_key"],
                "runtime": row["runtime"],
            }
            for row in snap["candidates"]
        ],
        "output": str(_resolve(snap["config"]["output"])),
        "reuse_policy": "Reuse complete receipts; only missing candidates launch Workers.",
    }


def _profile_runtime(runtime: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(runtime)
    value["vllm_benchmark_limit"] = int(profile["limit"])
    value["vllm_benchmark_warmup"] = int(profile["warmup"])
    value["vllm_benchmark_repeats"] = int(profile["repeats"])
    value["vllm_benchmark_batch_size"] = int(profile["batch_size"])
    return value


def _profiles(snap: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmark = snap["benchmark"]
    kind = snap["kind"]
    warmup = int(benchmark.get("warmup", 5))
    repeats = int(benchmark.get("repeats", 3))
    if kind == "runtime":
        return [
            {
                "name": "search",
                "split": "search",
                "limit": int(benchmark.get("search_limit", 128)),
                "warmup": warmup,
                "repeats": repeats,
                "batch_size": int(benchmark.get("batch_size", 1)),
            }
        ]
    return [
        {
            "name": f"batch{batch_size}",
            "split": "search",
            "limit": int(benchmark.get("search_limit", 128)),
            "warmup": warmup,
            "repeats": repeats,
            "batch_size": batch_size,
        }
        for batch_size in benchmark.get("batch_sizes", [1, 4, 8])
    ]


def _copy_receipts(attempt: Path, candidate_dir: Path) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name in ("probe.json", "result.json"):
        source = attempt / name
        if source.is_file():
            shutil.copy2(source, candidate_dir / name)
    for source in attempt.glob("vllm-*"):
        if source.is_dir():
            destination = candidate_dir / source.name
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)
    _repair_receipt_paths(candidate_dir)


def _repair_receipt_paths(candidate_dir: Path) -> dict[str, Any] | None:
    """Rebase paths copied out of `.work` so evidence remains resumable."""

    probe_path = candidate_dir / "probe.json"
    if not probe_path.is_file():
        return None
    probe = read(probe_path)
    for details in (probe.get("benchmarks", {}) or {}).values():
        if isinstance(details, dict) and details.get("report"):
            report_dir = Path(str(details["report"])).parent.name
            details["report"] = str(
                (candidate_dir / report_dir / "benchmark.json").resolve()
            )
    quality = probe.get("quality")
    if isinstance(quality, dict):
        if quality.get("report"):
            report_dir = Path(str(quality["report"])).parent.name
            quality["report"] = str(
                (candidate_dir / report_dir / "benchmark.json").resolve()
            )
        if quality.get("outputs"):
            output_dir = Path(str(quality["outputs"])).parent.name
            quality["outputs"] = str(
                (candidate_dir / output_dir / Path(str(quality["outputs"])).name).resolve()
            )
    save_json(probe_path, probe)
    return probe


def _reuse(candidate_dir: Path, candidate_key: str) -> dict[str, Any] | None:
    probe_path = candidate_dir / "probe.json"
    result_path = candidate_dir / "result.json"
    if not probe_path.is_file() or not result_path.is_file():
        return None
    try:
        probe = _repair_receipt_paths(candidate_dir) or read(probe_path)
        result = read(result_path)
    except (OSError, ValueError, TypeError):
        return None
    if probe.get("candidate_key") != candidate_key or result.get("candidate_key") != candidate_key:
        return None
    if probe.get("status") not in {"performance_ready", "qualified", "quality_rejected", "blocked_preflight", "runtime_blocked"}:
        return None
    from skills.stack_runtime import performance_ready
    if probe.get("status") in {"performance_ready", "qualified"} and not performance_ready(probe):
        return None
    if probe.get("status") == "quality_rejected":
        # Local Dev cannot veto this measured candidate under the new policy.
        measured = {**probe, "status": "performance_ready"}
        if not performance_ready(measured):
            return None
        history = candidate_dir / "legacy-local-quality"
        history.mkdir(exist_ok=True)
        for path in (probe_path, result_path):
            if not (history / path.name).exists():
                shutil.copy2(path, history / path.name)
        measured["legacy_quality"] = probe.get("quality")
        measured["quality"] = {"status": "pending_full_lmms"}
        probe = measured
    return probe


def _parent_report(snap: Mapping[str, Any]) -> dict[str, Any] | None:
    freeze = snap["parent"]["freeze"]
    if snap["kind"] == "runtime":
        paths = freeze.get("runtime_receipts", {})
        path = snap["config"].get("parent_benchmark") or paths.get("search128")
    else:
        # A KV candidate's batch-1 profile is compared with the selected
        # Runtime Search profile; the Runtime freeze has no batch-1 key.
        path = freeze.get("benchmarks", {}).get("search", {}).get("report")
    if not path:
        return None
    path = _resolve(path)
    try:
        value = read(path)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _comparison_profile(kind: str) -> str:
    """Return the candidate profile comparable with the frozen parent."""

    return "search" if kind == "runtime" else "batch1"


def _dense_comparison(snap: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compare measured serving profiles with the retained Transformers Dense run.

    Normalize across repeat counts. Batch-4/8 throughput is a serving metric,
    never a single-request latency claim. Timing boundaries are recorded since
    the Dense end-to-end timer includes input preparation outside vLLM's timer.
    """
    source = snap["config"].get("dense_benchmark")
    if not source:
        return None
    path = _resolve(source)
    dense = read(path)
    rounds = dense["rounds"]
    expected_count = int(snap["benchmark"].get("search_limit", 128))
    if not rounds or any(row["sample_count"] != expected_count for row in rounds):
        raise ValueError("Dense benchmark sample count does not match Search profile")
    seconds = sum(row["total_seconds"] for row in rounds)
    tokens = sum(row["output_tokens"] for row in rounds)
    count = sum(row["sample_count"] for row in rounds)
    if seconds <= 0 or tokens <= 0 or count <= 0:
        raise ValueError("Dense benchmark requires positive time, tokens and sample counts")
    profiles = {}
    for name, details in probe.get("benchmarks", {}).items():
        current = details["summary"]
        if current["elapsed_seconds"] <= 0 or current["request_count"] <= 0:
            raise ValueError("Candidate benchmark requires positive time and sample count")
        batch = details.get("runtime", {}).get("vllm_benchmark_batch_size", 1)
        profiles[name] = {
            "scope": "single_request" if batch == 1 else "concurrent_serving",
            "batch_size": batch,
            "request_throughput_speedup": (current["request_count"] / current["elapsed_seconds"]) / (count / seconds),
            "output_token_throughput_speedup": current["output_token_throughput"] / (tokens / seconds),
            "output_token_throughput": current["output_token_throughput"],
        }
    return {
        "baseline": "transformers_dense",
        "baseline_report": str(path.resolve()),
        "baseline_report_digest": digest(dense),
        "benchmark_split": "search",
        "sample_count_per_round": expected_count,
        "dense_rounds": len(rounds),
        "dense_output_token_throughput": tokens / seconds,
        "dense_request_throughput": count / seconds,
        "timing_scope": "transformers_end_to_end_vs_vllm_engine_generate",
        "profiles": profiles,
    }


def _freeze(snap: Mapping[str, Any], selected: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    kind = snap["kind"]
    status = "runtime_frozen" if kind == "runtime" else "kv_cache_frozen"
    schema = "a4fvlm.stack-runtime-freeze.v1" if kind == "runtime" else "a4fvlm.stack-kv-cache-freeze.v1"
    freeze = {
        "schema": schema,
        "status": status,
        "parent_freeze": snap["parent"]["path"],
        "parent_digest": snap["parent"]["digest"],
        "candidate_id": selected["candidate_id"],
        "candidate_key": selected["candidate_key"],
        "artifact": snap["parent"]["artifact"],
        # Runtime layers must keep the Stack's Token mechanism enabled.  The
        # freeze is the handoff consumed by the next layer, so dropping this
        # field would silently benchmark physical SP + GPTQ without Token.
        "token_keep_ratio": snap["parent"]["freeze"].get("token_keep_ratio"),
        "runtime": selected.get("runtime", {}),
        "benchmarks": selected.get("benchmarks", {}),
        "quality": selected.get("quality", {}),
        "comparison": selected.get("comparison"),
        "dense_comparison": selected.get("dense_comparison"),
    }
    final = destination / "final"
    final.mkdir(parents=True, exist_ok=True)
    save_json(final / ("runtime-freeze.json" if kind == "runtime" else "kv-cache-freeze.json"), freeze)
    return freeze


def execute(config_path: str | Path, *, kind: str, output: str | Path | None = None) -> dict[str, Any]:
    snap = snapshot(config_path, kind=kind)
    # Fail before launching a Worker if an explicitly configured denominator
    # is missing or cannot be compared with the requested Search profile.
    _dense_comparison(snap, {"benchmarks": {}})
    destination = _resolve(output or snap["config"]["output"])
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".run.lock"
    with lock(lock_path):
        save_json(
            destination / "config.json",
            {
                "schema": "a4fvlm.stack-acceleration.v1",
                "kind": kind,
                "config": snap["config"],
                "parent_digest": snap["parent"]["digest"],
                "candidate_ids": [row["id"] for row in snap["candidates"]],
            },
        )
        state_path = destination / "state.json"
        state = read(state_path) if state_path.is_file() else {"status": "running", "candidates": {}}
        state.update({"status": "running", "kind": kind, "parent_digest": snap["parent"]["digest"], "active_candidate": None})
        state.setdefault("candidates", {})
        save_json(state_path, state)
        work = destination / ".work"
        shutil.rmtree(work, ignore_errors=True)
        rows: list[dict[str, Any]] = []
        parent_report = _parent_report(snap)
        comparison_profile = _comparison_profile(kind)
        machine = hardware(snap["config"]["device"])
        for candidate in snap["candidates"][: int(snap["config"].get("limits", {}).get("total_candidates", len(snap["candidates"])))]:
            candidate_dir = destination / "candidates" / candidate["id"]
            reused = _reuse(candidate_dir, candidate["candidate_key"])
            if reused is not None:
                if parent_report:
                    report = reused.get("benchmarks", {}).get(comparison_profile, {}).get("report")
                    if report:
                        try:
                            reused["comparison"] = compare_reports(parent_report, read(_resolve(report)))
                            save_json(candidate_dir / "probe.json", reused)
                        except (OSError, ValueError, TypeError, KeyError):
                            pass
                rows.append(reused)
                reused["dense_comparison"] = _dense_comparison(snap, reused)
                save_json(candidate_dir / "probe.json", reused)
                state["candidates"][candidate["id"]] = reused
                save_json(state_path, state)
                continue
            attempt = work / candidate["id"]
            request = {
                "operation": "stack_runtime_candidate" if kind == "runtime" else "stack_kv_cache_candidate",
                "device": snap["config"]["device"],
                "hardware": machine,
                "worker_python": str(_resolve(snap["config"].get("worker_python", ".venv-quant-gptqmodel/bin/python"))),
                "artifact": snap["parent"]["artifact"],
                "parent_freeze": snap["parent"]["path"],
                "parent_digest": snap["parent"]["digest"],
                "candidate": candidate,
                "runtime": candidate["runtime"],
                "token_keep_ratio": snap["parent"]["freeze"].get("token_keep_ratio"),
                "profiles": _profiles(snap),
                "evaluation": snap["evaluation"],
                "selection": snap["token_config"]["selection"],
                "probe_output": str(attempt),
            }
            state.setdefault("attempts", []).append(str(attempt))
            state["active_candidate"] = candidate["id"]
            save_json(state_path, state)
            try:
                launch(request, attempt)
            except Exception as error:  # noqa: BLE001 - receipt is persisted below
                attempt.mkdir(parents=True, exist_ok=True)
                save_json(attempt / "probe.json", {
                    "schema": "a4fvlm.stack-acceleration-result.v1",
                    "candidate_id": candidate["id"],
                    "candidate_key": candidate["candidate_key"],
                    "status": "runtime_blocked",
                    "error": f"{type(error).__name__}: {error}",
                })
            _copy_receipts(attempt, candidate_dir)
            probe_path = candidate_dir / "probe.json"
            if probe_path.is_file():
                probe = read(probe_path)
            else:
                probe = {"candidate_id": candidate["id"], "candidate_key": candidate["candidate_key"], "status": "runtime_blocked"}
            if parent_report and probe.get("benchmarks", {}).get(comparison_profile, {}).get("report"):
                try:
                    current = read(_resolve(probe["benchmarks"][comparison_profile]["report"]))
                    probe["comparison"] = compare_reports(parent_report, current)
                except (OSError, ValueError, TypeError, KeyError):
                    pass
            probe["dense_comparison"] = _dense_comparison(snap, probe)
            save_json(probe_path, probe)
            result_path = candidate_dir / "result.json"
            if result_path.is_file():
                result = read(result_path)
            else:
                result = {"status": "failed"}
            result["candidate_id"] = candidate["id"]
            result["candidate_key"] = candidate["candidate_key"]
            result["report"] = str(probe_path.resolve())
            save_json(result_path, result)
            rows.append(probe)
            state["candidates"][candidate["id"]] = probe
            state["active_candidate"] = None
            save_json(state_path, state)
        shutil.rmtree(work, ignore_errors=True)
        from skills.stack_runtime import select_candidate as select_runtime
        from skills.stack_kv_cache import select_candidate as select_kv

        selected = select_runtime(rows) if kind == "runtime" else select_kv(rows)
        state["status"] = "completed" if selected else "blocked"
        state["selected_candidate"] = selected.get("candidate_id") if selected else None
        state["active_candidate"] = None
        save_json(state_path, state)
        summary: dict[str, Any] = {
            "schema": "a4fvlm.stack-acceleration-summary.v1",
            "status": state["status"],
            "kind": kind,
            "candidate_count": len(rows),
            "selected_candidate": selected.get("candidate_id") if selected else None,
            "results": rows,
            "parent_digest": snap["parent"]["digest"],
        }
        if selected:
            summary["freeze"] = _freeze(snap, selected, destination)
        save_json(destination / "summary.json", summary)
        lock_path.unlink(missing_ok=True)
        return summary


def main(kind: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = execute(args.config, kind=kind, output=args.output) if args.execute else plan(args.config, kind=kind)
    print(json.dumps(result, ensure_ascii=False, indent=2))
