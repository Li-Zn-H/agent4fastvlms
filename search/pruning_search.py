"""Progressive SP quality search with frozen allocation and reuse-first workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from typing import Any

from search.model import PROJECT_ROOT
from search.context import dense_root, search_root
from search.optimize import hardware, launch, lock, read, stage_score
from search.structured_pruning import (
    RESULTS,
    complete_measurement,
    existing_measurements,
    shape_key,
    snapshot,
)
from search.worker import save_json
from skills.structured_pruning import (
    allocation_path,
    coarse_candidate_ladder,
    load_calibration,
    make_recipe,
    next_evaluation,
    validate_search,
)


def pruning_root(snap):
    return (search_root(snap["evaluation"], project_root=PROJECT_ROOT) / "structured-pruning"
            if "result_root" in snap["evaluation"] else RESULTS)


def search_run_id(snap) -> str:
    """Stable provenance key for one model/evaluation/pruning protocol."""
    payload = {
        "model": snap.get("model", {}),
        "evaluation": snap.get("evaluation", {}),
        "dimensions": snap.get("dimensions", {}),
        "calibration": snap.get("config", {}).get("calibration"),
        "selection": snap.get("config", {}).get("selection"),
        "allocation": snap.get("config", {}).get("allocation"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return f"sp-{hashlib.sha256(encoded).hexdigest()[:16]}"


def ladder_metadata(snap, index):
    """Return the stable coarse-level metadata for a path candidate."""
    if "path" not in snap or "search" not in snap.get("config", {}):
        return None
    parent = snap.get("extension", {}).get("parent", 0)
    ladder = coarse_candidate_ladder(
        snap["path"], snap["config"]["search"], start_parent=parent
    )
    for row in ladder:
        if row["index"] == index:
            return row
    return None


def build_snapshot(
    config, model_config, evaluation_config, device, probe_output
) -> dict[str, Any]:
    probe = snapshot(config, model_config, evaluation_config, device)
    validate_search(probe["config"])
    records = existing_measurements(PROJECT_ROOT / probe_output, probe["context"])
    measured = {shape_key(row) for row in records if complete_measurement(row)}
    widths = sorted({row["width"] for row in probe["shapes"]}, reverse=True)
    admitted = [
        width
        for width in widths
        if all(
            shape_key(row) in measured
            for row in probe["shapes"]
            if row["width"] == width
        )
    ]
    dimensions = probe["context"]["dimensions"]
    scores = load_calibration(
        PROJECT_ROOT / probe["config"]["calibration"]["source"], dimensions
    )
    result: dict[str, Any] = {
        "model": read(PROJECT_ROOT / model_config),
        "evaluation": read(PROJECT_ROOT / evaluation_config),
        "config": probe["config"],
        "device": device,
        "dimensions": dimensions,
        "probe_source": str(
            (PROJECT_ROOT / probe_output / "measurements.json").resolve()
        ),
        "admitted_widths": admitted,
        "path": allocation_path(
            scores, dimensions, probe["config"]["selection"], admitted
        ),
    }
    result["search_run_id"] = search_run_id(result)
    # Validate both Dense reference stages before allowing any Worker to start.
    baseline = dense_root(result["evaluation"], project_root=PROJECT_ROOT) / "quality"
    for stage in ("search", "dev"):
        stage_score(result, stage, baseline / f"{stage}.jsonl")
    return result


def extend_snapshot(snap, previous):
    """Append a finite allocation suffix without rewriting the old frozen path."""
    previous = Path(previous).resolve()
    old = read(previous / "config.json")
    state = read(previous / "state.json")
    if (
        state.get("stop_reason") != "declared_range_exhausted"
        or state["status"] != "completed"
    ):
        raise ValueError("Extension requires a completed declared range")
    for key in ("model", "evaluation", "device", "dimensions"):
        if snap[key] != old[key]:
            raise ValueError(f"Extension changes {key}")
    for key in ("calibration", "selection", "allocation", "search"):
        if snap["config"][key] != old["config"][key]:
            raise ValueError(f"Extension changes {key}")
    if min(snap["admitted_widths"]) >= min(old["admitted_widths"]):
        raise ValueError("Extension needs newly admitted narrower widths")
    scores = load_calibration(
        PROJECT_ROOT / snap["config"]["calibration"]["source"], snap["dimensions"]
    )
    _, quality = discover(old, pruning_root(old), scores)
    anchor = len(old["path"]) - 1
    if quality.get(anchor, {}).get("stages", {}).get("dev", {}).get("action") != "pass":
        raise ValueError("The old range endpoint needs passing Dev evidence")
    suffix = allocation_path(
        scores,
        snap["dimensions"],
        snap["config"]["selection"],
        snap["admitted_widths"],
        starting_widths=old["path"][-1]["widths"],
    )
    snap["path"] = old["path"] + suffix[1:]
    snap["extension"] = {
        "source": str(previous),
        "parent": anchor,
        "previous_gpu_reserved_seconds": state.get("gpu_reserved_seconds", 0),
    }
    snap["search_run_id"] = old.get("search_run_id", search_run_id(old))
    return snap


def revise_snapshot(previous, config) -> dict[str, Any]:
    """Change only search scheduling; preserve the frozen physical path and costs."""
    previous = Path(previous).resolve()
    old = read(previous / "config.json")
    state = read(previous / "state.json")
    if state["status"] == "running" or not state.get("resource_accounting_complete"):
        raise ValueError("Stop the prior run and finish resource accounting first")
    updated = read(PROJECT_ROOT / config)
    validate_search(updated)
    if {k: v for k, v in updated.items() if k != "search"} != {
        k: v for k, v in old["config"].items() if k != "search"
    }:
        raise ValueError("Policy revision may only change search settings")
    if updated["search"] == old["config"]["search"]:
        raise ValueError("Policy revision requires changed search settings")
    inherited = old.get("revision", {}).get(
        "previous_gpu_reserved_seconds",
        old.get("extension", {}).get("previous_gpu_reserved_seconds", 0),
    )
    return {
        **old,
        "config": updated,
        "search_run_id": old.get("search_run_id", search_run_id(old)),
        "revision": {
            "source": str(previous),
            "reason": "search_policy_revision",
            "previous_search_policy": old["config"]["search"],
            "previous_status": state["status"],
            "previous_stop_reason": state.get("stop_reason"),
            "previous_gpu_reserved_seconds": inherited + state["gpu_reserved_seconds"],
        },
    }


def decision_for(snap, quality):
    parent = snap.get("extension", {}).get("parent", 0)
    if (
        parent
        and quality.get(parent, {}).get("stages", {}).get("dev", {}).get("action")
        != "pass"
    ):
        raise ValueError("Extension parent no longer has passing Dev evidence")
    return next_evaluation(
        snap["path"], quality, snap["config"]["search"], start_parent=parent
    )


def discover(snap, root, scores):
    """Only actual matching channel recipes can share quality evidence."""
    by_width = {tuple(row["widths"]): index for index, row in enumerate(snap["path"])}
    current_run_id = snap.get("search_run_id", search_run_id(snap))
    ladder = {
        row["index"]: row
        for row in coarse_candidate_ladder(
            snap["path"],
            snap["config"].get("search", {}),
            start_parent=snap.get("extension", {}).get("parent", 0),
        )
    }
    sources: dict[int, dict[str, str]] = {}
    forced = set()
    expected = {}
    for request_path in sorted(Path(root).rglob("request.json")):
        try:
            request = read(request_path)
            receipt = request_path.parent / "result.json"
            if receipt.exists() and read(receipt).get("status") != "completed":
                # A failed or interrupted Worker may have left partial JSONL
                # answers.  Never turn those into pending quality evidence or
                # let a stale force-dev request trigger another confirmation.
                continue
            if (
                request.get("operation") != "structured_pruning_candidate"
                or request["model"] != snap["model"]
            ):
                continue
            if (
                request["evaluation"]["generation"] != snap["evaluation"]["generation"]
                or request["evaluation"]["data"] != snap["evaluation"]["data"]
            ):
                continue
            if request.get("search_run_id") not in (None, current_run_id):
                continue
            recipe = read(request["recipe"])
            index = by_width.get(tuple(recipe["widths"]))
            if index is None or not index:
                continue
            requested_level = request.get("ladder_level")
            requested_parent = request.get("parent_candidate")
            if requested_level is None or requested_parent is None:
                decision = request.get("decision", {})
                requested_level = decision.get("ladder_level", requested_level)
                requested_parent = decision.get(
                    "parent_candidate", requested_parent
                )
            if requested_level is not None or requested_parent is not None:
                expected_ladder = ladder.get(index)
                if (
                    expected_ladder is None
                    or requested_level != expected_ladder["ladder_level"]
                    or requested_parent != expected_ladder["parent_candidate"]
                ):
                    continue
            if index not in expected:
                expected[index] = make_recipe(
                    scores,
                    snap["path"][index]["widths"],
                    snap["dimensions"],
                    snap["config"]["selection"],
                )
            if (
                recipe["dimensions"] != expected[index]["dimensions"]
                or recipe["retained_channels"] != expected[index]["retained_channels"]
            ):
                continue
            if request.get("force_dev"):
                forced.add(index)
            candidates = {
                stage: str(request_path.parent / "quality" / f"{stage}.jsonl")
                for stage in ("search", "dev")
            }
            candidates.update(request.get("quality_sources", {}))
            for stage, path in candidates.items():
                if stage in sources.get(index, {}):
                    continue
                try:
                    stage_score(snap, stage, path)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                sources.setdefault(index, {})[stage] = path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    quality = {}
    for index, stages in sources.items():
        if "search" not in stages:
            continue
        scored = {
            stage: stage_score(snap, stage, path) for stage, path in stages.items()
        }
        deciding = scored.get("dev", scored["search"])
        if index in forced and "dev" not in scored:
            quality[index] = {
                "action": "pending",
                "stages": scored,
                "forced_dev": True,
            }
            continue
        if deciding["action"] != "dev":
            quality[index] = {
                "action": deciding["action"],
                "stages": scored,
                "forced_dev": index in forced,
            }
    return sources, quality


def preview(snap, quality):
    decision = decision_for(snap, quality)
    result = {
        "admitted_widths": snap["admitted_widths"],
        "atomic_moves": len(snap["path"]) - 1,
        "maximum_declared_depth": snap["path"][-1]["actual_depth"],
        "evaluation_depth_step": snap["config"]["search"]["initial_depth_step"],
        "failure_recovery_steps": snap["config"]["search"].get(
            "failure_recovery_steps", 1
        ),
        "maximum_quality_candidates": snap["config"]["search"].get(
            "maximum_quality_candidates"
        ),
        "reusable_quality_candidates": len(quality),
        "next": decision,
    }
    if "index" in decision:
        result["next"] = {**decision, **snap["path"][decision["index"]]}
    return result


def execute(snap, destination, max_candidates=None):
    """Reconstruct policy from evidence on resume; never trust stale failure counts."""
    destination = Path(destination)
    if "allocation" in snap.get("config", {}):
        validate_search(snap["config"])
    snap.setdefault("search_run_id", search_run_id(snap))
    scores = load_calibration(
        PROJECT_ROOT / snap["config"]["calibration"]["source"], snap["dimensions"]
    )
    state_path = destination / "state.json"
    state: dict[str, Any] = (
        read(state_path) if state_path.exists() else {"attempts": [], "candidates": {}}
    )
    state["search_run_id"] = snap["search_run_id"]
    # This is persisted across invocations.  Reused historical quality records
    # do not consume the budget; only candidates for which this run launches a
    # Worker are counted.  That keeps a new run reuse-first while preventing a
    # resumed run from slowly turning into an unbounded physical-node scan.
    scheduled = {
        int(index) for index in state.get("scheduled_candidates", [])
    }
    budget = snap["config"]["search"].get("maximum_quality_candidates")
    state["scheduled_candidates"] = sorted(scheduled)
    state.update(status="running")
    state.pop("error", None)
    state.pop("active", None)
    charged = set()
    machine = None
    try:
        while True:
            sources, quality = discover(snap, pruning_root(snap), scores)
            if not destination.is_relative_to(pruning_root(snap)):
                extra_sources, extra_quality = discover(snap, destination, scores)
                for index, stages in extra_sources.items():
                    sources.setdefault(index, {}).update(stages)
                quality.update(extra_quality)
            candidate_state = {}
            for index, value in quality.items():
                metadata = ladder_metadata(snap, index)
                candidate_state[str(index)] = {
                    "actual_depth": snap["path"][index]["actual_depth"],
                    **(
                        {
                            "ladder_level": metadata["ladder_level"],
                            "parent_candidate": metadata["parent_candidate"],
                        }
                        if metadata is not None
                        else {}
                    ),
                    "quality": value,
                    "quality_sources": sources[index],
                }
            state["candidates"] = candidate_state
            decision = decision_for(snap, quality)
            state["next"] = decision
            if "stop_reason" in decision:
                state.update(status="completed", stop_reason=decision["stop_reason"])
                break
            index = decision["index"]
            if (
                budget is not None
                and index not in scheduled
                and len(scheduled) >= budget
            ):
                state.update(
                    status="incomplete",
                    stop_reason="evaluation_budget_exhausted",
                    evaluation_budget=budget,
                )
                break
            if (
                max_candidates is not None
                and index not in charged
                and len(charged) >= max_candidates
            ):
                state.update(status="incomplete", stop_reason="invocation_limit")
                break
            candidate = destination / "candidates" / f"step-{index:04d}"
            candidate.mkdir(parents=True, exist_ok=True)
            recipe_path = candidate / "recipe.json"
            recipe = make_recipe(
                scores,
                snap["path"][index]["widths"],
                snap["dimensions"],
                snap["config"]["selection"],
            )
            if recipe_path.exists():
                if read(recipe_path) != recipe:
                    raise ValueError("Frozen recipe changed; do not overwrite it")
            else:
                save_json(recipe_path, recipe)
            directory = candidate / f"attempt-{len(state['attempts']) + 1}"
            charged.add(index)
            if index not in scheduled:
                scheduled.add(index)
                state["scheduled_candidates"] = sorted(scheduled)
            state["attempts"].append(str(directory))
            state["active"] = str(directory)
            save_json(state_path, state)
            device = snap["device"]
            with lock(f"/tmp/agent4vlm-gpu-{os.getuid()}-{device}.lock"):
                if machine is None:
                    machine = hardware(device, snap["model"])
                print(
                    f"SP step={index} depth={recipe['actual_depth']:.6f} reason={decision['reason']}",
                    flush=True,
                )
                launch(
                    {
                        "operation": "structured_pruning_candidate",
                        "device": device,
                        "hardware": machine,
                        "model": snap["model"],
                        "evaluation": snap["evaluation"],
                        "recipe": str(recipe_path.resolve()),
                        "search_run_id": snap["search_run_id"],
                        **(
                            {
                                "ladder_level": decision["ladder_level"],
                                "parent_candidate": decision[
                                    "parent_candidate"
                                ],
                            }
                            if "ladder_level" in decision
                            else {}
                        ),
                        "quality_sources": sources.get(index, {}),
                        "force_dev": decision["force_dev"],
                        "decision": decision,
                    },
                    directory,
                )
            state.pop("active", None)
            # Stop on malformed output instead of submitting the same missing job forever.
            _, fresh = discover(snap, destination, scores)
            if index not in fresh or (
                decision["force_dev"] and "dev" not in fresh[index]["stages"]
            ):
                raise RuntimeError(
                    "Worker did not produce the requested complete quality evidence"
                )
    except BaseException as error:
        state.update(
            status="incomplete",
            stop_reason="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "execution_error",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        receipts = [
            read(Path(path) / "result.json")
            for path in state["attempts"]
            if (Path(path) / "result.json").exists()
        ]
        state["worker_seconds"] = sum(row.get("worker_seconds", 0) for row in receipts)
        state["gpu_reserved_seconds"] = sum(
            row.get("gpu_reserved_seconds", 0) for row in receipts
        )
        state["resource_accounting_complete"] = len(receipts) == len(state["attempts"])
        save_json(state_path, state)
        summary = {
            key: state[key]
            for key in (
                "status",
                "stop_reason",
                "worker_seconds",
                "gpu_reserved_seconds",
                "resource_accounting_complete",
            )
        }
        inherited = snap.get("revision", {}).get(
            "previous_gpu_reserved_seconds",
            snap.get("extension", {}).get("previous_gpu_reserved_seconds", 0),
        )
        state["inherited_search_gpu_reserved_seconds"] = inherited
        state["cumulative_search_gpu_reserved_seconds"] = (
            inherited + state["gpu_reserved_seconds"]
        )
        save_json(state_path, state)
        summary.update(
            inherited_search_gpu_reserved_seconds=inherited,
            cumulative_search_gpu_reserved_seconds=state[
                "cumulative_search_gpu_reserved_seconds"
            ],
            checkpoint="state.json",
            evaluated_candidates=len(state["candidates"]),
            passing_candidates=sum(
                row["quality"]["action"] == "pass"
                for row in state["candidates"].values()
            ),
            scheduled_candidate_count=len(scheduled),
            maximum_quality_candidates=budget,
            next=state.get("next"),
        )
        save_json(destination / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/structured_pruning.json")
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation.json")
    parser.add_argument("--probe-output")
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extend-from", help="Completed prior runner; append a narrower range"
    )
    parser.add_argument(
        "--revise-from", help="Stopped prior runner; revise search policy only"
    )
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if sum(bool(x) for x in (args.resume, args.extend_from, args.revise_from)) > 1:
        parser.error("--resume, --extend-from and --revise-from are mutually exclusive")
    evaluation = read(PROJECT_ROOT / args.evaluation_config)
    base = search_root(evaluation) / "structured-pruning"
    args.probe_output = args.probe_output or str(base / "hardware-probe")
    destination = (PROJECT_ROOT / args.output).resolve() if args.output else base / "runner"
    snap = (
        read(destination / "config.json")
        if args.resume
        else revise_snapshot(PROJECT_ROOT / args.revise_from, args.config)
        if args.revise_from
        else build_snapshot(
            args.config,
            args.model_config,
            args.evaluation_config,
            args.device,
            args.probe_output,
        )
    )
    if args.extend_from:
        if args.resume:
            parser.error(
                "--extend-from creates a new run; use --resume on that run later"
            )
        snap = extend_snapshot(snap, PROJECT_ROOT / args.extend_from)
    validate_search(snap["config"])
    snap.setdefault("search_run_id", search_run_id(snap))
    if not args.execute:
        scores = load_calibration(
            PROJECT_ROOT / snap["config"]["calibration"]["source"], snap["dimensions"]
        )
        _, quality = discover(snap, pruning_root(snap), scores)
        print(json.dumps(preview(snap, quality), indent=2))
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Coordinator received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    if not args.resume:
        destination.mkdir(parents=True, exist_ok=False)
    with lock(destination / ".run.lock"):
        if not args.resume:
            save_json(destination / "config.json", snap)
        print(json.dumps(execute(snap, destination, args.max_candidates), indent=2))


if __name__ == "__main__":
    main()
