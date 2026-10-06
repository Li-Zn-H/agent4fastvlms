"""Screen frozen shallow/middle/deep SP representatives and confirm improvements."""

import argparse
import os
from pathlib import Path
import signal
from typing import Any

from search.model import PROJECT_ROOT
from search.optimize import hardware, inspect_benchmark, launch, lock, read
from search.pruning_search import discover, ladder_metadata, search_run_id
from search.structured_pruning import RESULTS
from search.context import search_root
from search.worker import save_json
from skills.structured_pruning import load_calibration


def representatives(quality):
    """Select first, middle (upper median), and last passing depth before timing."""
    passing = sorted(i for i, row in quality.items() if row["action"] == "pass")
    return (
        sorted({passing[0], passing[len(passing) // 2], passing[-1]}) if passing else []
    )


def recipe_source(runner, index):
    """Find inherited recipes without copying historical artifacts."""
    runner = Path(runner)
    while True:
        recipe = runner / "candidates" / f"step-{index:04d}" / "recipe.json"
        if recipe.exists():
            return recipe.resolve()
        config = read(runner / "config.json")
        extension = config.get("revision") or config.get("extension")
        if not extension:
            raise FileNotFoundError(recipe)
        runner = Path(extension["source"])


def comparison(control, candidate):
    """Compare E2E and allocated memory only; output lengths may differ."""
    latency = control["mean_round_seconds"] / candidate["mean_round_seconds"]
    dense_memory = max(r["peak_allocated_bytes"] for r in control["rounds"])
    memory = max(r["peak_allocated_bytes"] for r in candidate["rounds"])
    return {
        "latency_speedup": latency,
        "peak_allocated_bytes": memory,
        "memory_reduction_fraction": 1 - memory / dense_memory,
        "improved": latency > 1 or memory < dense_memory,
    }


def screen(runner, execute=False):
    runner = Path(runner)
    snap = read(runner / "config.json")
    root = (search_root(snap["evaluation"], project_root=PROJECT_ROOT) / "structured-pruning"
            if "result_root" in snap["evaluation"] else RESULTS)
    search_state = read(runner / "state.json")
    if search_state["status"] != "completed":
        raise ValueError("Finish the SP quality search before performance screening")
    scores = load_calibration(
        PROJECT_ROOT / snap["config"]["calibration"]["source"], snap["dimensions"]
    )
    sources, quality = discover(snap, root, scores)
    destination = runner / "screen"
    checkpoint = destination / "summary.json"
    state: dict[str, Any] = (
        read(checkpoint)
        if checkpoint.exists()
        else {
            "selection": "first_middle_last_passing_depth",
            "selected": representatives(quality),
            "attempts": [],
            "candidates": {},
        }
    )
    dense = root.parent / snap["evaluation"]["scoring"]["baseline"]
    control = inspect_benchmark(snap, None, [dense / "request.json"])
    if control is None:
        raise ValueError(
            "A complete Dense benchmark is required; do not rerun silently"
        )
    if not execute:
        return {
            "selected": state["selected"],
            "control": control["source"],
            "saved_status": state.get("status", "not_started"),
        }
    destination.mkdir(parents=True, exist_ok=True)
    state.update(status="running", control=control)
    state.pop("error", None)
    machine = None

    def run(index, purpose):
        nonlocal machine
        directory = (
            destination / f"step-{index:04d}" / f"attempt-{len(state['attempts']) + 1}"
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        state["attempts"].append(str(directory))
        state["active"] = str(directory)
        save_json(checkpoint, state)
        device = snap["device"]
        with lock(f"/tmp/agent4vlm-gpu-{os.getuid()}-{device}.lock"):
            if machine is None:
                machine = hardware(device, snap["model"])
            print(f"SP screen step={index} phase={purpose}", flush=True)
            metadata = ladder_metadata(snap, index) or {}
            launch(
                {
                    "operation": "structured_pruning_candidate",
                    "device": device,
                    "hardware": machine,
                    "model": snap["model"],
                    "evaluation": snap["evaluation"],
                    "recipe": str(recipe_source(runner, index)),
                    "search_run_id": snap.get("search_run_id", search_run_id(snap)),
                    **(
                        {
                            "ladder_level": metadata["ladder_level"],
                            "parent_candidate": metadata["parent_candidate"],
                        }
                        if metadata
                        else {}
                    ),
                    "quality_sources": sources[index],
                    "evaluate": purpose == "dev",
                    "force_dev": purpose == "dev",
                    "measure": purpose == "benchmark",
                },
                directory,
            )
        state.pop("active", None)

    try:
        for index in state["selected"]:
            recipe_path = recipe_source(runner, index)
            recipe = read(recipe_path)
            requests = []
            for path in sorted(root.rglob("request.json")):
                request = read(path)
                if (
                    request.get("operation") == "structured_pruning_candidate"
                    and request.get("model") == snap["model"]
                    and request["evaluation"] == snap["evaluation"]
                    and read(request["recipe"]) == recipe
                ):
                    requests.append(path)
            performance = inspect_benchmark(snap, None, sorted(requests))
            if performance is None:
                run(index, "benchmark")
                performance = inspect_benchmark(
                    snap, None, [Path(state["attempts"][-1]) / "request.json"]
                )
            if performance is None:
                raise RuntimeError("Worker produced incomplete benchmark evidence")
            row = {
                "actual_depth": recipe["actual_depth"],
                "benchmark": performance,
                **comparison(control["summary"], performance["summary"]),
            }
            state["candidates"][str(index)] = row
            save_json(checkpoint, state)
            if row["improved"] and "dev" not in quality[index]["stages"]:
                run(index, "dev")
                sources, quality = discover(snap, root, scores)
                if "dev" not in quality.get(index, {}).get("stages", {}):
                    raise RuntimeError("Worker produced incomplete Dev evidence")
            row["quality"] = quality[index]
            row["quality_sources"] = sources[index]
            row["confirmed_improvement"] = (
                row["improved"]
                and quality[index]["stages"].get("dev", {}).get("action") == "pass"
            )
            save_json(checkpoint, state)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipts = [
            read(Path(p) / "result.json")
            for p in state["attempts"]
            if (Path(p) / "result.json").exists()
        ]
        state["worker_seconds"] = sum(r.get("worker_seconds", 0) for r in receipts)
        state["gpu_reserved_seconds"] = sum(
            r.get("gpu_reserved_seconds", 0) for r in receipts
        )
        state["resource_accounting_complete"] = len(receipts) == len(state["attempts"])
        save_json(checkpoint, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default=str(RESULTS / "runner"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runner = (PROJECT_ROOT / args.runner).resolve()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Coordinator received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if args.execute:
        with lock(runner / ".run.lock"):
            state = screen(runner, execute=True)
        print({"status": state["status"], "selected": state["selected"]})
    else:
        print(screen(runner))


if __name__ == "__main__":
    main()
