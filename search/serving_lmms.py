"""Resumable final vLLM LMMS gate, after Runtime and KV Cache selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from search.lmms import _task_complete
from search.lmms_gate import (DEFAULT_TASKS, aggregate_lmms, complete_lmms,
                              selection_policy, stack_handoff_eligible,
                              quality_speed_frontier, select_measured)
from search.model import PROJECT_ROOT
from search.optimize import lock, read
from search.worker import save_json
from skills.quantization import digest


def resolve(value):
    return (PROJECT_ROOT / value).absolute()


def qualified(probe):
    """Eligible for full LMMS, with no Search/Dev quality requirement."""
    from skills.stack_runtime import performance_ready

    return performance_ready(probe)


def validate_context_capacity(runtime, minimum):
    """Reject a serving profile that cannot cover the configured LMMS inputs."""
    if minimum is None:
        return
    capacity = runtime.get("vllm_max_model_len")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum_model_len must be a positive integer")
    if type(capacity) is not int or capacity < minimum:
        raise ValueError(f"LMMS requires max_model_len >= {minimum}; serving profile has {capacity}")


def batch1_latency(probe):
    profiles = probe.get("benchmarks", {})
    timing = profiles.get("batch1", profiles.get("search", {})).get("summary", {})
    elapsed, count = timing.get("elapsed_seconds"), timing.get("request_count")
    if not elapsed or not count:
        raise ValueError("Missing batch-1 timing for final candidate comparison")
    return elapsed / count


def plan(config):
    """Bind every layer to the same artifact, parent and Token mechanism."""
    policy = selection_policy(config)
    chain = read(resolve(config["serving_state"]))
    if chain.get("stop_reason") not in {"serving_qualified", "ready_for_lmms"} or chain.get("status") != "completed":
        raise ValueError("Runtime/KV chain has no completed performance handoff")
    kv_path = resolve(config["kv_freeze"])
    kv = read(kv_path)
    validate_context_capacity(kv["runtime"], config.get("minimum_model_len"))
    runtime_path = Path(kv["parent_freeze"])
    runtime = read(runtime_path)
    stack_path = Path(runtime["parent_freeze"])
    stack = read(stack_path)
    if (kv["parent_digest"] != digest(runtime) or runtime["parent_digest"] != digest(stack)
            or chain["stack_digest"] != digest(stack)
            or chain["runtime_selected"] != runtime["candidate_id"]
            or chain["kv_selected"] != kv["candidate_id"]):
        raise ValueError("Serving parent chain/digest mismatch")
    if (not stack_handoff_eligible(stack)
            or (stack.get("status") == "frozen_best_available" and policy != "best_effort")
            or not qualified(kv) or not qualified(runtime)):
        raise ValueError("Serving parent lacks Stack LMMS or Runtime/KV performance evidence")
    descriptor = read(Path(stack["lmms"]["model_config"]))
    for layer in (kv, runtime):
        if (Path(layer["artifact"]).resolve() != Path(stack["artifact"]).resolve()
                or layer["token_keep_ratio"] != stack["token_keep_ratio"]):
            raise ValueError("Artifact/Token drift across serving layers")
    if descriptor["token_keep_ratio"] != stack["token_keep_ratio"]:
        raise ValueError("Native LMMS Token descriptor mismatch")
    descriptor.update(path=stack["artifact"], execution_backend="vllm", stack_adapter=True,
                      stack_digest=digest(stack), dtype=descriptor.get("dtype", "bfloat16"), device="cuda:0")
    candidates, seen = [], set()

    def add(probe, source, runtime_id, kv_id):
        if not qualified(probe):
            return
        if policy == "best_effort":
            batch1_latency(probe)
        settings = dict(probe["runtime"])
        settings.setdefault("kv_cache_dtype", "auto")
        validate_context_capacity(settings, config.get("minimum_model_len"))
        key = digest({"stack": digest(stack), "runtime": settings,
                      "token_selection": descriptor["token_selection"], "bridge": 1})
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "id": "serving-" + key, "runtime_id": runtime_id, "kv_id": kv_id,
            "source": str(source), "source_digest": digest(probe),
            "model": descriptor | {"runtime": settings},
            "local_quality": probe.get("quality", {"status": "pending_full_lmms"}), "dense_comparison": probe.get("dense_comparison"),
        })

    add(kv, kv_path, runtime["candidate_id"], kv["candidate_id"])
    # First remove KV precision loss, then try the explicitly configured,
    # already measured runtime controls. Never disable Token/SP/quant.
    kv_root = kv_path.parent.parent
    auto_path = kv_root / "candidates/kv-auto/probe.json"
    if auto_path.is_file():
        auto = read(auto_path)
        if auto.get("parent_digest") != digest(runtime):
            raise ValueError("KV fallback parent mismatch")
        add(auto, auto_path, runtime["candidate_id"], "kv-auto")
    runtime_root = runtime_path.parent.parent
    for candidate in config.get("fallback_runtime_candidates", []):
        path = runtime_root / "candidates" / candidate / "probe.json"
        probe = read(path)
        if probe.get("parent_digest") != digest(stack):
            raise ValueError("Runtime fallback parent mismatch")
        # Runtime probes used the default auto KV; no FP8 receipt is inherited.
        add(probe, path, candidate, "auto_from_runtime_dev" if probe.get("status") == "qualified" else "auto_from_runtime")
    return {"stack_digest": digest(stack), "runtime_digest": digest(runtime),
            "kv_digest": digest(kv), "candidates": candidates}


def _run_tasks(config, model_path, output, *, limit=None):
    command = [str(resolve(config["worker_python"])), "-B", "-m", "search.lmms",
               "--model-config", str(model_path), "--output", str(output),
               "--tasks", ",".join(row["task"] for row in config["tasks"])]
    if limit is not None:
        command += ["--limit", str(limit)]
    if (output / "config.json").is_file():
        command.append("--resume")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config["device"])
    with (output.parent / (output.name + ".log")).open("a") as log:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def execute(config_path="configs/serving_lmms.json", *, wait=False):
    config = read(resolve(config_path))
    policy = selection_policy(config)
    if config["tasks"] != list(DEFAULT_TASKS) or config["minimum_retention"] < .95:
        raise ValueError("Final serving gate requires the five native tasks and retention >= 95%")
    if config["smoke_samples_per_task"] < 1:
        raise ValueError("Smoke must have at least one sample per task")
    root = resolve(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    config_path_saved = root / "config.json"
    measured_config = lambda value: {k: v for k, v in value.items() if k != "selection_policy"}
    if config_path_saved.exists() and measured_config(read(config_path_saved)) != measured_config(config):
        raise ValueError("Serving LMMS resume config mismatch")
    state = {"status": "running", "stage": "waiting_for_serving", "candidates": {}}
    lock_path = root / ".lmms.lock"
    with lock(lock_path):
        save_json(config_path_saved, config)
        save_json(root / "state.json", state)
        try:
            while True:
                chain = read(resolve(config["serving_state"]))
                if chain.get("status") == "completed":
                    break
                if not wait or chain.get("status") != "running":
                    raise ValueError("Serving search is not ready; no LMMS Worker launched")
                launch = read(resolve(config["serving_state"]).parent / "launch.json")
                os.kill(int(launch["pid"]), 0)
                time.sleep(15)
            snapshot = plan(config)
            saved = root / "plan.json"
            if saved.exists() and read(saved) != snapshot:
                raise ValueError("Serving evidence changed; do not reuse the previous LMMS run")
            save_json(saved, snapshot)
            evaluated = []
            for candidate in snapshot["candidates"]:
                directory = root / "candidates" / candidate["id"]
                directory.mkdir(parents=True, exist_ok=True)
                descriptor_path = directory / "model.json"
                save_json(descriptor_path, candidate["model"])
                state.update(stage="smoke", active_candidate=candidate["id"])
                save_json(root / "state.json", state)
                smoke = directory / "smoke"
                _run_tasks(config, descriptor_path, smoke, limit=config["smoke_samples_per_task"])
                for row in config["tasks"]:
                    if not _task_complete(smoke, row["task"], candidate["model"]):
                        raise ValueError("Incomplete serving smoke")
                save_json(smoke / "summary.json", {
                    "status": "mechanism_verified", "quality_gate": "not_evaluated",
                    "tasks": [row["task"] for row in config["tasks"]],
                    "samples_per_task": config["smoke_samples_per_task"],
                    "model_config": str(descriptor_path),
                })
                state.update(stage="lmms")
                save_json(root / "state.json", state)
                output = directory / "lmms"
                _run_tasks(config, descriptor_path, output)
                summary = aggregate_lmms(
                    output, resolve(config["baseline_root"]), tasks=config["tasks"],
                    minimum_retention=config["minimum_retention"],
                    token_keep_ratio=candidate["model"]["token_keep_ratio"],
                    candidate_id=candidate["id"], model_config=str(descriptor_path),
                    execution_backend="vllm",
                )
                if policy == "best_effort" and not complete_lmms(summary):
                    raise ValueError("Incomplete final LMMS decision; no best-effort promotion")
                # Catch changed parents before publishing any final promotion.
                if plan(config) != snapshot:
                    raise ValueError("Serving parent changed during LMMS")
                save_json(output / "summary.json", summary)
                state["candidates"][candidate["id"]] = summary
                evaluated.append(candidate | {"lmms": summary})
                save_json(root / "state.json", state)
                if summary["quality_gate"] == "passed":
                    if policy == "strict":
                        (root / "final").mkdir(exist_ok=True)
                        save_json(root / "final/serving-freeze.json", {
                            "schema": "a4fvlm.serving-freeze.v1", "status": "lmms_qualified",
                            **candidate, "lmms": summary, "stack_digest": snapshot["stack_digest"],
                        })
                    state.update(status="completed", stage=None, active_candidate=None,
                                 stop_reason="lmms_quality_passed", selected_candidate=candidate["id"],
                                 quality_target_met=True)
                    break
            else:
                state.update(status="completed", stage=None, active_candidate=None,
                             stop_reason="serving_quality_unmet", selected_candidate=None,
                             quality_target_met=False)
            if policy == "best_effort":
                # Every source already passed the Runtime/KV performance contract.
                # Keep these timings explicitly separate from end-to-end speed.
                measured = []
                (root / "final").mkdir(exist_ok=True)
                for candidate in evaluated:
                    probe = read(Path(candidate["source"]))
                    latency = batch1_latency(probe)
                    freeze = {"schema": "a4fvlm.serving-freeze.v1", **candidate,
                              "status": "lmms_qualified" if candidate["lmms"]["quality_gate"] == "passed" else "lmms_evaluated",
                              "selection_policy": policy,
                              "quality_target_met": candidate["lmms"]["quality_gate"] == "passed",
                              "stack_digest": snapshot["stack_digest"]}
                    path = root / "final" / (candidate["id"] + ".json")
                    save_json(path, freeze)
                    measured.append({"candidate_id": candidate["id"], "retention": candidate["lmms"]["retention"],
                                     "mean_seconds": latency, "timing_scope": "vllm_engine_batch1",
                                     "freeze": str(path)})
                frontier = quality_speed_frontier(measured)
                selected = select_measured(frontier, config["minimum_retention"])
                state["tradeoffs"] = frontier
                state["selection_policy"] = policy
                if selected:
                    freeze = read(Path(selected["freeze"]))
                    save_json(root / "final/serving-freeze.json", freeze)
                    state.update(selected_candidate=selected["candidate_id"],
                                 quality_target_met=freeze["quality_target_met"],
                                 stop_reason="lmms_quality_passed" if freeze["quality_target_met"] else "best_available_selected")
            save_json(root / "summary.json", state)
            return state
        except Exception as error:
            state.update(status="blocked", stop_reason=("blocked_preflight" if state["stage"] in
                         {"waiting_for_serving", "smoke"} else "execution_error"),
                         selected_candidate=None,
                         error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(root / "state.json", state)
            lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving_lmms.json")
    parser.add_argument("--wait", action="store_true", help="Wait for the live Runtime/KV coordinator")
    args = parser.parse_args()
    print(json.dumps(execute(args.config, wait=args.wait), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
