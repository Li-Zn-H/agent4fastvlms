"""Run Runtime then KV Cache on a Stack that passed the full LMMS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search.stack_acceleration import _resolve, execute, snapshot
from search.optimize import lock, read
from search.worker import save_json
from skills.quantization import digest
from search.lmms_gate import stack_handoff_eligible


def execute_pipeline(runtime_config="configs/stack_runtime.json", kv_config="configs/stack_kv_cache.json"):
    snap = snapshot(runtime_config, kind="runtime")
    parent = snap["parent"]
    if not stack_handoff_eligible(parent["freeze"]):
        raise ValueError("Serving pipeline requires an eligible complete Stack LMMS handoff")
    output = _resolve(snap["config"]["output"])
    runtime_freeze = output / "final/runtime-freeze.json"
    kv_settings = read(_resolve(kv_config))
    if _resolve(kv_settings["parent_freeze"]).resolve() != runtime_freeze.resolve():
        raise ValueError("KV config must consume this Runtime job's freeze")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "serving-state.json"
    lock_path = output / ".serving.lock"
    with lock(lock_path):
        state = {
            "status": "running", "stage": "runtime",
            "stack_candidate": parent["freeze"]["candidate_id"],
            "stack_digest": parent["digest"],
            "runtime_config": str(_resolve(runtime_config)),
            "kv_config": str(_resolve(kv_config)),
        }
        save_json(state_path, state)
        try:
            runtime = execute(runtime_config, kind="runtime")
            state["runtime_selected"] = runtime.get("selected_candidate")
            if not state["runtime_selected"]:
                state.update(status="blocked", stop_reason="runtime_no_qualified_candidate")
                return state
            if digest(read(Path(parent["path"]))) != parent["digest"]:
                raise ValueError("Stack parent changed during Runtime search")
            frozen = read(runtime_freeze)
            if frozen.get("parent_digest") != parent["digest"]:
                raise ValueError("Runtime freeze does not match the requested Stack")
            state["stage"] = "kv_cache"
            save_json(state_path, state)
            kv = execute(kv_config, kind="kv_cache")
            state["kv_selected"] = kv.get("selected_candidate")
            state.update(
                status="completed" if state["kv_selected"] else "blocked",
                stage=None,
                stop_reason="ready_for_lmms" if state["kv_selected"] else "kv_no_qualified_candidate",
            )
            return state
        except Exception as error:
            state.update(status="blocked", stop_reason="execution_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(state_path, state)
            lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="configs/stack_runtime.json")
    parser.add_argument("--kv-config", default="configs/stack_kv_cache.json")
    parser.add_argument("--lmms-config", default="configs/serving_lmms.json",
                        help="Final vLLM LMMS job, bound to the same serving state")
    args = parser.parse_args()
    # Validate the continuation before any expensive workers are started.
    lmms_settings = read(_resolve(args.lmms_config))
    runtime_settings = read(_resolve(args.runtime_config))
    from search.serving_lmms import validate_context_capacity
    for candidate in runtime_settings["candidates"]:
        validate_context_capacity(
            runtime_settings.get("runtime", {}) | candidate.get("runtime", {}),
            lmms_settings.get("minimum_model_len"),
        )
    if _resolve(lmms_settings["serving_state"]).resolve() != (_resolve(runtime_settings["output"]) / "serving-state.json").resolve():
        raise ValueError("Final LMMS job does not belong to this serving pipeline")
    result = execute_pipeline(args.runtime_config, args.kv_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("stop_reason") == "ready_for_lmms":
        from search.serving_lmms import execute as execute_lmms
        print(json.dumps(execute_lmms(args.lmms_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
