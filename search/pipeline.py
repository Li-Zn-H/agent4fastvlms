"""Read-only planning or resumable dispatch of the existing optimization stages.

No model/framework imports, candidate selection, scoring policy or GPU query
belongs in this coordinator. Stage owners retain their existing search loops.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from search.model import PROJECT_ROOT
from search.context import dense_root, search_root, hardware_adapter, model_adapter, require_stages, validate_result_scope
from search.optimize import LOCK_FDS, lock, read
from search.worker import save_json
from skills.quantization import digest
from search.lmms_gate import complete_lmms, selection_policy, stack_handoff_eligible


STAGES = ("dense", "dense_lmms", "token", "sp_probe", "sp", "quantization", "stack", "serving")
CONFIG_KEYS = ("optimization", "structured_pruning", "quantization", "stack", "runtime", "kv", "serving_lmms")


def resolve(value):
    return (PROJECT_ROOT / value).resolve()


def result(status, reason, **facts):
    return {"status": status, "reason": reason, **facts}


class Pipeline:
    def __init__(self, config="configs/pipeline.json"):
        self.path = resolve(config)
        self.job = read(self.path)
        self.configs = {key: read(resolve(self.job[key + "_config"])) for key in CONFIG_KEYS}
        self.opt = self.configs["optimization"]
        self.stack = self.configs["stack"]
        self.serving = self.configs["serving_lmms"]
        self.model = read(resolve(self.opt["model_config"]))
        self.evaluation = read(resolve(self.opt["evaluation_config"]))
        self.vision = read(resolve(self.opt["vision_token_config"]))
        self.root = resolve(self.job["output"])
        self.device = self.opt["device"]
        self.dense = dense_root(self.evaluation, project_root=PROJECT_ROOT)
        self.outputs = {
            "dense": self.dense,
            "dense_lmms": resolve(self.serving["baseline_root"]),
            "token": resolve(self.job["token_output"]),
            "sp_probe": resolve(self.job["sp_probe_output"]),
            "sp": resolve(self.stack["structured_pruning_root"]),
            **{key: resolve(self.configs[key]["output"]) for key in ("quantization", "stack")},
            "serving": resolve(self.serving["output"]),
        }
        self.validate()

    def validate(self):
        from adapters import select_model
        from search.lmms_gate import DEFAULT_TASKS
        from search.serving_lmms import validate_context_capacity

        checkpoint = read(resolve(self.model["path"]) / "config.json")
        model_adapter(self.model, project_root=PROJECT_ROOT)
        select_model(checkpoint, pipeline_id=self.job["model_id"])
        self.platform = hardware_adapter(self.model)
        self.capabilities = require_stages(self.model, None, device=self.device, project_root=PROJECT_ROOT)
        for key in ("quantization", "stack", "runtime", "kv", "serving_lmms"):
            cfg = self.configs[key]
            if cfg["device"] != self.device:
                raise ValueError(f"{key} device differs from optimization job")
            if cfg.get("hardware", self.platform.id) != self.platform.id:
                raise ValueError(f"{key} hardware platform differs from the model descriptor")
            for setting in ("model_config", "evaluation_config", "vision_token_config"):
                if setting in cfg and read(resolve(cfg[setting])) != read(resolve(self.opt[setting])):
                    raise ValueError(f"{key} {setting} mismatch")
        # Stack's legacy default must be bound too, even without an explicit key.
        if read(resolve(self.stack.get("vision_token_config", "configs/vision_token.json"))) != self.vision:
            raise ValueError("Stack Token selection/config mismatch")
        pairs = [
            (self.stack["quantization_root"], self.outputs["quantization"]),
            (self.configs["runtime"]["parent_freeze"], self.outputs["stack"] / "final/stack-freeze.json"),
            (self.configs["kv"]["parent_freeze"], resolve(self.configs["runtime"]["output"]) / "final/runtime-freeze.json"),
            (self.serving["kv_freeze"], resolve(self.configs["kv"]["output"]) / "final/kv-cache-freeze.json"),
            (self.serving["serving_state"], resolve(self.configs["runtime"]["output"]) / "serving-state.json"),
            (self.stack["lmms"]["baseline_root"], self.outputs["dense_lmms"]),
        ]
        for left, right in pairs:
            if resolve(left) != resolve(right):
                raise ValueError(f"Pipeline handoff path mismatch: {left} != {right}")
        if not self.outputs["token"].is_relative_to(resolve(self.stack["vision_token_root"])):
            raise ValueError("Token output must be discoverable under Stack vision_token_root")
        model_root = validate_result_scope(self.model, self.evaluation, project_root=PROJECT_ROOT)
        for path in [*self.outputs.values(), self.root,
                     resolve(self.configs["runtime"]["output"]), resolve(self.configs["kv"]["output"])]:
            if not path.is_relative_to(model_root):
                raise ValueError("Stage outputs must be under evaluation.result_root")
        if any(self.root == path or path.is_relative_to(self.root) or self.root.is_relative_to(path)
               for path in self.outputs.values()):
            raise ValueError("Pipeline output must be separate from stage outputs")
        for cfg in (self.stack["lmms"], self.serving):
            if cfg.get("tasks") != list(DEFAULT_TASKS) or cfg.get("minimum_retention", 0) < .95:
                raise ValueError("Final gates require the five tasks and retention >= 95%")
        if not self.stack["lmms"].get("enabled"):
            raise ValueError("Stack LMMS cannot be disabled in the complete pipeline")
        for candidate in self.configs["runtime"]["candidates"]:
            validate_context_capacity(self.configs["runtime"].get("runtime", {}) | candidate["runtime"],
                                      self.serving.get("minimum_model_len"))
        for key in ("runtime", "kv"):
            if resolve(self.configs[key]["dense_benchmark"]) != self.dense / "benchmark/summary.json":
                raise ValueError(f"{key} must use this model's Dense performance denominator")

    def snapshot(self):
        """Bind referenced JSONs/data, rather than duplicate their settings."""
        files = {self.path, *(resolve(self.job[key + "_config"]) for key in CONFIG_KEYS)}
        files.update(resolve(self.opt[key]) for key in ("model_config", "evaluation_config", "vision_token_config"))
        for cfg in self.configs.values():
            files.update(resolve(value) for key, value in cfg.items()
                         if key in {"model_config", "evaluation_config", "vision_token_config"})
        files.add(resolve(self.model["path"]) / "config.json")
        files.update(resolve(self.evaluation["data"][key]) for key in ("search", "dev"))
        for cfg in (self.configs["quantization"], self.stack.get("quantization", {})):
            if cfg.get("calibration", {}).get("data"):
                files.add(resolve(cfg["calibration"]["data"]))
        return {"schema": "a4fvlm.pipeline-config.v1", "job": self.job,
                "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}}

    def _saved_matches(self, root, expected, *, stack_policy=False):
        path = root / "config.json"
        if path.exists():
            saved = read(path)
            for key, value in expected.items():
                actual = saved.get(key)
                if stack_policy and key == "config" and isinstance(actual, dict):
                    from skills.stack import evidence_config
                    actual, value = evidence_config(actual), evidence_config(value)
                if actual != value:
                    raise ValueError(f"Saved config mismatch: {path}: {key}; use a matching/new stage job")

    def _lmms(self, root, freeze, *, serving=False):
        from search.lmms_gate import aggregate_lmms

        cfg = self.serving if serving else self.stack["lmms"]
        receipt = freeze["lmms"]
        descriptor = read(resolve(receipt["model_config"]))
        ratio = freeze["model"]["token_keep_ratio"] if serving else freeze["token_keep_ratio"]
        if descriptor.get("token_keep_ratio") != ratio:
            raise ValueError("LMMS Token descriptor mismatch")
        summary = aggregate_lmms(root, self.outputs["dense_lmms"], tasks=cfg["tasks"],
                                 minimum_retention=cfg["minimum_retention"], token_keep_ratio=ratio,
                                 candidate_id=receipt["candidate_id"], model_config=receipt["model_config"],
                                 execution_backend="vllm" if serving else "native_transformers")
        if summary["quality_gate"] != "passed" and not (
                selection_policy(cfg if serving else self.stack) == "best_effort"
                and complete_lmms(summary) and summary == receipt):
            raise ValueError("Frozen candidate no longer has a passed quality gate")
        return summary

    def inspect(self, stage):
        """Return reusable evidence or the next action; never execute a child."""
        try:
            return self._inspect(stage)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return result("blocked", "blocked_preflight", error=f"{type(error).__name__}: {error}")

    def _inspect(self, stage):
        root = self.outputs[stage]
        if stage == "dense":
            from search.baseline import complete_outputs
            from search.evaluate import load_samples
            from search.optimize import inspect_benchmark
            self._saved_matches(root, {"model": self.model, "evaluation": self.evaluation})
            missing = []
            for split in ("search", "dev"):
                samples = load_samples(resolve(self.evaluation["data"][split]),
                                       media_root=resolve(self.evaluation["data"]["media_root"]))
                if complete_outputs(root / "quality" / (split + ".jsonl"), samples) is None:
                    missing.append(split)
            benchmark = inspect_benchmark(self.token_snapshot(), None, [root / "request.json"])
            return result("complete" if not missing and benchmark else "pending", "dense_reference",
                          missing_quality=missing, missing_benchmark=benchmark is None, evidence=str(root))
        if stage == "dense_lmms":
            from search.lmms import _task_complete
            from search.lmms_gate import _task_result
            self._saved_matches(root, {"model": self.model, "limit": None,
                                       "tasks": [row["task"] for row in self.serving["tasks"]]})
            ready = all(_task_complete(root, row["task"], self.model) for row in self.serving["tasks"])
            if ready:
                for row in self.serving["tasks"]:
                    _, task = _task_result(root, row["task"])
                    counts = task["n-samples"][row["task"]]
                    if task.get("config", {}).get("limit") is not None or counts.get("original") != counts["effective"]:
                        raise ValueError("Dense LMMS denominator is a partial/smoke evaluation")
                    args = dict(item.split("=", 1) for item in task.get("config", {}).get("model_args", "").split(",") if "=" in item)
                    if not args.get("pretrained") or resolve(args["pretrained"]) != resolve(self.model["path"]):
                        raise ValueError("Dense LMMS belongs to another model")
            elif root.exists() and not (root / "config.json").exists():
                raise ValueError("Partial legacy Dense LMMS needs an explicit resume config or a new baseline root")
            return result("complete" if ready else "pending", "dense_lmms", evidence=str(root))
        if stage == "token":
            from search.optimize import evidence_paths, plan
            self._saved_matches(root, self.token_snapshot())
            planned = plan(self.token_snapshot(), evidence_paths([search_root(self.evaluation), root]))
            ready = (planned["stop_reason"] != "pending_evaluation"
                     and all(row["quality_work"] == "reuse" and row["performance"] in {"reuse", "not_required"}
                             for row in planned["candidates"]))
            passing = any(row["quality"] and row["quality"]["action"] == "pass" for row in planned["candidates"])
            if ready and passing:
                from search.stack import discover_tokens
                sources, _ = discover_tokens(resolve(self.stack["vision_token_root"]), self.vision,
                                             self.stack.get("token_keep_ratios"), model=self.model,
                                             evaluation=self.evaluation)
                allowed = self.stack.get("token_keep_ratios")
                needed = {row["keep_ratio"] for row in planned["candidates"]
                          if row["quality"] and row["quality"]["action"] == "pass"
                          and (allowed is None or row["keep_ratio"] in allowed)}
                # Execute a CPU-only Token handoff when global evidence is
                # complete but not yet discoverable in Stack's source root.
                ready = needed.issubset({row["ratio"] for row in sources})
            return result("complete" if ready and passing else "quality_unmet" if ready else "pending",
                          planned["stop_reason"], evidence=str(root), candidates=len(planned["candidates"]))
        if stage == "sp_probe":
            from search.structured_pruning import snapshot, existing_measurements, pending_shapes
            from skills.structured_pruning import load_calibration
            snap = snapshot(self.job["structured_pruning_config"], self.opt["model_config"],
                            self.opt["evaluation_config"], self.device)
            load_calibration(resolve(snap["config"]["calibration"]["source"]), snap["context"]["dimensions"])
            pending = pending_shapes(snap, existing_measurements(root, snap["context"]))
            return result("pending" if pending else "complete", "width_probe", remaining_shapes=len(pending), evidence=str(root))
        if stage == "sp":
            from search.stack import discover_structured_pruning
            self._saved_matches(root, {"model": self.model, "evaluation": self.evaluation,
                                       "config": self.configs["structured_pruning"], "device": self.device})
            summary = read(root / "summary.json") if (root / "summary.json").exists() else {}
            if summary.get("status") == "completed":
                rows, _ = discover_structured_pruning(root)
                reason = summary.get("stop_reason")
                ready = bool(rows) and reason in {"recovered_boundary", "declared_range_exhausted"}
                return result("complete" if ready else "quality_unmet", reason, evidence=str(root / "summary.json"))
            return result("pending", "sp_search", evidence=str(root))
        if stage == "quantization":
            from search.quantization import snapshot
            from search.stack import discover_quantization
            snap = snapshot(self.job["quantization_config"])
            self._saved_matches(root, {key: snap[key] for key in ("model", "evaluation", "device", "candidates")})
            summary = read(root / "summary.json") if (root / "summary.json").exists() else {}
            rows, _ = discover_quantization(root)
            if summary.get("status") == "completed" and rows and summary.get("selected_candidate"):
                return result("complete", "quantization_qualified", evidence=str(root / "summary.json"))
            if summary.get("status") == "completed":
                probes = [read(p).get("status") for p in root.glob("candidates/*/probe.json")]
                if "quality_rejected" in probes:
                    return result("quality_unmet", "quantization_quality_unmet", evidence=str(root / "summary.json"))
                return result("blocked", "blocked_preflight", evidence=str(root / "summary.json"),
                              error="No qualified quantization route; inspect backend receipts", retryable=True)
            return result("pending", "quantization_search", evidence=str(root))
        if stage == "stack":
            self._saved_matches(root, {"config": self.stack, "model_digest": digest(self.model)}, stack_policy=True)
            path = root / "final/stack-freeze.json"
            if path.exists():
                freeze = read(path)
                if stack_handoff_eligible(freeze) and (
                        freeze.get("status") != "frozen_best_available" or selection_policy(self.stack) == "best_effort"):
                    from skills.stack_vllm import validate_stack_artifact
                    validate_stack_artifact(resolve(freeze["artifact"]))
                    summary = self._lmms(root / "candidates" / freeze["candidate_id"] / "lmms", freeze)
                    met = summary["quality_gate"] == "passed"
                    return result("complete", "lmms_quality_passed" if met else "best_available_selected",
                                  evidence=str(path), retention=summary["retention"], quality_target_met=met)
            state = read(root / "state.json") if (root / "state.json").exists() else {}
            if state.get("quality_policy") == "full_lmms" and state.get("stop_reason") in {"candidate_budget_exhausted", "recovery_ladder_exhausted"}:
                return result("quality_unmet", state["stop_reason"], evidence=str(root / "state.json"))
            return result("pending", "stack_search", evidence=str(root))
        if stage == "serving":
            for key in ("runtime", "kv"):
                cfg = self.configs[key]
                self._saved_matches(resolve(cfg["output"]), {"config": cfg}, stack_policy=True)
            self._saved_matches(root, self.serving)
            parent = read(resolve(self.configs["runtime"]["parent_freeze"]))
            benchmark = self.configs["runtime"].get("parent_benchmark")
            expected_benchmark = self.outputs["stack"] / "candidates" / parent["candidate_id"] / "vllm-benchmark/benchmark.json"
            if benchmark and resolve(benchmark) != expected_benchmark:
                raise ValueError("Runtime parent_benchmark belongs to a different Stack; bind the new parent before running")
            runtime_path = resolve(self.configs["kv"]["parent_freeze"])
            kv_path = resolve(self.serving["kv_freeze"])
            if runtime_path.exists() and read(runtime_path).get("parent_digest") != digest(parent):
                raise ValueError("Runtime evidence belongs to another Stack; use a matching/new runtime job")
            if kv_path.exists() and (not runtime_path.exists() or read(kv_path).get("parent_digest") != digest(read(runtime_path))):
                raise ValueError("KV evidence belongs to another Runtime; use a matching/new KV job")
            path = root / "final/serving-freeze.json"
            if path.exists():
                from search.serving_lmms import plan
                freeze = read(path)
                planned = plan(self.serving)
                matched = next((row for row in planned["candidates"] if row["id"] == freeze.get("id")), None)
                fallback = (selection_policy(self.serving) == "best_effort"
                            and freeze.get("status") == "lmms_evaluated"
                            and freeze.get("quality_target_met") is False
                            and freeze.get("selection_policy") == "best_effort")
                if (freeze.get("status") != "lmms_qualified" and not fallback or not matched
                        or any(freeze.get(key) != value for key, value in matched.items())
                        or freeze.get("stack_digest") != planned["stack_digest"]):
                    raise ValueError("Final serving freeze does not match the requested parent/configuration")
                summary = self._lmms(root / "candidates" / freeze["id"] / "lmms", freeze, serving=True)
                comparison = freeze.get("dense_comparison") or {}
                if (comparison.get("baseline_report_digest") != digest(read(self.dense / "benchmark/summary.json"))
                        or resolve(comparison.get("baseline_report", ".")) != self.dense / "benchmark/summary.json"):
                    raise ValueError("Serving performance denominator no longer matches Dense")
                met = summary["quality_gate"] == "passed"
                return result("complete", "lmms_quality_passed" if met else "best_available_selected",
                              evidence=str(path), retention=summary["retention"], quality_target_met=met,
                              dense_comparison=freeze.get("dense_comparison"), candidate_id=freeze["id"])
            state = read(root / "state.json") if (root / "state.json").exists() else {}
            if state.get("stop_reason") == "serving_quality_unmet":
                return result("quality_unmet", "serving_quality_unmet", evidence=str(root / "state.json"))
            for key in ("runtime", "kv"):
                summary_path = resolve(self.configs[key]["output"]) / "summary.json"
                if summary_path.exists():
                    summary = read(summary_path)
                    if summary.get("status") == "blocked" and not summary.get("selected_candidate"):
                        probes = summary.get("results", [])
                        quality = bool(probes) and all(row.get("status") == "quality_rejected" for row in probes)
                        if not quality:
                            return result("blocked", "blocked_preflight", evidence=str(summary_path), retryable=True)
                        # Legacy Dev rejections are pending performance/full-LMMS work.
            return result("pending", "runtime_kv_final_lmms", evidence=str(root),
                          state_files=[str(resolve(self.configs[key]["output"]) / "state.json")
                                       for key in ("runtime", "kv", "serving_lmms")])
        raise ValueError(f"Unknown stage: {stage}")

    def token_snapshot(self):
        return {"model": self.model, "evaluation": self.evaluation, "vision": self.vision, "device": self.device,
                **({"worker_python": self.opt["worker_python"]} if "worker_python" in self.opt else {})}

    def command(self, stage):
        base = [str(PROJECT_ROOT / self.job["coordinator_python"]), "-B", "-m"]
        root = self.outputs[stage]
        if stage == "dense":
            base[0] = str(PROJECT_ROOT / self.opt.get("worker_python", self.job["coordinator_python"]))
            return base + ["search.baseline", "--model-config", self.opt["model_config"],
                           "--evaluation-config", self.opt["evaluation_config"], "--device", self.platform.worker_device,
                           "--resume", "--output", str(root / "quality")]
        if stage == "dense_lmms":
            return base + ["search.lmms", "--model-config", self.opt["model_config"], "--output", str(root),
                           "--tasks", ",".join(row["task"] for row in self.serving["tasks"])] + (
                               ["--resume"] if (root / "config.json").exists() else [])
        if stage == "token":
            return base + ["search.optimize", "--config", self.job["optimization_config"], "--execute", "--output", str(root)] + (
                ["--resume"] if (root / "config.json").exists() else [])
        if stage in {"sp_probe", "sp"}:
            return base + ["search.structured_pruning" if stage == "sp_probe" else "search.pruning_search",
                           "--config", self.job["structured_pruning_config"], "--model-config", self.opt["model_config"],
                           "--evaluation-config", self.opt["evaluation_config"], "--device", str(self.device),
                           "--execute", "--output", str(root)] + (
                               [] if stage == "sp_probe" else ["--probe-output", str(self.outputs["sp_probe"])] +
                               (["--resume"] if (root / "config.json").exists() else []))
        if stage in {"quantization", "stack"}:
            return base + ["search." + stage, "--config", self.job[stage + "_config"], "--execute"]
        from search.serving_lmms import plan as serving_plan
        try:
            serving_plan(self.serving)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        else:
            return base + ["search.serving_lmms", "--config", self.job["serving_lmms_config"]]
        return base + ["search.stack_serving", "--runtime-config", self.job["runtime_config"],
                       "--kv-config", self.job["kv_config"], "--lmms-config", self.job["serving_lmms_config"]]


def plan(pipeline):
    stages, waiting = {}, None
    for name in STAGES:
        row = pipeline.inspect(name) if waiting is None else result("waiting", "upstream_stage", dependency=waiting)
        stages[name] = row
        if waiting is None and row["status"] != "complete":
            waiting = name
        if row["status"] == "pending":
            row["command"] = pipeline.command(name)
    status = "completed" if waiting is None else stages[waiting]["status"]
    return {"schema": "a4fvlm.pipeline-plan.v1", "status": status,
            "next_stage": waiting, "stages": stages, "output": str(pipeline.root),
            **({"quality_target_met": stages["serving"].get("quality_target_met", True)}
               if waiting is None else {}),
            **({"capabilities": pipeline.capabilities} if hasattr(pipeline, "capabilities") else {})}


def run_command(command, log_path, *, device=None, on_start=None):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PATH"] = str(Path(command[0]).parent) + os.pathsep + env.get("PATH", "")
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=tuple(LOCK_FDS))
        try:
            if on_start:
                on_start(process.pid)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def run_stage(pipeline, stage, directory, on_start):
    """Delegate computation; Dense's two legacy operations share one stage."""
    command = pipeline.command(stage)
    gpu_lock = Path(f"/tmp/agent4vlm-gpu-{os.getuid()}-{pipeline.device}.lock")
    if stage in {"dense", "dense_lmms"}:
        context = lock(gpu_lock)
    else:
        context = nullcontext()  # Stage coordinators own their GPU locks.
    with context:
        if stage != "dense":
            run_command(command, directory / "stage.log", device=pipeline.device if stage == "dense_lmms" else None,
                        on_start=on_start)
            return
        from search.optimize import hardware, launch
        details = pipeline.inspect("dense")
        pipeline.dense.mkdir(parents=True, exist_ok=True)
        save_json(pipeline.dense / "config.json", {"model": pipeline.model, "evaluation": pipeline.evaluation})
        if details["missing_quality"]:
            run_command(command, directory / "stage.log", device=pipeline.device, on_start=on_start)
        if details["missing_benchmark"]:
            request = {"operation": "control", "model": pipeline.model, "evaluation": pipeline.evaluation,
                       "keep_ratio": None, "device": pipeline.device, "hardware": hardware(pipeline.device, pipeline.model),
                       "worker_python": str(PROJECT_ROOT / pipeline.opt.get("worker_python", pipeline.job["coordinator_python"]))}
            attempt = directory / "control"
            launch(request, attempt)
            shutil.copytree(attempt / "benchmark", pipeline.dense / "benchmark", dirs_exist_ok=True)
            for name in ("request.json", "result.json"):
                shutil.copy2(attempt / name, pipeline.dense / name)


def execute(pipeline, *, resume=False, runner=run_stage):
    snapshot = pipeline.snapshot()
    root = pipeline.root
    # One orchestration owner per model, even if callers pick different outputs.
    model_lock = Path(f"/tmp/agent4vlm-pipeline-{os.getuid()}-{pipeline.job['model_id']}.lock")
    with lock(model_lock):
        if resume:
            if read(root / "config.json") != snapshot:
                raise ValueError("Pipeline resume configuration/data changed; inspect and use a new job")
            state = read(root / "state.json")
        else:
            if root.exists():
                # A read-only plan may have been redirected into the pipeline
                # directory before execution.  It is not a resumable job
                # because no snapshot/state exists, but it must not prevent the
                # first real execution.  Preserve that plan while initializing
                # the durable orchestration files.  Any other existing content
                # still requires an explicit --resume/new output to avoid
                # overwriting an interrupted job.
                existing = {path.name for path in root.iterdir()}
                if existing - {"summary.json"}:
                    raise FileExistsError(
                        f"Pipeline output already contains a job: {root}; use --resume or a new output"
                    )
                plan_summary = root / "summary.json"
                if plan_summary.exists():
                    plan_summary.replace(root / "plan-summary.json")
            else:
                root.mkdir(parents=True, exist_ok=False)
            save_json(root / "config.json", snapshot)
            state = {"schema": "a4fvlm.pipeline-state.v1", "stages": {}, "attempts": [], "stage_invocations": 0}
        state.update(status="running", active_stage=None, next_stage=None, stop_reason=None, pid=os.getpid())
        state.pop("error", None)
        state.pop("final_freeze", None)
        state["stages"] = {name: result("waiting", "upstream_stage") for name in STAGES}

        def save():
            save_json(root / "state.json", state)
            save_json(root / "summary.json", {key: value for key, value in state.items() if key not in {"attempts", "pid"}})

        try:
            save()
            for stage in STAGES:
                if pipeline.snapshot() != snapshot:
                    raise ValueError("Pipeline inputs changed during execution")
                state.update(active_stage=stage, next_stage=stage)
                row = pipeline.inspect(stage)
                state["stages"][stage] = row
                save()
                if row["status"] == "complete":
                    row["action"] = "reused"
                    save()
                    continue
                if row["status"] in {"blocked", "quality_unmet"} and not (
                        resume and row["status"] == "blocked" and row.get("retryable")):
                    state.update(status=row["status"], stop_reason=row["reason"])
                    break
                directory = root / "attempts" / f"{len(state['attempts']) + 1:03d}-{stage}"
                directory.mkdir(parents=True)
                attempt = {"stage": stage, "directory": str(directory), "command": pipeline.command(stage)}
                state["attempts"].append(attempt)
                state["stage_invocations"] += 1
                save()

                def started(pid):
                    attempt["pid"] = pid
                    save()

                runner(pipeline, stage, directory, started)
                if pipeline.snapshot() != snapshot:
                    raise ValueError("Pipeline inputs changed while a stage was running")
                row = pipeline.inspect(stage)
                state["stages"][stage] = row | {"action": "executed"}
                if row["status"] != "complete":
                    state.update(status="quality_unmet" if row["status"] == "quality_unmet" else "blocked",
                                 stop_reason=row["reason"] if row["status"] != "pending" else "stage_incomplete")
                    break
                save()
            else:
                met = state["stages"]["serving"].get("quality_target_met", True)
                state.update(status="completed", stop_reason="lmms_quality_passed" if met else "best_available_selected",
                             quality_target_met=met, next_stage=None,
                             final_freeze=state["stages"]["serving"]["evidence"])
        except BaseException as error:
            state.update(status="incomplete", stop_reason="interrupted" if isinstance(error, KeyboardInterrupt)
                         else "execution_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["active_stage"] = None
            save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--execute", action="store_true", help="Run only stages lacking applicable evidence")
    parser.add_argument("--resume", action="store_true", help="Resume the saved orchestration; inputs must match")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Pipeline received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        pipeline = Pipeline(args.config)
        if args.resume and read(pipeline.root / "config.json") != pipeline.snapshot():
            raise ValueError("Pipeline resume configuration/data changed")
        output = execute(pipeline, resume=args.resume) if args.execute else plan(pipeline)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(result("blocked", "blocked_preflight", error=str(error)), ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.execute and output["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
