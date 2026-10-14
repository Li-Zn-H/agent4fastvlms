"""Plan or execute a resumable, reuse-first vision-token search."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from search.evaluate import compare_outputs, load_samples
from search.model import PROJECT_ROOT
from search.context import dense_root, search_root, hardware_adapter, require_stages, validate_result_scope, validate_dense_reference
from search.run import quality_action
from search.vision_token import candidate_ratios
from search.worker import save_json
from skills.vision_token import geometry_keep_count, validate_selection

LOCK_FDS = []

RESULTS = PROJECT_ROOT / "results/model/qwen3vl/search"


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def snapshot(config):
    """Read the selected JSON settings without loading a model or querying GPUs."""
    job = read(PROJECT_ROOT / config)
    snap = {
        "model": read(PROJECT_ROOT / job["model_config"]),
        "evaluation": read(PROJECT_ROOT / job["evaluation_config"]),
        "vision": read(PROJECT_ROOT / job["vision_token_config"]),
        "device": job["device"],
    }
    require_stages(snap["model"], ("dense", "token"), device=snap["device"])
    validate_result_scope(snap["model"], snap["evaluation"])
    if "worker_python" in job:
        snap["worker_python"] = job["worker_python"]
    validate_selection(snap["vision"]["selection"])
    candidate_ratios(snap["vision"]["search"])
    return snap


def evidence_paths(roots):
    """Include existing result directories even when they predate worker metadata."""
    paths = set()
    for root in roots:
        if not root.exists():
            continue
        paths.update(root.rglob("request.json"))
        for name in ("quality", "benchmark"):
            paths.update(
                directory.parent / "request.json"
                for directory in root.rglob(name)
                if directory.is_dir()
            )
        paths.update(
            path.parent / "request.json" for path in root.rglob("geometry.json")
        )
    return sorted(paths)


def recorded_request(path):
    if path.exists():
        return read(path)
    directory = path.parent
    if directory.name.startswith("keep-"):
        return {"operation": "candidate", "keep_ratio": float(directory.name[5:]) / 100}
    return {
        "operation": "prepare"
        if (directory / "geometry.json").exists()
        else ("control" if directory.name == "dense" else "unknown"),
        "keep_ratio": None,
    }


def compatible(request, snap, ratio):
    # Existing records in this model's Search directory are the user's experiment.
    # Missing historical metadata does not turn completed work into new work.
    strict = "result_root" in snap["evaluation"]
    if (strict or "model" in request) and request.get("model") != snap.get("model"):
        return False
    if strict and request.get("evaluation") != snap["evaluation"]:
        return False
    if not strict and "evaluation" in request and any(
        request["evaluation"].get(key) != snap["evaluation"].get(key) for key in ("data", "generation")
    ):
        return False
    return request.get("keep_ratio") == ratio and (
        ratio is None
        or request.get("selection", snap["vision"]["selection"])
        == snap["vision"]["selection"]
    )


def stage_score(snap, stage, path):
    config = snap["evaluation"]
    validate_dense_reference(snap.get("model"), config, project_root=PROJECT_ROOT)
    data = config["data"]
    samples = load_samples(
        PROJECT_ROOT / data[stage], media_root=PROJECT_ROOT / data["media_root"]
    )
    baseline = dense_root(config, project_root=PROJECT_ROOT, legacy=RESULTS) / "quality" / f"{stage}.jsonl"
    options = {
        key: config["scoring"][key]
        for key in ("maximum_response_tokens", "negation_window_tokens")
    }
    score = compare_outputs(samples, lines(baseline), lines(path), **options)
    return {
        **asdict(score),
        "action": quality_action(score.retention, stage=stage, **config["quality"]),
    }


def inspect_quality(snap, ratio, paths):
    sources, stages = {}, {}
    for stage in ("search", "dev"):
        for path in paths:
            try:
                request = recorded_request(path)
                output = path.parent / "quality" / f"{stage}.jsonl"
                if request.get("operation") != "candidate" or not compatible(
                    request, snap, ratio
                ):
                    continue
                score = stage_score(snap, stage, output)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            sources[stage] = str(output)
            stages[stage] = score
            break
        if stage not in stages:
            return sources, None
        if stages[stage]["action"] != "dev":
            return sources, {"action": stages[stage]["action"], "stages": stages}
    raise AssertionError("Dev must decide quality")


def inspect_benchmark(snap, ratio, paths, hardware=None) -> dict[str, Any] | None:
    for path in paths:
        try:
            request = recorded_request(path)
            if not compatible(request, snap, ratio):
                continue
            if "result_root" in snap["evaluation"] and (not request.get("hardware") or
                    (hardware is not None and request["hardware"] != hardware)):
                continue
            if request.get("operation") in ("prepare", "unknown"):
                continue
            summary_path = path.parent / "benchmark/summary.json"
            summary = read(summary_path)
            records = lines(path.parent / "benchmark/metrics.jsonl")
            samples = load_samples(
                PROJECT_ROOT / snap["evaluation"]["data"]["search"],
                media_root=PROJECT_ROOT / snap["evaluation"]["data"]["media_root"],
            )
            expected = [row["stable_row_id"] for row in samples]
            rounds = summary["rounds"]
            if not rounds or len(records) != len(expected) * len(rounds):
                continue
            totals = []
            for number, details in enumerate(rounds, 1):
                batch = [row for row in records if row["round"] == number]
                if [row["sample_id"] for row in batch] != expected:
                    break
                total = sum(row["performance"]["end_to_end_seconds"] for row in batch)
                if (
                    details["sample_count"] != len(expected)
                    or not math.isfinite(total)
                    or total <= 0
                    or not math.isclose(total, details["total_seconds"])
                ):
                    break
                totals.append(total)
            if len(totals) != len(rounds) or not math.isclose(
                sum(totals) / len(totals), summary["mean_round_seconds"]
            ):
                continue
            return {"source": str(summary_path), "summary": summary}
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
    return None


def plan(snap, paths):
    candidates = []
    failures = 0
    stop = "declared_range_exhausted"
    for ratio in candidate_ratios(snap["vision"]["search"]):
        sources, quality = inspect_quality(snap, ratio, paths)
        performance = inspect_benchmark(snap, ratio, paths)
        candidates.append(
            {
                "keep_ratio": ratio,
                "quality_sources": sources,
                "quality": quality,
                "quality_work": "reuse"
                if quality
                else ("dev" if "search" in sources else "search_then_conditional_dev"),
                "performance": "not_required"
                if quality and quality["action"] == "reject"
                else ("reuse" if performance else "measure_if_quality_passes"),
                "benchmark_source": performance["source"] if performance else None,
            }
        )
        if quality is None:
            failures = 0
            stop = "pending_evaluation"
        elif quality["action"] == "pass":
            failures = 0
        else:
            failures += 1
        if failures >= snap["vision"]["search"]["consecutive_dev_failures"]:
            stop = "consecutive_dev_failures"
            break
    control = inspect_benchmark(snap, None, paths)
    return {
        "device": snap["device"],
        "candidates": candidates,
        "stop_reason": stop,
        "control_source": control["source"] if control else None,
        "reuse_policy": "Reuse complete existing records; only run missing stages. Historical round counts and missing phase fields do not trigger reruns.",
    }


@contextmanager
def lock(path):
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another coordinator holds {path}") from error
        LOCK_FDS.append(stream.fileno())
        try:
            yield
        finally:
            LOCK_FDS.remove(stream.fileno())
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(request, directory):
    model = request.get("model", {})
    platform = hardware_adapter(model)
    if request.get("device") is not None:
        platform.validate_device(request["device"])
    if model.get("path"):
        stage = {
            "control": "dense", "prepare": "token", "candidate": "token", "benchmark": "token",
            "structured_pruning_candidate": "sp", "structured_pruning_probe": "sp",
            "quantization_framework_probe": "quantization", "quantization_candidate": "quantization",
            "quantization_runtime_validation": "quantization", "quantization_vllm_benchmark": "quantization",
            "stack_candidate": "stack", "stack_runtime_validation": "runtime",
            "stack_runtime_candidate": "runtime", "stack_kv_cache_candidate": "kv_cache",
        }.get(request["operation"])
        if stage:
            require_stages(model, (stage,), device=request.get("device"))
    return platform.resolve("launch_worker")(request, directory)


def _cuda_launch(request, directory):
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "request.json", request)
    worker_python = request.get("worker_python", sys.executable)
    if not worker_python:
        worker_python = sys.executable
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ""
        if request["device"] is None
        else request["hardware"].split(",", 1)[0].strip(),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Some hosted environments use zero as a placeholder for thread limits.
    # OpenMP and PyTorch reject it, and every runtime child inherits this
    # environment. Remove invalid values at the common process boundary.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = env.get(variable)
        try:
            valid = value is not None and int(value) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            env.pop(variable, None)
    # Invoking an interpreter by absolute path does not activate its virtual
    # environment.  Backend runtimes such as vLLM spawn compiler helpers
    # (``ninja``, CUDA utilities, etc.) by PATH, so expose the selected
    # environment's bin directory to the Worker and its child processes.
    # Keep the virtualenv path itself.  The interpreter is often a symlink to
    # uv's shared Python; resolving it would remove the env's ``bin`` directory
    # and hide helpers such as ninja from spawned runtimes.
    worker_bin = str(Path(worker_python).parent)
    env["PATH"] = worker_bin + os.pathsep + env.get("PATH", "")
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                worker_python,
                "-B",
                "-m",
                "search.worker",
                "--request",
                str(directory / "request.json"),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=tuple(LOCK_FDS),
        )
        try:
            code = process.wait()
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
    result_path = directory / "result.json"
    result = (
        read(result_path)
        if result_path.exists()
        else {
            "status": "failed",
            "failure": "missing_result",
            "error": f"Worker exit code {code}",
        }
    )
    if code or result.get("status") != "completed":
        raise RuntimeError(f"Worker failed in {directory}: {result}")
    return result


def hardware(device, model=None):
    platform = hardware_adapter(model or {})
    platform.validate_device(device)
    return platform.resolve("probe_identity")(device)


def _cuda_hardware(device):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return output.strip()


def run_summary(state):
    """Keep the checkpoint authoritative; summarize without copying its evidence."""
    result = {
        key: state[key]
        for key in (
            "status",
            "stop_reason",
            "best_measured_keep_ratio",
            "speedup",
            "worker_seconds",
            "gpu_reserved_seconds",
            "resource_accounting_complete",
            "error",
        )
        if key in state
    }
    candidates = state.get("candidates", [])
    result.update(
        checkpoint="state.json",
        evaluated_candidates=sum("quality" in row for row in candidates),
        reused_candidates=sum(bool(row.get("reused_all")) for row in candidates),
        duplicate_configurations=sum(
            row.get("status") == "duplicate_physical_configuration"
            for row in candidates
        ),
    )
    return result


def execute(snap, destination, max_candidates=None):
    state_path = destination / "state.json"
    state: dict[str, Any] = (
        read(state_path) if state_path.exists() else {"attempts": [], "candidates": []}
    )
    roots = [search_root(snap["evaluation"], project_root=PROJECT_ROOT, legacy=RESULTS), destination]
    paths = evidence_paths(roots)
    device = snap["device"]
    state.pop("error", None)
    state.pop("active", None)
    state["status"] = "running"
    save_json(state_path, state)

    def request(operation, ratio=None, sources=None, measure=True):
        return {
            **{key: snap[key] for key in ("model", "evaluation")},
            "operation": operation,
            "device": None if operation == "prepare" else device,
            "selection": snap["vision"]["selection"],
            "keep_ratio": ratio,
            "quality_sources": sources or {},
            "measure": measure,
            "hardware": machine,
            "benchmark_protocol": {"warmup": 5, "repeats": 1},
            **({"worker_python": snap["worker_python"]} if "worker_python" in snap else {}),
        }

    machine = None

    def run(operation, label, ratio=None, sources=None, measure=True):
        nonlocal machine
        if operation != "prepare" and machine is None:
            machine = hardware(device, snap["model"])
        directory = destination / label / f"attempt-{len(state['attempts']) + 1}"
        state["attempts"].append(str(directory))
        state["active"] = str(directory)
        save_json(state_path, state)
        print(f"{operation}: {label}", flush=True)
        result = launch(request(operation, ratio, sources, measure), directory)
        paths.append(directory / "request.json")
        state.pop("active", None)
        save_json(state_path, state)
        return result

    try:
        geometry = None
        for path in paths:
            try:
                if (path.parent / "geometry.json").exists():
                    source = recorded_request(path)
                    if "result_root" in snap["evaluation"] and any(
                        source.get(key) != snap[key] for key in ("model", "evaluation")
                    ):
                        continue
                    rows = read(path.parent / "geometry.json")
                    data = snap["evaluation"]["data"]
                    expected_ids = {
                        sample["stable_row_id"]
                        for stage in ("search", "dev")
                        for sample in load_samples(
                            PROJECT_ROOT / data[stage],
                            media_root=PROJECT_ROOT / data["media_root"],
                        )
                    }
                    if (
                        len(rows) == len(expected_ids)
                        and {row["sample_id"] for row in rows} == expected_ids
                        and all(
                            type(row[key]) is int and row[key] > 0
                            for row in rows
                            for key in ("rows", "columns")
                        )
                    ):
                        geometry = rows
                        break
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if geometry is None:
            geometry = read(run("prepare", "preparation")["geometry"])
        grid = snap["vision"]["selection"]["coverage_grid"]
        previous = tuple(geometry_keep_count(row, 1, grid) for row in geometry)
        seen = {previous}
        failures = 0
        records = []
        state["candidates"] = records
        control = inspect_benchmark(snap, None, paths, machine)
        executed = 0
        stop = "declared_range_exhausted"
        for ratio in candidate_ratios(snap["vision"]["search"]):
            physical = tuple(
                geometry_keep_count(row, ratio, grid)
                for row in geometry
            )
            if physical in seen:
                records.append(
                    {"keep_ratio": ratio, "status": "duplicate_physical_configuration"}
                )
                continue
            if any(new > old for new, old in zip(physical, previous, strict=True)):
                raise ValueError("Token count increased")
            sources, quality = inspect_quality(snap, ratio, paths)
            performance = inspect_benchmark(snap, ratio, paths, machine)
            work = quality is None or (
                quality["action"] == "pass" and performance is None
            )
            if work and max_candidates is not None and executed >= max_candidates:
                stop = "invocation_limit"
                break
            seen.add(physical)
            previous = physical
            label = f"candidates/keep-{ratio * 100:g}"
            if quality is None:
                run("candidate", label, ratio, sources, measure=performance is None)
                sources, quality = inspect_quality(snap, ratio, paths)
                performance = inspect_benchmark(snap, ratio, paths, machine)
                if quality is None:
                    raise RuntimeError(
                        "Worker did not produce complete quality evidence"
                    )
            if quality["action"] == "pass":
                failures = 0
                if performance is None:
                    run("benchmark", label, ratio)
                    performance = inspect_benchmark(snap, ratio, paths, machine)
                if control is None:
                    run("control", "control")
                    control = inspect_benchmark(snap, None, paths, machine)
                if performance is None or control is None:
                    raise RuntimeError("Incomplete benchmark evidence")
            else:
                failures += 1
            executed += int(work)
            records.append(
                {
                    "keep_ratio": ratio,
                    "quality": quality,
                    "quality_sources": sources,
                    "benchmark": performance if quality["action"] == "pass" else None,
                    "confirmed_failure_streak": failures,
                    "reused_all": not work,
                }
            )
            save_json(state_path, state)
            if failures >= snap["vision"]["search"]["consecutive_dev_failures"]:
                stop = "consecutive_dev_failures"
                break
        state.update(
            status="incomplete" if stop == "invocation_limit" else "completed",
            stop_reason=stop,
            control=control,
        )
        measured = [row for row in records if row.get("benchmark")]
        best = (
            min(
                measured,
                key=lambda row: row["benchmark"]["summary"]["mean_round_seconds"],
            )
            if measured
            else None
        )
        state["best_measured_keep_ratio"] = best["keep_ratio"] if best else None
        state["speedup"] = (
            control["summary"]["mean_round_seconds"]
            / best["benchmark"]["summary"]["mean_round_seconds"]
            if best and control is not None
            else None
        )
    except BaseException as error:
        state.update(
            status="incomplete",
            stop_reason="execution_error",
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
        save_json(destination / "summary.json", run_summary(state))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.json")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if (args.execute or args.resume) and not args.output:
        parser.error("--output is required for execute/resume")
    destination = (PROJECT_ROOT / args.output).resolve() if args.output else None
    if args.resume:
        assert destination is not None
        snap = read(destination / "config.json")
    else:
        snap = snapshot(args.config)
    if not args.execute:
        roots = [search_root(snap["evaluation"], project_root=PROJECT_ROOT, legacy=RESULTS)] + ([destination] if destination else [])
        print(json.dumps(plan(snap, evidence_paths(roots)), indent=2))
        return

    assert destination is not None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Coordinator received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    with lock(Path("/tmp") / f"agent4vlm-gpu-{os.getuid()}-{snap['device']}.lock"):
        if not args.resume:
            destination.mkdir(parents=True, exist_ok=False)
            save_json(destination / "config.json", snap)
        with lock(destination / ".run.lock"):
            print(json.dumps(execute(snap, destination, args.max_candidates), indent=2))


if __name__ == "__main__":
    main()
