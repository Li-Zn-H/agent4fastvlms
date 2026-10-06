"""One isolated preparation, Dense control, or candidate execution per process."""

import argparse
import asyncio
import ctypes
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
import threading
from time import perf_counter
from typing import Any


def save_json(path, value):
    """Replace a complete JSON record atomically, including after interruption."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _vllm_reload_subprocess(artifact_dir: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    """Construct vLLM in a clean process after quantizer work is complete.

    GPTQModel and Transformers can leave CUDA allocator state behind even
    after their models are deleted. vLLM's KV-cache planner sees that state
    during engine startup, so an in-process reload can fail despite a clean
    standalone vLLM check. A fresh interpreter gives the runtime an honest
    VRAM snapshot and keeps backend-specific process state out of the Worker.
    """

    script = """
import json
import sys
import vllm
from skills.stack_vllm import prepare_vllm_artifact
from skills.stack_runtime import build_vllm_runtime_kwargs

artifact = sys.argv[1]
runtime = json.loads(sys.argv[2])
adapter = prepare_vllm_artifact(artifact)
kwargs = {"model": artifact, "trust_remote_code": True}
if adapter.get("model_class_overrides"):
    kwargs["model_class_overrides"] = adapter["model_class_overrides"]
if adapter.get("custom_stack"):
    # vLLM 0.26's V1 engine starts a fresh EngineCore process.  A registry
    # mutation in the launcher process alone is therefore insufficient; the
    # public model_class_overrides field re-registers this module in every
    # process that inspects or constructs the model.
    kwargs["model_class_overrides"] = {
        adapter["architecture"]: "skills.stack_vllm:" + adapter["architecture"]
    }
if runtime.get("vllm_gpu_memory_utilization") is not None:
    kwargs["gpu_memory_utilization"] = float(runtime["vllm_gpu_memory_utilization"])
if runtime.get("vllm_max_model_len") is not None:
    kwargs["max_model_len"] = int(runtime["vllm_max_model_len"])
kwargs.update(build_vllm_runtime_kwargs(runtime))
server = vllm.LLM(**kwargs)
del server
print("a4fvlm_stack_adapter=" + json.dumps(adapter, sort_keys=True), flush=True)
print("vllm_reload_ok", flush=True)
"""
    environment = os.environ.copy()
    # ``sys.executable`` may be a symlink into uv's shared interpreter.  Use
    # the selected virtualenv prefix so its console scripts remain visible.
    worker_bin = str(Path(sys.prefix) / "bin")
    environment["PATH"] = worker_bin + os.pathsep + environment.get("PATH", "")
    # vLLM's optional FlashInfer sampler JIT invokes ninja.  A backend
    # environment may be intentionally minimal; keep the score-based Token
    # path usable there without changing the Dense environment or installing
    # packages at runtime.
    if shutil.which("ninja", path=environment["PATH"]) is None:
        environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    timeout = int(runtime.get("serve_timeout_seconds", 120))
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(artifact_dir),
            json.dumps(runtime, ensure_ascii=True),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"vLLM reload subprocess failed with exit code {result.returncode}: "
            f"{details[-4000:]}"
        )
    for line in (result.stdout or "").splitlines():
        marker = "a4fvlm_stack_adapter="
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                break
    return {}


def execute(request, destination):
    # These imports happen only in an explicitly launched execution process.
    from search.model import PROJECT_ROOT, load_model, load_processor, prepare_inputs, processor_adapter
    from search.context import hardware_adapter

    operation = request["operation"]
    if operation == "serving_use":
        from search.serving import worker
        return worker(request, destination)
    if operation == "quantization_framework_probe":
        return run_quantization_framework_probe(request)
    if operation == "quantization_candidate":
        return run_quantization_candidate(request)
    if operation == "quantization_runtime_validation":
        return run_quantization_runtime_validation(request)
    if operation == "quantization_vllm_benchmark":
        return run_quantization_vllm_benchmark(request)
    if operation == "stack_runtime_validation":
        return run_stack_runtime_validation(request)
    if operation == "stack_runtime_candidate":
        return run_stack_acceleration_candidate(request)
    if operation == "stack_kv_cache_candidate":
        return run_stack_acceleration_candidate(request)
    if operation == "stack_candidate":
        return run_stack_candidate(request)
    if operation == "structured_pruning_probe":
        return run_pruning_probe(request)
    if operation == "structured_pruning_candidate":
        from search.run import evaluate_candidate
        from skills.structured_pruning import audit_recipe

        recipe = json.loads(Path(request["recipe"]).read_text())
        model, processor = load_model(
            {**request["model"], "device": hardware_adapter(request["model"]).worker_device}, structured_pruning=recipe
        )
        handles = []
        mechanism = None
        observations = [
            {"prefill_calls": 0, "decode_calls": 0} for _ in recipe["widths"]
        ]
        try:
            mechanism = audit_recipe(model, recipe)
            for index, layer in enumerate(model.model.language_model.layers):

                def observe(module, args, output, layer_index=index):
                    phase = (
                        "decode_calls" if args[0].shape[-2] == 1 else "prefill_calls"
                    )
                    observations[layer_index][phase] += 1

                handles.append(layer.mlp.register_forward_hook(observe))
            save_json(destination / "mechanism.json", mechanism)
            result = {}
            if request.get("evaluate", True):
                result["quality"] = evaluate_candidate(
                    model,
                    processor,
                    request["evaluation"],
                    output_dir=destination / "quality",
                    reuse_outputs=request.get("quality_sources"),
                    force_dev=request.get("force_dev", False),
                )
            # Keep diagnostic hooks out of timed forwards.
            for handle in handles:
                handle.remove()
            handles.clear()
            if request.get("measure", False):
                from search.benchmark import benchmark

                result["benchmark"] = benchmark(
                    model,
                    processor,
                    request["evaluation"],
                    output_dir=destination / "benchmark",
                )
            return result
        finally:
            for handle in handles:
                handle.remove()
            if mechanism is not None:
                mechanism["forward_observations"] = observations
                save_json(destination / "mechanism.json", mechanism)
            del model, processor
    if operation == "prepare":
        from search.evaluate import load_samples

        processor = load_processor(request["model"])
        geometry_for = processor_adapter(processor).resolve("token_geometry")
        data = request["evaluation"]["data"]
        unique = {}
        for stage in ("search", "dev"):
            for row in load_samples(
                PROJECT_ROOT / data[stage], media_root=PROJECT_ROOT / data["media_root"]
            ):
                if (
                    row["stable_row_id"] in unique
                    and row != unique[row["stable_row_id"]]
                ):
                    raise ValueError("Search and Dev disagree on a shared sample")
                unique[row["stable_row_id"]] = row
        geometry = []
        for sample in unique.values():
            inputs = prepare_inputs(processor, sample, data["media_root"])
            geometry.append(
                {
                    "sample_id": sample["stable_row_id"],
                    **geometry_for(processor, inputs),
                }
            )
        save_json(destination / "geometry.json", geometry)
        return {"geometry": str(destination / "geometry.json")}
    if operation not in ("control", "candidate", "benchmark"):
        raise ValueError("Unknown worker operation")
    from search.benchmark import benchmark
    from search.run import evaluate_candidate

    model_config = {**request["model"], "device": hardware_adapter(request["model"]).worker_device}
    model: Any
    model, processor = load_model(model_config)
    result = {}
    try:
        if operation in ("candidate", "benchmark"):
            model.vision_token_controller = processor_adapter(processor).resolve("token_controller")(
                model, request["selection"], request["keep_ratio"]
            )
        if operation == "candidate":
            result["quality"] = evaluate_candidate(
                model,
                processor,
                request["evaluation"],
                output_dir=destination / "quality",
                reuse_outputs=request.get("quality_sources"),
            )
        if operation in ("control", "benchmark") or (
            result["quality"]["action"] == "pass" and request.get("measure", True)
        ):
            result["benchmark"] = benchmark(
                model,
                processor,
                request["evaluation"],
                output_dir=destination / "benchmark",
            )
        return result
    finally:
        if hasattr(model, "vision_token_controller"):
            del model.vision_token_controller
        # The process exits after this request; no runtime or CUDA context is reused.
        del model


def run_pruning_probe(request):
    """Worker-only GPU work; no Dense model weights or quality generation."""
    import torch
    import transformers

    snap = request["probe"]
    output = Path(request["probe_output"])
    path = output / "measurements.json"
    records = json.loads(path.read_text())["records"] if path.exists() else []
    dimensions = snap["context"]["dimensions"]
    protocol = snap["config"]["hardware_probe"]
    adapter_id = request.get("model", {}).get("adapter", "qwen3_vl")
    if adapter_id == "llava_onevision":
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        mlp_config = Qwen2Config
        mlp_class = Qwen2MLP
        default_dtype = "float16"
    elif adapter_id == "llava15":
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        mlp_config = LlamaConfig
        mlp_class = LlamaMLP
        default_dtype = "float16"
    else:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        mlp_config = Qwen3VLTextConfig
        mlp_class = Qwen3VLTextMLP
        default_dtype = "bfloat16"
    dtype_name = request.get("model", {}).get("dtype", default_dtype)
    dtype = getattr(torch, dtype_name)
    environment = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "hardware": request["hardware"],
    }
    failed = 0
    for shape in request["shapes_to_measure"]:
        row = {**shape, "environment": environment, **protocol}
        module = inputs = result = None
        try:
            torch.manual_seed(0)
            config = mlp_config(
                **{
                    **dimensions,
                    "intermediate_size": shape["width"],
                    "hidden_act": snap["context"].get("hidden_act", "silu"),
                }
            )
            with torch.device(f"cuda:{request.get('device', 0)}"):
                module = mlp_class(config).to(dtype=dtype).eval()
                inputs = torch.randn(
                    (1, shape["input_tokens"], dimensions["hidden_size"]),
                    dtype=dtype,
                )
            with torch.inference_mode():
                for _ in range(protocol["warmup_iterations"]):
                    result = module(inputs)
                assert result is not None
                if not bool(torch.isfinite(result).all()):
                    raise ValueError("Non-finite SwiGLU output")
                if list(result.shape) != [
                    1,
                    shape["input_tokens"],
                    dimensions["hidden_size"],
                ]:
                    raise ValueError("Unexpected SwiGLU output shape")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                events = []
                for _ in range(protocol["measurement_iterations"]):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    result = module(inputs)
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize()
                row.update(
                    status="measured",
                    finite_output=True,
                    milliseconds=[start.elapsed_time(end) for start, end in events],
                    timed_batch_wall_seconds=time.perf_counter() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    weight_shapes={
                        name: list(getattr(module, name).weight.shape)
                        for name in ("gate_proj", "up_proj", "down_proj")
                    },
                )
        except torch.OutOfMemoryError as error:
            row.update(status="oom", error=str(error))
        except (RuntimeError, ValueError) as error:
            row.update(
                status="execution_error", error=f"{type(error).__name__}: {error}"
            )
        finally:
            del module, inputs, result
            torch.cuda.empty_cache()
        failed += int(row["status"] != "measured")
        records.append(row)
        save_json(
            output / "measurements.json",
            {"context": snap["context"], "records": records},
        )
        print(f"{shape}: {row['status']}", flush=True)
    return {
        "probe_status": "incomplete" if failed else "completed",
        "failed_shapes": failed,
    }


def _stack_report(request):
    candidate = request["candidate"]
    return {
        "schema": "a4fvlm.stack-result.v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_key": candidate["candidate_key"],
        "recovery": candidate.get("recovery"),
        "skills": candidate.get("skills", []),
        "transformation_order": [
            "structured_pruning",
            "quantization",
            "vision_token",
        ],
        "fresh_evidence": False,
        "metrics_transferable": False,
        "mechanism": {"status": "not_run"},
        "reload": {"status": "not_run"},
        "quality": {"status": "pending_full_lmms"},
        "benchmark": {"status": "not_run"},
    }


def _finish_stack_block(destination, report, status, reason, message=None):
    report["status"] = status
    report["mechanism"] = {"status": "blocked", "reason": reason}
    if message:
        report["mechanism"]["message"] = message
    save_json(destination / "probe.json", report)
    return {"probe_status": status, "report": str(destination / "probe.json")}


def _run_stack_vllm_benchmark(request, artifact_dir: Path, destination: Path) -> dict[str, Any]:
    """Run the optional vLLM runtime benchmark after the clean reload gate."""

    runtime = dict(request.get("runtime", {}))
    runtime.setdefault("vllm_gpu_memory_utilization", 0.8)
    runtime.setdefault("vllm_max_model_len", 8192)
    runtime.setdefault("vllm_benchmark_limit", 128)
    runtime.setdefault("vllm_benchmark_warmup", 5)
    runtime.setdefault("vllm_benchmark_repeats", 1)
    runtime.setdefault("vllm_benchmark_batch_size", 1)
    benchmark_request = {
        "operation": "quantization_vllm_benchmark",
        "device": request.get("device"),
        "model_role": "stack_sp_quantized_vllm_topology",
        "model": {"path": str(artifact_dir)},
        "evaluation": request["evaluation"],
        "runtime": runtime,
        "probe_output": str(destination / "vllm-benchmark"),
    }
    if runtime.get("vllm_benchmark_split") is not None:
        benchmark_request["benchmark_split"] = runtime["vllm_benchmark_split"]
    candidate = request.get("candidate") or {}
    if candidate.get("token_keep_ratio") is not None:
        benchmark_request["vision_token"] = {
            "keep_ratio": candidate["token_keep_ratio"],
            "selection": request.get("selection"),
        }
    result = run_quantization_vllm_benchmark(benchmark_request)
    if result.get("probe_status") != "vllm_benchmark_passed":
        report_path = result.get("report")
        details = "vLLM benchmark failed"
        if report_path and Path(report_path).is_file():
            try:
                details = json.loads(Path(report_path).read_text()).get(
                    "error", details
                )
            except (OSError, ValueError, TypeError):
                pass
        raise RuntimeError(details)
    report_path = Path(result["report"])
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_stack_acceleration_candidate(request):
    """Benchmark one runtime layer on the already frozen Stack artifact.

    The Worker never rewrites the artifact here.  It only changes vLLM engine
    settings and records independent performance and held-out quality receipts.
    """

    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    candidate = request.get("candidate") or {}
    report = {
        "schema": "a4fvlm.stack-acceleration-result.v1",
        "kind": "kv_cache" if request.get("operation") == "stack_kv_cache_candidate" else "runtime",
        "candidate_id": candidate.get("id"),
        "candidate_key": candidate.get("candidate_key"),
        "parent_freeze": request.get("parent_freeze"),
        "parent_digest": request.get("parent_digest"),
        "artifact": str(request.get("artifact", "")),
        "runtime": request.get("runtime", {}),
        "benchmarks": {},
        "quality": {"status": "pending_full_lmms"},
    }
    try:
        artifact = Path(str(request["artifact"]))
        if not artifact.is_dir():
            raise FileNotFoundError(f"Stack artifact is missing: {artifact}")
        from skills.stack_vllm import validate_stack_artifact

        validate_stack_artifact(artifact)
        profiles = request.get("profiles") or []
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("runtime candidate requires at least one benchmark profile")
        for profile in profiles:
            if not isinstance(profile, dict) or not profile.get("name"):
                raise ValueError("invalid runtime benchmark profile")
            runtime = dict(request.get("runtime", {}))
            runtime.update(
                {
                    "vllm_benchmark_limit": int(profile["limit"]),
                    "vllm_benchmark_warmup": int(profile["warmup"]),
                    "vllm_benchmark_repeats": int(profile["repeats"]),
                    "vllm_benchmark_batch_size": int(profile["batch_size"]),
                }
            )
            benchmark_request = {
                "operation": "quantization_vllm_benchmark",
                "device": request.get("device"),
                "model_role": f"stack_{report['kind']}_runtime",
                "model": {"path": str(artifact)},
                "evaluation": request["evaluation"],
                "runtime": runtime,
                "benchmark_split": profile.get("split", "search"),
                "probe_output": str(destination / f"vllm-benchmark-{profile['name']}"),
                "vision_token": {
                    "keep_ratio": request.get("token_keep_ratio"),
                    "selection": request.get("selection"),
                },
            }
            if benchmark_request["vision_token"]["keep_ratio"] is None:
                benchmark_request.pop("vision_token")
            result = run_quantization_vllm_benchmark(benchmark_request)
            if result.get("probe_status") != "vllm_benchmark_passed":
                report["status"] = "runtime_blocked"
                report["error"] = "vLLM benchmark did not complete"
                save_json(destination / "probe.json", report)
                return {"probe_status": report["status"], "report": str(destination / "probe.json")}
            report_path = Path(result["report"])
            benchmark = json.loads(report_path.read_text(encoding="utf-8"))
            report["benchmarks"][profile["name"]] = {
                "status": benchmark.get("status"),
                "summary": benchmark.get("summary"),
                "report": str(report_path.resolve()),
                "runtime": benchmark.get("runtime", {}),
            }

        report["status"] = "performance_ready"
        report["fresh_evidence"] = True
        save_json(destination / "probe.json", report)
        return {"probe_status": "performance_ready", "report": str(destination / "probe.json")}
    except (ImportError, TypeError, ValueError, RuntimeError, FileNotFoundError) as error:
        report["status"] = "blocked_preflight" if isinstance(error, (ImportError, TypeError, ValueError)) else "runtime_blocked"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}


def run_stack_sp_quantized_candidate(request, report=None, destination=None):
    """Materialize SP, re-quantize that physical parent, then run Token."""

    from search.benchmark import benchmark
    from search.model import load_model
    from skills.stack_adapters import StackAdapterError, export_pruned_gptqmodel
    from skills.stack_model import load_stack_model, write_pruned_parent
    from skills.structured_pruning import audit_recipe
    from search.context import model_adapter

    VisionTokenController = model_adapter(request["model"]).resolve("token_controller")

    if destination is None:
        destination = Path(request["probe_output"])
        destination.mkdir(parents=True, exist_ok=True)
    if report is None:
        report = _stack_report(request)
    candidate = request["candidate"]
    sp = candidate.get("structured_pruning") or {}
    quant = candidate.get("quantization") or {}
    token = candidate.get("vision_token")
    if not token:
        return _finish_stack_block(destination, report, "stack_invalid", "token_source_required")
    route = quant.get("route")
    # Legacy callers that only carry a Dense artifact have not supplied a
    # topology route at all. Keep this explicit compatibility receipt rather
    # than pretending that artifact can be reused after physical SP.
    if route is None:
        return _finish_stack_block(
            destination,
            report,
            "stack_bridge_blocked",
            "sp_quantization_bridge_required",
            "A Dense GPTQ artifact cannot be reused after physical SP; a route-aware adapter is required.",
        )
    worker_python = request.get("worker_python") or quant.get("worker_python")
    calibration = request.get("calibration") or quant.get("calibration")
    quant_runtime = (
        request.get("quantization_runtime")
        or quant.get("runtime_config")
        or request.get("runtime")
        or {}
    )
    if not worker_python:
        return _finish_stack_block(
            destination, report, "stack_bridge_blocked", "quantization_worker_python_missing"
        )
    if not isinstance(calibration, dict) or not calibration.get("data"):
        return _finish_stack_block(
            destination, report, "stack_bridge_blocked", "stack_calibration_missing"
        )
    calibration_path = Path(str(calibration["data"]))
    if not calibration_path.is_absolute():
        calibration_path = Path(__file__).resolve().parents[1] / calibration_path
    if not calibration_path.is_file():
        return _finish_stack_block(
            destination, report, "stack_bridge_blocked", "stack_calibration_missing"
        )
    if quant_runtime.get("quantize") is not True:
        return _finish_stack_block(
            destination,
            report,
            "stack_bridge_blocked",
            "stack_quantization_not_enabled",
        )
    if route != "gptqmodel_w4a16":
        return _finish_stack_block(
            destination,
            report,
            "stack_bridge_blocked",
            "unsupported_sp_quantization_route",
            "Only the validated GPTQModel W4A16 route has a topology adapter.",
        )
    recipe_path = sp.get("recipe_path")
    if not recipe_path or not Path(recipe_path).is_file():
        return _finish_stack_block(destination, report, "stack_bridge_blocked", "sp_recipe_missing")

    parent_dir = destination / "parent"
    artifact_dir = destination / "artifact"
    model = processor = None
    stack_model = stack_processor = None
    try:
        if request.get("resume_artifact"):
            from skills.stack_model import validate_resumable_export

            artifact_dir = Path(request["resume_artifact"])
            saved_request = json.loads(Path(request["export_request"]).read_text())
            report["export_reuse"] = validate_resumable_export(artifact_dir, saved_request, request)
            report["mechanism"] = dict(request["prior_probe"]["mechanism"])
            export = {"manifest": str(artifact_dir / "manifest.json")}
        else:
            recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
            model, processor = load_model(request["model"], structured_pruning=recipe)
            physical = audit_recipe(model, recipe)
            write_pruned_parent(model, processor, recipe, parent_dir)
            module_names = [name for name, module in model.named_modules() if module.__class__.__name__ == "Linear"]
            report["mechanism"] = {
                "status": "parent_written",
                "structured_pruning": True,
                "quantization": True,
                "vision_token": True,
                "parent_recipe_digest": sp.get("recipe_digest"),
                "parent_widths": recipe.get("widths"),
                "physical_shapes": physical,
                "fresh_shape_reparse": True,
            }
            del model
            model = None
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            try:
                export = export_pruned_gptqmodel(
                    candidate=quant,
                    parent_dir=parent_dir,
                    processor=processor,
                    calibration=calibration,
                    output_dir=artifact_dir,
                    model={**request["model"], "path": str(parent_dir)},
                    module_names=module_names,
                    recipe_digest=sp.get("recipe_digest", ""),
                )
            except (StackAdapterError, ImportError, RuntimeError, ValueError) as error:
                return _finish_stack_block(
                    destination,
                    report,
                    "stack_bridge_blocked",
                    "sp_quantization_export_failed",
                    f"{type(error).__name__}: {error}",
                )
        report["mechanism"].update(
            {
                "status": "passed",
                "artifact": str(artifact_dir),
                "manifest": export.get("manifest"),
                "fresh_calibration": True,
            }
        )
        processor = None
        stack_model, stack_processor = load_stack_model(
            {**request["model"], "path": str(artifact_dir), "device": "cuda:0"}
        )
        report["reload"] = {
            "status": "passed",
            "backend": "transformers",
            "vllm": {
                "status": "skipped" if not request.get("runtime", {}).get("vllm_reload", False) else "required"
            },
        }
        stack_model.vision_token_controller = VisionTokenController(
            stack_model, request["selection"], candidate["token_keep_ratio"]
        )
        performance = benchmark(
            stack_model,
            stack_processor,
            request["evaluation"],
            output_dir=destination / "benchmark",
            warmup=int(request.get("runtime", {}).get("warmup", 5)),
            repeats=int(request.get("runtime", {}).get("repeats", 1)),
        )
        report["benchmark"] = {
            "status": "passed",
            "mean_seconds": performance.get("mean_round_seconds"),
            "peak_allocated_bytes": max(
                (row.get("peak_allocated_bytes", 0) for row in performance.get("rounds", [])),
                default=0,
            ),
        }
        if request.get("runtime", {}).get("vllm_reload", False):
            try:
                # Release the Transformers model before starting another
                # runtime on this GPU. The clean child then gets an honest VRAM
                # snapshot, and the optional AsyncLLMEngine can start safely.
                stack_model = None
                stack_processor = None
                import gc

                gc.collect()
                import torch

                torch.cuda.empty_cache()
                adapter = _vllm_reload_subprocess(
                    artifact_dir, request.get("runtime", {})
                )
            except Exception as error:  # noqa: BLE001 - runtime preflight is not quality
                report["reload"]["vllm"] = {
                    "status": "blocked",
                    "reason": "stack_vllm_reload_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                report["status"] = "stack_runtime_blocked"
                save_json(destination / "probe.json", report)
                return {"probe_status": report["status"], "report": str(destination / "probe.json"), "artifact": str(artifact_dir)}
            report["reload"]["vllm"] = {"status": "passed", **adapter}
            if request.get("runtime", {}).get("vllm_benchmark", False):
                try:
                    vllm_report = _run_stack_vllm_benchmark(
                        request, artifact_dir, destination
                    )
                except Exception as error:  # noqa: BLE001 - runtime gate
                    report["vllm_benchmark"] = {
                        "status": "blocked",
                        "reason": "stack_vllm_benchmark_failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    report["status"] = "stack_runtime_blocked"
                    save_json(destination / "probe.json", report)
                    return {
                        "probe_status": report["status"],
                        "report": str(destination / "probe.json"),
                        "artifact": str(artifact_dir),
                    }
                report["vllm_benchmark"] = {
                    "status": "passed",
                    "summary": vllm_report.get("summary"),
                    "report": str(
                        destination / "vllm-benchmark" / "benchmark.json"
                    ),
                    "adapter": vllm_report.get("adapter"),
                    "vision_token": vllm_report.get("vision_token"),
                }
            report["status"] = "ready_for_lmms"
        else:
            report["status"] = "mechanism_ready"
        report["artifact"] = str(artifact_dir)
        report["fresh_evidence"] = True
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json"), "artifact": str(artifact_dir)}
    except Exception as error:  # noqa: BLE001 - preserve a durable receipt
        report["status"] = "candidate_error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc(limit=40)
        result = {"probe_status": report["status"], "report": str(destination / "probe.json")}
        if report.get("mechanism", {}).get("status") == "passed" and (artifact_dir / "manifest.json").is_file():
            report["artifact"] = str(artifact_dir)
            result["artifact"] = str(artifact_dir)
        save_json(destination / "probe.json", report)
        return result
    finally:
        for value in (model, processor, stack_model, stack_processor):
            if value is not None:
                del value


def run_stack_runtime_validation(request):
    """Complete vLLM gates for an existing development Stack artifact.

    Enabling the vLLM route after a Transformers-only run must not repeat SP
    materialization or GPTQ calibration. This operation reuses the durable
    candidate artifact and prior mechanism/performance evidence.
    """

    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(request.get("artifact", ""))
    prior = request.get("prior_probe") or {}
    report = json.loads(json.dumps(prior))
    report["artifact"] = str(artifact_dir)
    report["quality"] = {"status": "pending_full_lmms"}
    try:
        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Stack runtime artifact is missing: {artifact_dir}"
            )
        if request.get("runtime", {}).get("vllm_reload") is not True:
            raise ValueError("Stack runtime validation requires vllm_reload=true")
        import gc

        gc.collect()
        adapter = _vllm_reload_subprocess(
            artifact_dir, request.get("runtime", {})
        )
        report.setdefault("reload", {})["vllm"] = {"status": "passed", **adapter}
        if request.get("runtime", {}).get("vllm_benchmark", False):
            vllm_report = _run_stack_vllm_benchmark(
                request, artifact_dir, destination
            )
            report["vllm_benchmark"] = {
                "status": "passed",
                "summary": vllm_report.get("summary"),
                "report": str(
                    destination / "vllm-benchmark" / "benchmark.json"
                ),
                "adapter": vllm_report.get("adapter"),
                "vision_token": vllm_report.get("vision_token"),
            }
        report["fresh_evidence"] = True
        report["status"] = "ready_for_lmms"
        save_json(destination / "probe.json", report)
        return {
            "probe_status": "ready_for_lmms",
            "report": str(destination / "probe.json"),
            "artifact": str(artifact_dir),
        }
    except Exception as error:  # noqa: BLE001 - persist runtime block
        report["status"] = "stack_runtime_blocked"
        report.setdefault("reload", {}).setdefault("vllm", {})
        report["reload"]["vllm"].update(
            {
                "status": "blocked",
                "reason": "stack_vllm_runtime_validation_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        save_json(destination / "probe.json", report)
        return {
            "probe_status": report["status"],
            "report": str(destination / "probe.json"),
            "artifact": str(artifact_dir),
        }


def run_stack_candidate(request):
    """Evaluate one bounded Stack candidate in a clean Worker."""

    from search.benchmark import benchmark
    from search.model import load_model
    from search.context import model_adapter

    VisionTokenController = model_adapter(request["model"]).resolve("token_controller")

    candidate = request["candidate"]
    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    report = _stack_report(request)
    sp = candidate.get("structured_pruning")
    quant = candidate.get("quantization")
    token = candidate.get("vision_token")
    if sp is not None and quant is not None:
        return run_stack_sp_quantized_candidate(request, report, destination)
    if token is None:
        return _finish_stack_block(destination, report, "stack_invalid", "token_source_required")

    model = processor = None
    try:
        model_config = dict(request["model"])
        if quant is not None:
            artifact = quant.get("artifact")
            if not artifact or not Path(artifact).is_dir():
                raise FileNotFoundError("quantized Stack artifact is missing")
            model_config["path"] = artifact
        recipe = None
        if sp is not None:
            recipe_path = sp.get("recipe_path")
            if not recipe_path:
                raise ValueError("SP Stack source has no recipe_path")
            recipe = json.loads(Path(recipe_path).read_text())
        model, processor = load_model(model_config, structured_pruning=recipe)
        report["mechanism"] = {"status": "passed", "structured_pruning": sp is not None, "quantization": quant is not None, "vision_token": True}
        report["reload"] = {"status": "passed", "backend": "transformers", "vllm": {"status": "skipped"}}
        model.vision_token_controller = VisionTokenController(model, request["selection"], candidate["token_keep_ratio"])
        performance = benchmark(model, processor, request["evaluation"], output_dir=destination / "benchmark", warmup=int(request.get("runtime", {}).get("warmup", 5)), repeats=int(request.get("runtime", {}).get("repeats", 1)))
        report["benchmark"] = {"status": "passed", "mean_seconds": performance.get("mean_round_seconds"), "peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0) for row in performance.get("rounds", [])), default=0)}
        report["fresh_evidence"] = True
        if request.get("runtime", {}).get("vllm_reload", False):
            artifact = quant.get("artifact") if quant is not None else model_config.get("path")
            try:
                if not artifact or not Path(artifact).is_dir():
                    raise FileNotFoundError("Stack runtime artifact is missing")
                import gc
                del model, processor
                model = processor = None
                gc.collect()
                adapter = _vllm_reload_subprocess(
                    Path(artifact), request.get("runtime", {})
                )
                report["reload"]["vllm"] = {"status": "passed", **adapter}
                report["status"] = "ready_for_lmms"
            except Exception as error:  # noqa: BLE001 - runtime preflight is not quality
                report["reload"]["vllm"] = {"status": "blocked", "reason": "stack_vllm_reload_failed", "error": f"{type(error).__name__}: {error}"}
                report["status"] = "stack_runtime_blocked"
        else:
            report["status"] = "mechanism_ready"
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}
    except Exception as error:  # noqa: BLE001 - receipt must classify the candidate
        report["status"] = "candidate_error"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor


def run_quantization_framework_probe(request):
    """Validate one quantization candidate against the Dense Qwen3-VL model.

    This operation deliberately has no structured-pruning recipe input.  The
    first pass validates dependency/API/model coverage only.  Actual export is
    opt-in through the JSON runtime settings after a candidate is accepted.
    """
    import importlib
    import importlib.metadata
    import importlib.util

    from skills.quantization import validate_module_inventory

    candidate = request["candidate"]
    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    imports = candidate.get("imports", [])
    missing = [name for name in imports if importlib.util.find_spec(name) is None]
    runtime_imports = candidate.get("runtime_imports", [])
    missing_runtime = [
        name for name in runtime_imports if importlib.util.find_spec(name) is None
    ]
    runtime_status = {}
    for name in runtime_imports:
        try:
            importlib.import_module(name)
            runtime_status[name] = {
                "available": True,
                "version": importlib.metadata.version(name),
            }
        except Exception as error:
            runtime_status[name] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    report = {
        "schema": "a4fvlm.quantization-probe.v1",
        "candidate": candidate["id"],
        "route": candidate["route"],
        "framework": candidate["framework"],
        "runtime": request.get("runtime", {}),
        "dependencies": {},
        "runtime_dependencies": runtime_status,
        "stages": {},
    }
    for name in imports + ["transformers"]:
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    if missing:
        report["status"] = "blocked_missing_dependency"
        report["stages"]["dependency"] = {"status": "failed", "missing": missing}
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}

    model = processor = None
    try:
        report["stages"]["dependency"] = {"status": "passed", "missing": []}
        from search.model import load_model

        model, processor = load_model(request["model"])
        import torch

        linear_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        inventory = validate_module_inventory(
            linear_names,
            require_vision=True,
            model_adapter=request.get("model", {}).get("adapter", "qwen3_vl"),
        )
        report["module_inventory"] = inventory
        report["stages"]["model_inventory"] = {
            "status": "passed" if inventory["valid"] else "failed",
            "linear_count": len(linear_names),
        }
        if not inventory["valid"]:
            report["status"] = "incompatible_model_topology"
            save_json(destination / "probe.json", report)
            return {"probe_status": report["status"], "report": str(destination / "probe.json")}

        # Check the backend's public API without pretending that an unexecuted
        # quantization is a successful artifact build.
        if candidate["route"] == "gptqmodel_w4a16":
            package = importlib.import_module("gptqmodel")
            api = {
                "GPTQModel": hasattr(package, "GPTQModel"),
                "QuantizeConfig": hasattr(package, "QuantizeConfig"),
            }
        elif candidate["route"] in {"llm_compressor_w4a16", "llm_compressor_fp8"}:
            package = importlib.import_module("llmcompressor")
            api = {"oneshot": hasattr(package, "oneshot")}
        elif candidate["route"] == "autoround_w4a16":
            package = importlib.import_module("auto_round")
            api = {
                "AutoRound": hasattr(package, "AutoRound"),
                "export": hasattr(package, "export_to_tokenizer"),
            }
        else:
            importlib.import_module("torchao")
            try:
                quantization = importlib.import_module("torchao.quantization")
                api = {"quantize": hasattr(quantization, "quantize_")}
            except Exception:
                api = {"quantize": False}
        report["backend_api"] = api
        report["stages"]["backend_api"] = {
            "status": "passed" if all(api.values()) else "failed",
            "available": api,
        }
        export_requested = bool(request.get("runtime", {}).get("quantize", False))
        report["stages"]["quantization"] = {
            "status": "export_not_implemented"
            if export_requested
            else "not_run",
            "reason": "The framework probe intentionally stops before export; add a backend skill after selecting a route."
            if export_requested
            else "Set runtime.quantize=true only after selecting a framework route",
        }
        report["stages"]["vllm_reload"] = {
            "status": "blocked_missing_dependency"
            if any(not row["available"] for row in runtime_status.values())
            else "not_run",
            "reason": "Requires a successfully exported candidate artifact"
            if all(row["available"] for row in runtime_status.values())
            else f"Runtime import unavailable: {runtime_status}",
        }
        backend_ready = all(api.values())
        runtime_ready = all(
            row["available"] for row in runtime_status.values()
        )
        if not backend_ready:
            report["status"] = "backend_api_incomplete"
        elif not runtime_ready:
            # The framework can be imported and inspected, but the candidate
            # cannot yet be validated as a deployable route without its
            # configured runtime (normally vLLM).
            report["status"] = "framework_ready_runtime_blocked"
        else:
            report["status"] = "framework_ready"
        if export_requested:
            report["status"] = "framework_ready_export_not_implemented"
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}
    except Exception as error:
        report["status"] = "probe_error"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(destination / "probe.json", report)
        return {"probe_status": report["status"], "report": str(destination / "probe.json")}
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor


def run_quantization_candidate(request):
    """Run the opt-in Dense quantization lifecycle in one isolated Worker.

    The framework probe remains the default operation.  This operation adds a
    strict lifecycle receipt for a future route adapter; it never reports a
    deployable artifact unless every stage and both quality/runtime gates have
    completed.  Keeping this boundary in the Worker prevents quantizer imports
    from leaking into planning or the structured-pruning process.
    """
    import importlib
    import importlib.metadata
    import importlib.util

    from skills.quantization import stage_template, validate_module_inventory

    candidate = request["candidate"]
    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    stages = stage_template()
    report = {
        "schema": "a4fvlm.quantization-candidate.v1",
        "candidate": candidate["id"],
        "route": candidate["route"],
        "framework": candidate["framework"],
        "stages": stages,
        "quality": {"status": "not_run"},
        "performance": {"status": "not_run"},
    }
    imports = candidate.get("imports", [])
    missing = [name for name in imports if importlib.util.find_spec(name) is None]
    if missing:
        report["status"] = "blocked_missing_dependency"
        report["stages"]["prepare"] = {
            "status": "blocked",
            "missing": missing,
        }
        save_json(destination / "probe.json", report)
        return {
            "probe_status": report["status"],
            "report": str(destination / "probe.json"),
        }

    # Export is explicit because the default probe must remain cheap and
    # side-effect free.  A route adapter is the only code allowed to create an
    # artifact; no copy of the Dense model is treated as quantized output.
    if not request.get("runtime", {}).get("quantize", False):
        report["status"] = "framework_ready_export_not_requested"
        report["reason"] = "Set runtime.quantize=true after selecting a route adapter"
        save_json(destination / "probe.json", report)
        return {
            "probe_status": report["status"],
            "report": str(destination / "probe.json"),
        }

    try:
        from search.model import load_model

        report["stages"]["prepare"] = {"status": "running"}
        model, processor = load_model(request["model"])
        try:
            import torch

            names = [
                name
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
            ]
            inventory = validate_module_inventory(
                names,
                require_vision=True,
                model_adapter=request.get("model", {}).get("adapter", "qwen3_vl"),
            )
            report["module_inventory"] = inventory
            if not inventory["valid"]:
                report["status"] = "incompatible_model_topology"
                report["stages"]["prepare"] = {
                    "status": "failed",
                    "inventory": inventory,
                }
                save_json(destination / "probe.json", report)
                return {
                    "probe_status": report["status"],
                    "report": str(destination / "probe.json"),
                }
            report["stages"]["prepare"] = {
                "status": "passed",
                "linear_count": len(names),
                "assignment_digest": inventory["assignment_digest"],
            }

            calibration = request.get("calibration", {})
            data = calibration.get("data")
            if candidate.get("calibration_required") and not data:
                raise ValueError("Calibration data is required for this candidate")
            report["stages"]["calibrate"] = {
                "status": "passed",
                "data": data,
                "limit": calibration.get("limit"),
            }

            # Route-specific backend code is deliberately isolated in a
            # separately installable adapter module.  The first implementation
            # must be validated against the installed framework before it can
            # be allowed to emit an artifact.
            try:
                adapter = importlib.import_module("skills.quantization_adapters")
                export = getattr(adapter, "export_candidate")
            except (ImportError, AttributeError):
                report["status"] = "blocked_adapter_not_implemented"
                report["stages"]["export"] = {
                    "status": "blocked",
                    "reason": "No validated route adapter is installed",
                }
                save_json(destination / "probe.json", report)
                return {
                    "probe_status": report["status"],
                    "report": str(destination / "probe.json"),
                }
            artifact_dir = destination / "artifact"
            export_result = export(
                candidate=candidate,
                model=request["model"],
                processor=processor,
                calibration=calibration,
                output_dir=artifact_dir,
                module_names=names,
            )
            # The adapter loads its own quantizer model.  Release the Dense
            # Transformers model before loading the exported artifact so the
            # reload gate measures the artifact rather than two checkpoints
            # competing for the same GPU.
            model = None
            processor = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            report["stages"]["export"] = {
                "status": "passed",
                **(export_result or {}),
            }
            manifest = export_result.get("manifest") if export_result else None
            if not manifest:
                raise ValueError("Route adapter did not return a manifest")
            report["stages"]["manifest"] = {
                "status": "passed",
                "path": str(manifest),
            }

            # Clean Transformers reload is mandatory before quality or timing is
            # considered. The artifact is still temporary at this point.
            reload_config = {
                **request["model"],
                "path": str(artifact_dir),
                "device": "cuda:0",
            }
            reloaded, reload_processor = load_model(reload_config)
            report["stages"]["reload"] = {"status": "passed"}

            evaluation = request.get("evaluation")
            if not evaluation:
                raise ValueError("evaluation config is required for candidate qualification")
            from search.run import evaluate_candidate

            quality_dir = destination / "quality"
            quality = evaluate_candidate(
                reloaded,
                reload_processor,
                evaluation,
                output_dir=quality_dir,
                force_dev=True,
            )
            quality_stages = quality.get("stages", {})
            dev = quality_stages.get("dev", {})
            report["quality"] = {
                "status": "passed" if quality.get("action") == "pass" else "rejected",
                "action": quality.get("action"),
                "search_retention": quality_stages.get("search", {}).get("retention"),
                "dev_retention": dev.get("retention"),
            }
            if quality.get("action") != "pass":
                report["status"] = "quality_rejected"
                save_json(destination / "probe.json", report)
                return {
                    "probe_status": report["status"],
                    "report": str(destination / "probe.json"),
                }

            from search.benchmark import benchmark

            benchmark_dir = destination / "benchmark"
            performance = benchmark(
                reloaded,
                reload_processor,
                evaluation,
                output_dir=benchmark_dir,
                warmup=int(request.get("runtime", {}).get("warmup", 5)),
                repeats=int(request.get("runtime", {}).get("repeats", 1)),
            )
            report["performance"] = {
                "status": "passed",
                "mean_round_seconds": performance.get("mean_round_seconds"),
                "peak_allocated_bytes": max(
                    row.get("peak_allocated_bytes", 0)
                    for row in performance.get("rounds", [])
                ),
            }

            # A vLLM check is an actual model construction, not merely an
            # import check. It runs only after the artifact passes Transformers
            # reload and quality gates.
            if request.get("runtime", {}).get("vllm_reload", False):
                # vLLM starts a child engine and needs enough free VRAM for
                # its KV cache.  Do not leave the Transformers reload alive
                # while constructing that second runtime.
                reloaded = None
                reload_processor = None
                gc.collect()
                torch.cuda.empty_cache()
                _vllm_reload_subprocess(
                    artifact_dir, request.get("runtime", {})
                )
            report["vllm_reload"] = {
                "status": "passed"
                if request.get("runtime", {}).get("vllm_reload", False)
                else "skipped"
            }
            report["status"] = "qualified"
            report["artifact"] = str(artifact_dir)
            save_json(destination / "probe.json", report)
            return {
                "probe_status": report["status"],
                "report": str(destination / "probe.json"),
                "artifact": str(artifact_dir),
            }
        finally:
            if "reloaded" in locals():
                del reloaded, reload_processor
            del model, processor

    except Exception as error:
        report["status"] = "candidate_error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc(limit=40)
        save_json(destination / "probe.json", report)
        return {
            "probe_status": report["status"],
            "report": str(destination / "probe.json"),
        }


def run_quantization_runtime_validation(request):
    """Resume only the deploy-runtime gate for an exported artifact.

    Quality and benchmark evidence are already durable when this operation is
    requested.  The coordinator uses it after a late vLLM/environment failure
    so a corrected runtime does not force a second quantization pass.
    """

    artifact_dir = Path(request["artifact"])
    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    prior = request.get("prior_probe", {})
    report = {
        "schema": "a4fvlm.quantization-candidate.v1",
        "candidate": request["candidate"]["id"],
        "route": request["candidate"]["route"],
        "framework": request["candidate"]["framework"],
        "stages": dict(prior.get("stages", {})),
        "quality": prior.get("quality", {"status": "not_run"}),
        "performance": prior.get("performance", {"status": "not_run"}),
        "module_inventory": prior.get("module_inventory"),
        "reuse_provenance": request.get("reuse_provenance"),
        "runtime_config": request.get("runtime", {}),
    }
    try:
        manifest = artifact_dir / "manifest.json"
        if not artifact_dir.is_dir() or not manifest.is_file():
            raise FileNotFoundError(
                f"retry artifact is missing or has no manifest: {artifact_dir}"
            )
        report.setdefault("stages", {})["manifest"] = {
            "status": "passed",
            "path": str(manifest),
        }
        from search.model import load_model

        reloaded, reload_processor = load_model(
            {**request["model"], "path": str(artifact_dir), "device": "cuda:0"}
        )
        del reloaded, reload_processor
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        report.setdefault("stages", {})["reload"] = {"status": "passed"}
        runtime = request.get("runtime", {})
        if runtime.get("vllm_reload", False):
            _vllm_reload_subprocess(artifact_dir, runtime)
        report["vllm_reload"] = {
            "status": "passed" if runtime.get("vllm_reload", False) else "skipped",
            "runtime_config": runtime,
        }
        report["status"] = "qualified"
        report["artifact"] = str(artifact_dir)
        save_json(destination / "probe.json", report)
        return {
            "probe_status": "qualified",
            "report": str(destination / "probe.json"),
            "artifact": str(artifact_dir),
        }
    except Exception as error:
        report["status"] = "candidate_error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc(limit=40)
        save_json(destination / "probe.json", report)
        return {
            "probe_status": "candidate_error",
            "report": str(destination / "probe.json"),
        }


class _GpuMemorySampler:
    """Poll nvidia-smi so vLLM child-process memory is included."""

    def __init__(self):
        self.maximum = 0
        self.current = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _read(self):
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            values = [int(value.strip()) for value in output.splitlines() if value.strip()]
            if values:
                self.current = max(values) * 1024 * 1024
                self.maximum = max(self.maximum, self.current)
        except (OSError, subprocess.SubprocessError, ValueError):
            return

    def _poll(self):
        while not self._stop.is_set():
            self._read()
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._read()

    def mark_baseline(self):
        self._read()
        self.baseline = self.current
        self.maximum = self.current


async def _stream_vllm_request(engine, prompt_spec, sampling, request_id):
    """Stream one multimodal request and capture true first-token timing."""

    from vllm.renderers import ChatParams

    started = perf_counter()
    conversation = prompt_spec["conversation"]
    mm_processor_kwargs = prompt_spec.get("mm_processor_kwargs")
    _, engine_inputs = engine.renderer.render_chat(
        [conversation],
        ChatParams(
            chat_template_kwargs={"add_generation_prompt": True},
            mm_processor_kwargs=mm_processor_kwargs,
        ),
    )
    engine_input = engine_inputs[0]
    prompt_token_ids = engine_input.get("prompt_token_ids", [])
    hf_config = getattr(engine.model_config, "hf_config", None)
    image_token_id = getattr(hf_config, "image_token_id", getattr(hf_config, "image_token_index", None))
    image_tokens = (
        sum(int(token) == int(image_token_id) for token in prompt_token_ids)
        if image_token_id is not None
        else None
    )
    first_token = None
    output_tokens = 0
    output_text = ""
    async for output in engine.generate(engine_inputs[0], sampling, request_id):
        if not output.outputs:
            continue
        output_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        output_tokens = max(output_tokens, token_count)
        if token_count and first_token is None:
            first_token = perf_counter()
    finished = perf_counter()
    return {
        "latency": finished - started,
        "ttft": None if first_token is None else first_token - started,
        "tokens": output_tokens,
        "image_tokens": image_tokens,
        "output_text": output_text,
    }


async def _run_streaming_vllm_benchmark(
    *, engine, prompts, sampling, warmup, repeats, batch_size
):
    """Run warmup and repeated streaming requests on one async vLLM engine."""

    async def run_batch(batch, prefix):
        return await asyncio.gather(
            *(
                _stream_vllm_request(
                    engine,
                    prompt,
                    sampling,
                    f"{prefix}-{index}",
                )
                for index, prompt in enumerate(batch)
            )
        )

    if warmup:
        for start in range(0, warmup, batch_size):
            await run_batch(prompts[start : start + batch_size], "warmup")
    rounds = []
    all_latencies = []
    all_ttfts = []
    all_image_tokens = []
    all_output_texts = []
    total_tokens = 0
    total_elapsed = 0.0
    for repeat in range(repeats):
        round_started = perf_counter()
        results = []
        for start in range(0, len(prompts), batch_size):
            results.extend(
                await run_batch(
                    prompts[start : start + batch_size],
                    f"round-{repeat}-{start}",
                )
            )
        round_elapsed = perf_counter() - round_started
        round_latencies = [row["latency"] for row in results]
        round_ttfts = [row["ttft"] for row in results]
        round_tokens = sum(row["tokens"] for row in results)
        all_image_tokens.extend(
            row["image_tokens"] for row in results if row["image_tokens"] is not None
        )
        all_output_texts.extend(row["output_text"] for row in results)
        from search.vllm_benchmark import summarize_round

        rounds.append(
            summarize_round(
                elapsed_seconds=round_elapsed,
                output_tokens=round_tokens,
                request_latencies=round_latencies,
                ttfts=round_ttfts,
                request_count=len(prompts),
            )
        )
        all_latencies.extend(round_latencies)
        all_ttfts.extend(round_ttfts)
        total_tokens += round_tokens
        total_elapsed += round_elapsed
    return (
        rounds,
        all_latencies,
        all_ttfts,
        total_tokens,
        total_elapsed,
        all_image_tokens,
        all_output_texts,
    )


async def _run_async_vllm_engine(engine_args, prompts, sampling, warmup, repeats, batch_size, memory):
    """Create and drive AsyncLLMEngine on one event loop."""

    from vllm import AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    try:
        await engine.check_health()
        memory.mark_baseline()
        return await _run_streaming_vllm_benchmark(
            engine=engine,
            prompts=prompts,
            sampling=sampling,
            warmup=warmup,
            repeats=repeats,
            batch_size=batch_size,
        )
    finally:
        engine.shutdown()


def run_quantization_vllm_benchmark(request):
    """Benchmark one model through vLLM in an isolated Worker process."""

    destination = Path(request["probe_output"])
    destination.mkdir(parents=True, exist_ok=True)
    runtime = request.get("runtime", {})
    report = {
        "schema": "a4fvlm.vllm-benchmark.v5",
        "model_role": request.get("model_role", "unknown"),
        "model": request["model"].get("path"),
        "benchmark_split": request.get("benchmark_split", "search"),
        "runtime": {
            key: runtime[key]
            for key in (
                "vllm_gpu_memory_utilization",
                "vllm_max_model_len",
                "vllm_benchmark_limit",
                "vllm_benchmark_warmup",
                "vllm_benchmark_repeats",
                "vllm_benchmark_batch_size",
                "attention_backend",
                "optimization_level",
                "kv_cache_dtype",
                "compile_mm_encoder",
                "cudagraph_mm_encoder",
                "qk_norm_rope_fusion",
                "enable_prefix_caching",
            )
            if key in runtime
        },
    }
    try:
        if shutil.which("ninja") is None:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from search.evaluate import load_samples
        from search.model import PROJECT_ROOT
        from search.vllm_benchmark import chat_prompt, summarize_round
        from skills.stack_vllm import prepare_vllm_artifact
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        token_runtime = request.get("vision_token") or {}
        token_keep_ratio = token_runtime.get("keep_ratio")
        adapter = prepare_vllm_artifact(
            request["model"]["path"],
            token_runtime=token_runtime if token_keep_ratio is not None else None,
        )
        report["adapter"] = adapter
        adapter_id = adapter.get("model_type") or request["model"].get("adapter", "qwen3_vl")
        evaluation = request["evaluation"]
        benchmark_split = request.get("benchmark_split", "search")
        if benchmark_split not in ("search", "dev"):
            raise ValueError("benchmark_split must be search or dev")
        media_root = PROJECT_ROOT / evaluation["data"]["media_root"]
        samples = list(
            load_samples(
                PROJECT_ROOT / evaluation["data"][benchmark_split],
                media_root=media_root,
            )
        )
        limit = int(runtime.get("vllm_benchmark_limit", len(samples)))
        if limit < 1:
            raise ValueError("vllm_benchmark_limit must be positive")
        samples = samples[:limit]
        prompts = []
        for sample in samples:
            image_path = (
                media_root / sample["media"]["logical_path"]
            ).resolve()
            prompt_spec = {"conversation": chat_prompt(sample, str(image_path))}
            prompts.append(prompt_spec)
        if token_keep_ratio is not None:
            report["vision_token"] = {
                "status": "enabled",
                "method": "vllm_score_based_post_encoder",
                "requested_keep_ratio": float(token_keep_ratio),
                "selection": token_runtime.get("selection"),
                "score": "post_encoder_main_visual_feature_l2_norm",
                "placeholder_count": "selected_count",
            }
            if adapter_id == "qwen3_vl":
                report["vision_token"]["mrope_pruner_switch"] = {
                    "vllm_video_pruning_rate_sentinel": 1e-9,
                    "purpose": "enable_vLLM_dynamic_mrope_rewrite_for_images",
                }
            else:
                report["vision_token"]["position_type"] = "1d_compacted"
        generation = evaluation["generation"]
        sampling_kwargs = {
            "temperature": 0.0,
            "max_tokens": int(generation["max_new_tokens"]),
        }
        if generation.get("ignore_eos", False):
            sampling_kwargs["ignore_eos"] = True
        sampling = SamplingParams(**sampling_kwargs)
        llm_kwargs = {
            "model": str(request["model"]["path"]),
            "dtype": request["model"].get("dtype", "auto"),
            "trust_remote_code": True,
            "allowed_local_media_path": str(media_root.resolve()),
        }
        if token_keep_ratio is not None and adapter_id == "qwen3_vl":
            # vLLM 0.26 exposes its multimodal-pruner switch through the
            # video_pruning_rate config field.  The custom Qwen3-VL class
            # performs image selection itself; this tiny positive sentinel
            # enables the shared mRoPE/position-rewrite path without changing
            # the image algorithm (the benchmark accepts still images only).
            llm_kwargs["video_pruning_rate"] = 1e-9
        if adapter.get("custom_stack"):
            llm_kwargs["model_class_overrides"] = {
                adapter["architecture"]: "skills.stack_vllm:" + adapter["architecture"]
            }
        elif adapter.get("model_class_overrides"):
            llm_kwargs["model_class_overrides"] = adapter["model_class_overrides"]
        if runtime.get("vllm_gpu_memory_utilization") is not None:
            llm_kwargs["gpu_memory_utilization"] = float(
                runtime["vllm_gpu_memory_utilization"]
            )
        if runtime.get("vllm_max_model_len") is not None:
            llm_kwargs["max_model_len"] = int(runtime["vllm_max_model_len"])
        from skills.stack_runtime import build_vllm_runtime_kwargs

        llm_kwargs.update(build_vllm_runtime_kwargs(runtime))
        with _GpuMemorySampler() as memory:
            engine_args = AsyncEngineArgs(
                **llm_kwargs,
                disable_log_stats=True,
                enable_log_requests=False,
            )
            warmup = int(runtime.get("vllm_benchmark_warmup", 5))
            if warmup < 0 or warmup > len(prompts):
                raise ValueError("vllm_benchmark_warmup is outside benchmark samples")
            repeats = int(runtime.get("vllm_benchmark_repeats", 3))
            if repeats < 1:
                raise ValueError("vllm_benchmark_repeats must be positive")
            batch_size = int(runtime.get("vllm_benchmark_batch_size", 1))
            if batch_size < 1:
                raise ValueError("vllm_benchmark_batch_size must be positive")
            (
                rounds,
                all_latencies,
                all_ttfts,
                total_tokens,
                total_elapsed,
                all_image_tokens,
                all_output_texts,
            ) = asyncio.run(
                _run_async_vllm_engine(
                    engine_args,
                    prompts,
                    sampling,
                    warmup,
                    repeats,
                    batch_size,
                    memory,
                )
            )
        if token_keep_ratio is not None:
            image_tokens = [int(value) for value in all_image_tokens]
            if not image_tokens:
                raise RuntimeError("vLLM Token path produced no image token counts")
            report["vision_token"].update(
                {
                    "actual_image_tokens": {
                        "count": len(image_tokens),
                        "min": min(image_tokens),
                        "max": max(image_tokens),
                        "mean": sum(image_tokens) / len(image_tokens),
                    }
                }
            )
        summary = summarize_round(
            elapsed_seconds=total_elapsed,
            output_tokens=total_tokens,
            request_latencies=all_latencies,
            ttfts=all_ttfts,
            request_count=len(prompts) * repeats,
        )
        report.update(
            {
                "sample_count": len(samples),
                "rounds": rounds,
                "summary": summary,
                "engine_loaded_memory_bytes": memory.baseline or None,
                "peak_inference_memory_bytes": memory.maximum or None,
                "inference_memory_delta_bytes": (
                    max(memory.maximum - memory.baseline, 0)
                    if memory.maximum and memory.baseline
                    else None
                ),
                "peak_gpu_memory_bytes": memory.maximum or None,
                "status": "passed",
            }
        )
        save_json(destination / "benchmark.json", report)
        return {
            "probe_status": "vllm_benchmark_passed",
            "report": str(destination / "benchmark.json"),
        }
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        save_json(destination / "benchmark.json", report)
        return {
            "probe_status": "vllm_benchmark_failed",
            "report": str(destination / "benchmark.json"),
        }
def failure_kind(error):
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    if "OutOfMemory" in type(error).__name__ or "out of memory" in str(error).lower():
        return "oom"
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return "input_error"
    return "execution_error"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text())
    destination = request_path.parent
    # Linux parent-death notification prevents orphan GPU work after coordinator loss.
    parent = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError("Could not install parent-death signal")

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    started = time.monotonic()
    result = {
        "pid": os.getpid(),
        "device": request["device"],
        "operation": request["operation"],
    }
    exit_code = 0
    try:
        if parent == 1 or os.getppid() != parent:
            raise KeyboardInterrupt("Coordinator exited before worker startup")
        result.update(execute(request, destination))
        result["status"] = "completed"
    except BaseException as error:
        result.update(
            status="failed",
            failure=failure_kind(error),
            error=f"{type(error).__name__}: {error}",
        )
        traceback.print_exc()
        exit_code = 1
    finally:
        result["worker_seconds"] = time.monotonic() - started
        result["gpu_reserved_seconds"] = (
            result["worker_seconds"] if request["device"] is not None else 0.0
        )
        save_json(destination / "result.json", result)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
