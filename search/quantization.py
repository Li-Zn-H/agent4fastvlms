"""Independent JSON-driven quantization framework candidate validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from search.model import PROJECT_ROOT
from search.optimize import hardware, launch, lock, read
from search.vllm_benchmark import compare_reports
from search.worker import save_json
from skills.quantization import candidate_defaults, digest, validate_config, validate_manifest, validate_stage_trace

_BENCHMARK_ONLY_RUNTIME_KEYS = {
    "vllm_benchmark",
    "vllm_benchmark_limit",
    "vllm_benchmark_warmup",
    "vllm_benchmark_repeats",
    "vllm_benchmark_batch_size",
}

# Deployment controls do not change exported weights, but MUST remain in the
# receipt identity. Only an authenticated export may use the narrower identity.
_DEPLOYMENT_ONLY_RUNTIME_KEYS = {
    "runtime_python",
    "clean_reload",
    "vllm_reload",
    "serve_timeout_seconds",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_len",
}

# A compact candidate receipt is reusable when the probe operation finished,
# including a deliberate preflight block. A coordinator/worker failure is
# resumable and must be retried on the next invocation.
_REUSABLE_RECEIPT_STATUSES = {"completed", "blocked"}
_NON_REUSABLE_PROBE_STATUSES = {
    "candidate_error",
    "worker_failed",
    "probe_error",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _candidate_digest(candidate: dict[str, Any]) -> str:
    """Keep benchmark controls from invalidating an existing quant artifact."""

    return digest(_candidate_digest_value(candidate))


def _candidate_digest_value(candidate: dict[str, Any]) -> dict[str, Any]:
    """Legacy receipt identity, including deployment controls."""

    value = dict(candidate)
    runtime = value.get("runtime_config")
    if isinstance(runtime, dict):
        value["runtime_config"] = {
            key: item
            for key, item in runtime.items()
            if key not in _BENCHMARK_ONLY_RUNTIME_KEYS
        }
    return value


def _quantization_identity(candidate: dict[str, Any]) -> str:
    """Export/Transformers-quality identity; never a deploy-runtime receipt."""

    value = _candidate_digest_value(candidate)
    runtime = value.get("runtime_config")
    if isinstance(runtime, dict):
        value["runtime_config"] = {
            key: item for key, item in runtime.items()
            if key not in _DEPLOYMENT_ONLY_RUNTIME_KEYS
        }
    return digest(value)


def _candidate(job: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    route = raw.get("route")
    result = {
        **candidate_defaults(route),
        **raw,
        "calibration": job.get("calibration", {}),
        "runtime_config": job.get("runtime", {}),
    }
    validate_config(result)
    return result


def _worker_python(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("worker_python")
    if not value:
        return None
    return str(_resolve(value))


def _runtime_python(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("runtime_config", {}).get("runtime_python")
    return str(_resolve(value)) if value else _worker_python(candidate)


def _module_available(name: str, worker_python: str | None) -> bool:
    if worker_python is None:
        return importlib.util.find_spec(name) is not None
    try:
        result = subprocess.run(
            [
                worker_python,
                "-c",
                "import importlib, sys; importlib.import_module(sys.argv[1])",
                name,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_MODULE_DISTRIBUTIONS = {
    "gptqmodel": "GPTQModel",
    "llmcompressor": "llmcompressor",
    "torchao": "torchao",
    "auto_round": "auto-round",
    "vllm": "vllm",
}


def _distribution_for_module(name: str) -> str:
    """Map an import name to its PyPI distribution for metadata checks."""

    return _MODULE_DISTRIBUTIONS.get(name, name)


def _dependency_preflight(
    candidate: dict[str, Any], worker_python: str | None, runtime_required: bool
) -> dict[str, Any]:
    """Check direct framework/runtime constraints inside the selected venv.

    ``pip check`` is intentionally not used as the gate: vLLM's CUDA wheels
    can report unrelated transitive protobuf constraints even when the engine
    runs correctly.  The useful signal here is a direct conflict between the
    selected framework and its configured runtime, such as LLM Compressor
    0.14.0 requiring compressed-tensors 0.19.0 while vLLM 0.26.0 pins 0.17.0.
    """

    roots = [_distribution_for_module(name) for name in candidate.get("imports", [])]
    if runtime_required:
        roots.extend(
            _distribution_for_module(name)
            for name in candidate.get("runtime_imports", [])
        )
    roots = list(dict.fromkeys(roots))
    if not roots:
        return {
            "versions": {},
            "constraints": {},
            "missing_distributions": [],
            "dependency_conflicts": [],
            "probe_error": None,
        }

    script = """
import importlib.metadata as metadata
import json
import sys

result = {}
for name in json.loads(sys.argv[1]):
    try:
        result[name] = {
            "version": metadata.version(name),
            "requires": metadata.requires(name) or [],
        }
    except metadata.PackageNotFoundError:
        result[name] = None
needed = set()
from packaging.requirements import Requirement
for row in result.values():
    if not isinstance(row, dict):
        continue
    for raw_requirement in row.get("requires", []):
        try:
            needed.add(Requirement(raw_requirement).name)
        except (TypeError, ValueError):
            continue
installed = {}
for name in sorted(needed):
    try:
        installed[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
result["__installed__"] = installed
print(json.dumps(result, sort_keys=True))
"""
    if worker_python is None:
        command = [sys.executable, "-c", script, json.dumps(roots)]
    else:
        command = [worker_python, "-c", script, json.dumps(roots)]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "versions": {},
            "constraints": {},
            "missing_distributions": [],
            "dependency_conflicts": [],
            "probe_error": f"{type(error).__name__}: {error}",
        }
    if result.returncode != 0:
        return {
            "versions": {},
            "constraints": {},
            "missing_distributions": [],
            "dependency_conflicts": [],
            "probe_error": (result.stderr or result.stdout or "metadata probe failed").strip()[-2000:],
        }
    try:
        metadata_rows = json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        return {
            "versions": {},
            "constraints": {},
            "missing_distributions": [],
            "dependency_conflicts": [],
            "probe_error": f"invalid metadata probe output: {error}",
        }

    installed_versions = metadata_rows.pop("__installed__", {})
    versions = {
        name: row["version"]
        for name, row in metadata_rows.items()
        if isinstance(row, dict) and row.get("version")
    }
    versions.update(installed_versions)
    missing_distributions = sorted(
        name for name, row in metadata_rows.items() if row is None
    )
    constraints: dict[str, list[dict[str, str]]] = {}
    for source, row in metadata_rows.items():
        if not isinstance(row, dict):
            continue
        for raw_requirement in row.get("requires", []):
            try:
                requirement = Requirement(raw_requirement)
            except (TypeError, ValueError):
                continue
            # Extras such as ``vllm[grpc]`` are not active in this worker.
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            name = requirement.name.lower().replace("_", "-")
            constraints.setdefault(name, []).append(
                {"source": source, "requirement": str(requirement)}
            )

    conflicts = []
    for dependency, requirements in sorted(constraints.items()):
        installed = next(
            (
                version
                for name, version in versions.items()
                if name.lower().replace("_", "-") == dependency
            ),
            None,
        )
        if installed is None:
            continue
        unsatisfied = []
        for row in requirements:
            try:
                requirement = Requirement(row["requirement"])
                if not requirement.specifier.contains(installed, prereleases=True):
                    unsatisfied.append(row)
            except (TypeError, ValueError):
                continue
        if unsatisfied:
            conflicts.append(
                {
                    "dependency": dependency,
                    "installed": installed,
                    "constraints": requirements,
                    "unsatisfied": unsatisfied,
                }
            )
    important_names = {
        name.lower().replace("_", "-")
        for name in (*roots, "compressed-tensors", "transformers", "tokenizers", "torch", "protobuf")
    }
    reported_versions = {
        name: version
        for name, version in versions.items()
        if name.lower().replace("_", "-") in important_names
    }
    return {
        "versions": reported_versions,
        "constraints": constraints,
        "missing_distributions": missing_distributions,
        "dependency_conflicts": conflicts,
        "probe_error": None,
    }


def snapshot(config_path: str | Path) -> dict[str, Any]:
    """Read JSON only; planning never imports a framework or loads a model."""

    job = read(_resolve(config_path))
    model_config_path = _resolve(job["model_config"])
    model = read(model_config_path)
    model_path = _resolve(model["path"])
    evaluation_config_path = _resolve(
        job.get("evaluation_config", "configs/evaluation.json")
    )
    evaluation = read(evaluation_config_path)
    from search.context import require_stages, validate_result_scope
    require_stages(model, ("quantization",), device=job["device"])
    model_root = validate_result_scope(model, evaluation)
    candidates = [_candidate(job, row) for row in job.get("candidates", [])]
    if not candidates:
        raise ValueError("quantization config must declare at least one candidate")
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("quantization candidate ids must be unique")
    return {
        "config_path": str(_resolve(config_path)),
        "model_config_path": str(model_config_path),
        "evaluation_config_path": str(evaluation_config_path),
        "evaluation": evaluation,
        "model": model,
        "model_path": str(model_path),
        "device": job.get("device"),
        "output": str(_resolve(job.get("output", model_root / "quantization"))),
        "candidates": candidates,
    }


def _preflight(snap: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    imports = candidate["imports"]
    worker_python = _worker_python(candidate)
    runtime_python = _runtime_python(candidate)
    missing_environment = []
    if worker_python and not Path(worker_python).is_file():
        missing_environment.append(worker_python)
    if runtime_python and runtime_python != worker_python and not Path(runtime_python).is_file():
        missing_environment.append(runtime_python)
    missing = [
        name
        for name in imports
        if not missing_environment and not _module_available(name, worker_python)
    ]
    runtime_imports = candidate.get("runtime_imports", [])
    missing_runtime = [
        name
        for name in runtime_imports
        if not missing_environment and not _module_available(name, runtime_python)
    ]
    calibration = candidate.get("calibration", {})
    calibration_path = (
        str(_resolve(calibration["data"])) if calibration.get("data") else None
    )
    requirement_file = candidate.get("requirements")
    missing_requirements = []
    if requirement_file:
        requirement_path = _resolve(requirement_file)
        if not requirement_path.is_file():
            missing_requirements.append(str(requirement_path))
    missing_files = []
    if not Path(snap["model_path"]).exists():
        missing_files.append(snap["model_path"])
    if candidate_defaults(candidate["route"])["calibration_required"] and (
        not calibration_path or not Path(calibration_path).exists()
    ):
        missing_files.append(calibration_path or "calibration.data")
    runtime_required = bool(
        candidate.get("runtime_config", {}).get("quantize", False)
        or candidate.get("runtime_config", {}).get("vllm_reload", False)
    )
    split_runtime = runtime_python != worker_python
    dependency_check = _dependency_preflight(candidate, worker_python, runtime_required and not split_runtime)
    runtime_dependency_check = (_dependency_preflight(
        {"imports": runtime_imports}, runtime_python, False
    ) if split_runtime and runtime_required else None)
    dependency_blocked = bool(
        dependency_check["probe_error"]
        or (runtime_required and dependency_check["dependency_conflicts"])
        or (runtime_dependency_check and (runtime_dependency_check["probe_error"]
                                         or runtime_dependency_check["dependency_conflicts"]))
    )
    architecture_check = None
    artifact_audit = None
    if snap["model"].get("adapter") in {"llava_onevision", "llava15"} and candidate["route"] == "gptqmodel_w4a16":
        from skills.quantization_evidence import audit_gptq_export, llava_gptq_preflight, llava15_gptq_preflight

        if snap["model"].get("adapter") == "llava15":
            architecture_check = llava15_gptq_preflight(
                read(Path(snap["model_path"]) / "config.json"), candidate.get("group_size"))
        else:
            architecture_check = llava_gptq_preflight(
                read(Path(snap["model_path"]) / "config.json"), candidate.get("group_size"),
                padding=candidate.get("vision_quantization") == "siglip_gptq_zero_pad_v1",
            )
        artifact = _candidate_dir(Path(snap["output"]), candidate["id"]) / "retry-artifact"
        if (artifact / "manifest.json").is_file():
            artifact_audit = audit_gptq_export(artifact, read(artifact / "manifest.json"))
    implementation_blocked = any(
        check and check["status"] == "blocked_preflight"
        for check in (architecture_check, artifact_audit)
    )
    return {
        "candidate": candidate["id"],
        "route": candidate["route"],
        "framework": candidate["framework"],
        "worker_python": worker_python,
        "runtime_python": runtime_python,
        "runtime_dependency_check": runtime_dependency_check,
        "missing_environment": missing_environment,
        "runtime": candidate["runtime"],
        "missing_imports": missing,
        "missing_runtime_imports": missing_runtime,
        "runtime_required": runtime_required,
        "requirements": requirement_file,
        "missing_requirements": missing_requirements,
        "environment_versions": dependency_check["versions"],
        "dependency_conflicts": dependency_check["dependency_conflicts"],
        "missing_distributions": dependency_check["missing_distributions"],
        "environment_probe_error": dependency_check["probe_error"],
        "missing_files": missing_files,
        "architecture_check": architecture_check,
        "exported_precision_audit": artifact_audit,
        "implementation_blocked": implementation_blocked,
        "ready_for_worker": not missing
        and not missing_files
        and not missing_requirements
        and not missing_environment
        and not dependency_blocked
        and not implementation_blocked
        and (not runtime_required or not missing_runtime),
    }


def _candidate_dir(destination: Path, candidate_id: str) -> Path:
    return destination / "candidates" / candidate_id


def _prepare_candidate_dir(destination: Path, candidate_id: str) -> Path:
    """Create a candidate directory while preserving receipts and raw evidence."""

    candidate_dir = _candidate_dir(destination, candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for path in candidate_dir.iterdir():
        if path.name in {"probe.json", "result.json", "retry-artifact", "evidence"}:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return candidate_dir


def _retain_attempt_evidence(destination, candidate_id, attempt_dir):
    """Preserve raw answers/timings and diagnostics before temporary cleanup."""
    evidence = _candidate_dir(destination, candidate_id) / "evidence" / attempt_dir.name
    for name in ("request.json", "worker.log", "probe.json", "result.json", "quality", "benchmark", "evidence"):
        source = attempt_dir / name
        if not source.exists():
            continue
        evidence.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, evidence / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, evidence / name)
    # A rejected or failed export need not retain a second checkpoint, but its
    # precision/calibration records must remain available with the attempt.
    for name in ("config.json", "manifest.json", "precision-audit.json", "vision-quantization.json"):
        source = attempt_dir / "artifact" / name
        if source.is_file():
            (evidence / "artifact-metadata").mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, evidence / "artifact-metadata" / name)


def _existing_receipt(
    destination: Path, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a valid durable receipt that can be reused during resume."""

    candidate_id = candidate["id"]
    candidate_dir = _candidate_dir(destination, candidate_id)
    result_path = candidate_dir / "result.json"
    probe_path = candidate_dir / "probe.json"
    if not result_path.is_file() or not probe_path.is_file():
        return None
    try:
        result = read(result_path)
        probe = read(probe_path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(result, dict) or not isinstance(probe, dict):
        return None
    if result.get("status") not in _REUSABLE_RECEIPT_STATUSES:
        return None
    # ``status=completed`` describes the Worker process, not the candidate.
    # A completed Worker can still have returned ``probe_status=candidate_error``
    # and must be retried after an adapter fix.
    probe_status = result.get("probe_status", result.get("status"))
    if probe_status in _NON_REUSABLE_PROBE_STATUSES:
        return None
    if result.get("candidate_digest") != _candidate_digest(candidate):
        return None
    return {
        "status": probe_status,
        "probe": str(probe_path),
        "result": str(result_path),
        "preflight": probe.get("preflight"),
    }


def _retry_artifact(
    destination: Path, candidate: dict[str, Any], saved: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Authenticate a late-stage export against the PREVIOUS saved snapshot."""

    candidate_dir = _candidate_dir(destination, candidate["id"])
    artifact = candidate_dir / "retry-artifact"
    probe_path = candidate_dir / "probe.json"
    if not artifact.is_dir() or not probe_path.is_file():
        return None
    try:
        probe = read(probe_path)
        saved_candidate = next(row for row in saved.get("candidates", [])
                               if row.get("id") == candidate.get("id"))
        manifest = read(artifact / "manifest.json")
        validate_manifest(manifest)
        validate_stage_trace(probe.get("stages", {}))
    except (OSError, ValueError, TypeError, StopIteration, AttributeError):
        return None
    if not isinstance(probe, dict) or probe.get("status") != "candidate_error":
        return None
    # Recognize both historical digest encodings ONLY for an errored export;
    # neither grants reuse of old vLLM results after settings change.
    if probe.get("candidate_digest") not in {
        _candidate_digest(saved_candidate), _quantization_identity(saved_candidate)
    } or _quantization_identity(saved_candidate) != _quantization_identity(candidate):
        return None
    if (probe.get("candidate") != candidate["id"]
            or probe.get("route") != candidate["route"]
            or manifest.get("status") != "exported"
            or manifest.get("route") != candidate["route"]
            or manifest.get("model") != {**saved["model"], "device": "cuda:0"}
            or manifest.get("model_digest") != digest(manifest["model"])
            or manifest.get("calibration") != candidate.get("calibration")
            or manifest.get("calibration_digest") != digest(candidate.get("calibration"))
            or manifest.get("module_assignment_digest") != probe["stages"]["prepare"].get("assignment_digest")):
        return None
    quant = manifest.get("quantization", {})
    for manifest_key, candidate_key in (("bits", "bits"), ("group_size", "group_size"),
                                        ("weight_activation", "scheme"), ("mapper_precision", "mapper_precision")):
        if quant.get(manifest_key) != candidate.get(candidate_key):
            return None
    if probe.get("quality", {}).get("status") != "passed":
        return None
    if probe.get("performance", {}).get("status") != "passed":
        return None
    return artifact, probe


def _save_blocked_candidate(
    destination: Path, candidate: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    """Persist only a compact preflight report for a blocked candidate."""

    candidate_dir = _prepare_candidate_dir(destination, candidate["id"])
    # A new preflight finding must not erase the previous execution failure or
    # its quality/performance evidence, even when the JSON itself is unchanged.
    old_probe = candidate_dir / "probe.json"
    if old_probe.is_file():
        prior = read(old_probe)
        if prior.get("status") != "blocked_preflight":
            archive = destination / "history" / ("probe-" + digest(prior))
            archive.mkdir(parents=True, exist_ok=True)
            for filename in ("probe.json", "result.json"):
                source = candidate_dir / filename
                if source.is_file():
                    shutil.copy2(source, archive / filename)
    probe = {
        "schema": "a4fvlm.quantization-probe.v1",
        "candidate": candidate["id"],
        "route": candidate["route"],
        "framework": candidate["framework"],
        "candidate_digest": _candidate_digest(candidate),
        "status": "blocked_preflight",
        "preflight": preflight,
    }
    result = {
        "status": "blocked",
        "probe_status": "blocked_preflight",
        "candidate": candidate["id"],
        "candidate_digest": _candidate_digest(candidate),
    }
    save_json(candidate_dir / "probe.json", probe)
    save_json(candidate_dir / "result.json", result)
    return {
        "status": "blocked",
        "probe": str(candidate_dir / "probe.json"),
        "result": str(candidate_dir / "result.json"),
    }


def _retain_probe_result(
    destination: Path,
    candidate: dict[str, Any],
    attempt_dir: Path,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    """Copy only probe/receipt evidence out of a temporary Worker attempt."""

    candidate_dir = _prepare_candidate_dir(destination, candidate["id"])
    probe_source = attempt_dir / "probe.json"
    probe = None
    if probe_source.exists():
        probe = read(probe_source)
        if isinstance(probe, dict) and "artifact" in probe:
            # The Worker path is temporary and is deleted below. The only
            # durable artifact path is created if this candidate is selected.
            probe["artifact"] = "final/artifact"
        if isinstance(probe, dict):
            probe["candidate_digest"] = _candidate_digest(candidate)
        save_json(candidate_dir / "probe.json", probe)
    else:
        save_json(
            candidate_dir / "probe.json",
            {
                "schema": "a4fvlm.quantization-probe.v1",
                "candidate": candidate["id"],
                "route": candidate["route"],
                "status": "missing_probe",
            },
        )
    receipt = dict(worker_result)
    receipt.pop("artifact", None)
    receipt["candidate"] = candidate["id"]
    receipt["candidate_digest"] = _candidate_digest(candidate)
    if (
        receipt.get("probe_status") == "candidate_error"
        and isinstance(probe, dict)
        and probe.get("quality", {}).get("status") == "passed"
        and probe.get("performance", {}).get("status") == "passed"
        and (attempt_dir / "artifact").is_dir()
    ):
        # Keep a resumable late-stage artifact; it is removed after final
        # selection and never published as the final artifact on failure.
        shutil.copytree(
            attempt_dir / "artifact",
            candidate_dir / "retry-artifact",
            dirs_exist_ok=True,
        )
    if receipt.get("report"):
        receipt["report"] = str(candidate_dir / "probe.json")
    save_json(candidate_dir / "result.json", receipt)
    return {
        "status": receipt.get("probe_status", receipt.get("status", "unknown")),
        "probe": str(candidate_dir / "probe.json"),
        "result": str(candidate_dir / "result.json"),
    }


def _stage_qualified_artifact(
    destination: Path,
    candidate: dict[str, Any],
    attempt_dir: Path,
    worker_result: dict[str, Any],
) -> Path | None:
    """Hold a qualified artifact until all candidates have been compared."""

    if worker_result.get("probe_status") != "qualified":
        return None
    raw_source = worker_result.get("artifact")
    if not raw_source:
        return None
    source = Path(raw_source)
    if not source.is_dir():
        return None
    staging = destination / ".selection" / candidate["id"]
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging / "artifact", dirs_exist_ok=True)
    return staging


def _select_final_artifact(destination: Path, staged: dict[str, Path]) -> str | None:
    """Select one qualified route, retaining only its final artifact."""

    if not staged:
        return None
    scored = []
    for candidate_id, staging in staged.items():
        try:
            probe = read(_candidate_dir(destination, candidate_id) / "probe.json")
            quality = probe.get("quality", {})
            performance = probe.get("performance", {})
            dev = quality.get("dev_retention")
            mean = performance.get("mean_round_seconds")
            if not isinstance(dev, (int, float)) or not isinstance(mean, (int, float)):
                continue
            scored.append((float(dev), float(mean), candidate_id, staging))
        except (OSError, ValueError, TypeError):
            continue
    if not scored:
        return None
    _, _, candidate_id, staging = max(scored, key=lambda row: (row[0], -row[1]))
    final = destination / "final"
    artifact = final / "artifact"
    if artifact.exists():
        shutil.rmtree(artifact, ignore_errors=True)
    shutil.copytree(staging / "artifact", artifact)
    manifest = artifact / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Selected artifact for {candidate_id} has no manifest")
    return candidate_id


def _run_vllm_comparison(
    snap: dict[str, Any], destination: Path, candidate: dict[str, Any]
) -> dict[str, Any]:
    """Run/reuse clean-process Dense versus selected-artifact vLLM timing."""

    output_path = destination / "vllm-benchmark.json"
    candidate_digest = _candidate_digest(candidate)
    if output_path.is_file():
        try:
            existing = read(output_path)
            if (
                existing.get("status") == "passed"
                and existing.get("schema") == "a4fvlm.vllm-comparison.v3"
                and existing.get("candidate_digest") == candidate_digest
            ):
                return existing
        except (OSError, ValueError, TypeError):
            pass
    work_root = destination / ".vllm-work"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    runtime = candidate.get("runtime_config", {})
    worker_python = _runtime_python(candidate)
    models = {
        "dense": str(_resolve(snap["model"]["path"])),
        "quantized": str(destination / "final" / "artifact"),
    }
    reports: dict[str, dict[str, Any]] = {}
    error = None
    try:
        for role, model_path in models.items():
            attempt_dir = work_root / role
            request = {
                "operation": "quantization_vllm_benchmark",
                "device": snap["device"],
                "hardware": hardware(snap["device"], snap["model"]),
                "worker_python": worker_python,
                "model": {**snap["model"], "path": model_path, "device": "cuda:0"},
                "evaluation": snap["evaluation"],
                "candidate": candidate,
                "runtime": runtime,
                "model_role": role,
                "probe_output": str(attempt_dir),
            }
            result = launch(request, attempt_dir)
            report_path = attempt_dir / "benchmark.json"
            if not report_path.is_file():
                raise RuntimeError(f"vLLM benchmark report is missing for {role}")
            report = read(report_path)
            if result.get("probe_status") != "vllm_benchmark_passed":
                raise RuntimeError(report.get("error", f"vLLM benchmark failed for {role}"))
            reports[role] = report
    except Exception as exc:  # noqa: BLE001 - persist a compact failed receipt
        error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    if error is not None:
        receipt = {
            "schema": "a4fvlm.vllm-comparison.v3",
            "candidate": candidate["id"],
            "candidate_digest": candidate_digest,
            "status": "failed",
            "error": error,
        }
    else:
        receipt = {
            "schema": "a4fvlm.vllm-comparison.v3",
            "candidate": candidate["id"],
            "candidate_digest": candidate_digest,
            "status": "passed",
            "dense": reports["dense"],
            "quantized": reports["quantized"],
            "comparison": compare_reports(reports["dense"], reports["quantized"]),
        }
    save_json(output_path, receipt)
    return receipt


def _normalize_selected_probe(destination: Path, candidate_id: str) -> None:
    """Replace temporary Worker paths with the durable final artifact path."""

    probe_path = _candidate_dir(destination, candidate_id) / "probe.json"
    try:
        probe = read(probe_path)
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(probe, dict):
        return
    if "artifact" in probe:
        probe["artifact"] = "final/artifact"
    stages = probe.get("stages")
    if isinstance(stages, dict):
        for stage_name in ("export", "manifest"):
            stage = stages.get(stage_name)
            if isinstance(stage, dict) and "manifest" in stage:
                stage["manifest"] = "final/artifact/manifest.json"
            if isinstance(stage, dict) and "path" in stage:
                stage["path"] = "final/artifact/manifest.json"
    save_json(probe_path, probe)


def plan(config_path: str | Path) -> dict[str, Any]:
    snap = snapshot(config_path)
    rows = []
    for candidate in snap["candidates"]:
        preflight = _preflight(snap, candidate)
        rows.append(
            {
                **preflight,
                "enabled": candidate.get("enabled", True),
                "action": (
                    "run_quantization_candidate"
                    if candidate.get("runtime_config", {}).get("quantize", False)
                    else "run_framework_probe"
                )
                if candidate.get("enabled", True) and preflight["ready_for_worker"]
                else "blocked_or_disabled",
            }
        )
    return {
        "model": snap["model_path"],
        "device": snap["device"],
        "candidates": rows,
        "independent_from_structured_pruning": True,
        "output": snap["output"],
    }


def execute(config_path: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    snap = snapshot(config_path)
    destination = _resolve(output or snap["output"])
    destination.mkdir(parents=True, exist_ok=True)
    with lock(destination / ".run.lock"):
        return _execute_snapshot(snap, destination)


def _execute_snapshot(snap: dict[str, Any], destination: Path) -> dict[str, Any]:
    saved_path = destination / "config.json"
    saved = read(saved_path) if saved_path.is_file() else {}
    if saved:
        mismatches = [key for key in ("model", "evaluation", "device", "model_path")
                      if saved.get(key) != snap.get(key)]
        if mismatches:
            raise ValueError("Saved quantization context mismatch: " + ", ".join(mismatches))
    retries = {candidate["id"]: _retry_artifact(destination, candidate, saved)
               for candidate in snap["candidates"]}
    if saved and saved != snap:
        # Keep the evidence used to authorize reuse, including failures. Do not
        # replace the only copy of the snapshot that authenticates the export.
        archive = destination / "history" / digest(saved)
        if not archive.exists():
            archive.mkdir(parents=True)
            save_json(archive / "config.json", saved)
            for candidate in saved.get("candidates", []):
                for filename in ("probe.json", "result.json"):
                    source = _candidate_dir(destination, candidate["id"]) / filename
                    if source.is_file():
                        target = archive / "candidates" / candidate["id"] / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
    save_json(destination / "config.json", snap)
    state_path = destination / "state.json"
    state = (
        read(state_path)
        if state_path.exists()
        else {"status": "running", "attempts": [], "candidates": {}}
    )
    state.setdefault("attempts", [])
    state.setdefault("candidates", {})
    if not isinstance(state["attempts"], list):
        state["attempts"] = []
    if not isinstance(state["candidates"], dict):
        state["candidates"] = {}
    state["status"] = "running"
    state["config"] = snap["config_path"]
    state["active_candidate"] = None
    state["active_attempt"] = None
    save_json(state_path, state)
    work_root = destination / ".work"
    # A coordinator can be terminated while its Worker is running. Workers
    # receive a parent-death signal, so any leftover work directory is stale
    # and safe to remove before resuming.
    shutil.rmtree(work_root, ignore_errors=True)
    staged: dict[str, Path] = {}
    for candidate in snap["candidates"]:
        candidate_id = candidate["id"]
        if not candidate.get("enabled", True):
            state["candidates"][candidate_id] = {"status": "disabled"}
            continue
        preflight = _preflight(snap, candidate)
        retained = _existing_receipt(destination, candidate)
        stale_block = (
            retained is not None
            and retained.get("status") == "blocked_preflight"
            and (
                preflight["ready_for_worker"]
                or retained.get("preflight") != preflight
            )
        )
        if retained is not None and not stale_block and not preflight.get("implementation_blocked", False):
            state["candidates"][candidate_id] = {
                key: value for key, value in retained.items() if key != "preflight"
            }
            save_json(state_path, state)
            continue
        if not preflight["ready_for_worker"]:
            state["candidates"][candidate_id] = _save_blocked_candidate(
                destination, candidate, preflight
            )
            save_json(state_path, state)
            continue
        # A quantization attempt can finish the expensive export, reload,
        # quality, and benchmark stages before a deploy-runtime check fails.
        # Keep that artifact resumable and retry only the late runtime gate
        # after the environment is repaired.  This avoids repeating hours of
        # calibration and evaluation on the next coordinator invocation.
        retry = retries[candidate_id]
        attempt = len(state["attempts"]) + 1
        attempt_dir = work_root / f"candidate-{candidate_id}" / f"attempt-{attempt}"
        request = {
            "operation": (
                "quantization_runtime_validation"
                if retry is not None
                else (
                    "quantization_candidate"
                    if candidate.get("runtime_config", {}).get("quantize", False)
                    else "quantization_framework_probe"
                )
            ),
            "device": snap["device"],
            "hardware": hardware(snap["device"], snap["model"]),
            "worker_python": _worker_python(candidate),
            "model": {**snap["model"], "device": "cuda:0"},
            "evaluation": snap["evaluation"],
            "candidate": candidate,
            "runtime": candidate.get("runtime_config", {}),
            "calibration": candidate.get("calibration", {}),
            "probe_output": str(attempt_dir),
        }
        if retry is not None:
            retry_path, prior_probe = retry
            request.update(
                {
                    "artifact": str(retry_path),
                    "prior_probe": prior_probe,
                    "reuse_provenance": {
                        "source_snapshot_digest": digest(saved),
                        "source_candidate_digest": prior_probe["candidate_digest"],
                        "export_identity": _quantization_identity(candidate),
                        "runtime_config": candidate.get("runtime_config", {}),
                    },
                }
            )
        state["active_candidate"] = candidate_id
        state["active_attempt"] = str(attempt_dir)
        save_json(state_path, state)
        try:
            result = launch(request, attempt_dir)
            staged_artifact = _stage_qualified_artifact(
                destination, candidate, attempt_dir, result
            )
            if staged_artifact is not None:
                staged[candidate_id] = staged_artifact
            retained = _retain_probe_result(
                destination, candidate, attempt_dir, result
            )
            state["candidates"][candidate_id] = retained
            state["attempts"].append(
                {"candidate": candidate_id, "status": retained["status"]}
            )
        except Exception as error:  # noqa: BLE001 - retryable Worker boundary
            candidate_dir = _prepare_candidate_dir(destination, candidate_id)
            failure = {
                "status": "worker_failed",
                "candidate": candidate_id,
                "candidate_digest": _candidate_digest(candidate),
                "error": f"{type(error).__name__}: {error}",
            }
            save_json(candidate_dir / "result.json", failure)
            save_json(
                candidate_dir / "probe.json",
                {
                    "schema": "a4fvlm.quantization-probe.v1",
                    "candidate": candidate_id,
                    "route": candidate["route"],
                    "candidate_digest": _candidate_digest(candidate),
                    "status": "worker_failed",
                },
            )
            state["candidates"][candidate_id] = {
                "status": "worker_failed",
                "probe": str(candidate_dir / "probe.json"),
                "result": str(candidate_dir / "result.json"),
            }
            state["attempts"].append(
                {"candidate": candidate_id, "status": "worker_failed"}
            )
        finally:
            _retain_attempt_evidence(destination, candidate_id, attempt_dir)
            shutil.rmtree(attempt_dir, ignore_errors=True)
            # The temporary root is never part of the retained result layout.
            shutil.rmtree(work_root, ignore_errors=True)
        state["active_candidate"] = None
        state["active_attempt"] = None
        save_json(state_path, state)
    state["status"] = "completed"
    selected_candidate = _select_final_artifact(destination, staged)
    if selected_candidate is None:
        previous = state.get("selected_candidate")
        selected_candidate = next((row["id"] for row in snap["candidates"]
                                   if row["id"] == previous
                                   and (state["candidates"].get(previous) or {}).get("status") == "qualified"
                                   and _existing_receipt(destination, row)), None)
    if selected_candidate is not None:
        _normalize_selected_probe(destination, selected_candidate)
    shutil.rmtree(destination / ".selection", ignore_errors=True)
    if selected_candidate is not None:
        # Once the selected artifact has been copied to final/artifact, the
        # resumable copies are no longer needed and would violate the compact
        # durable-results layout.
        for candidate in snap["candidates"]:
            shutil.rmtree(
                _candidate_dir(destination, candidate["id"]) / "retry-artifact",
                ignore_errors=True,
            )
    if selected_candidate is not None:
        selected = next(
            row for row in snap["candidates"] if row["id"] == selected_candidate
        )
        if selected.get("runtime_config", {}).get("vllm_benchmark", False):
            comparison = _run_vllm_comparison(destination=destination, snap=snap, candidate=selected)
        else:
            comparison = None
    else:
        comparison = None
    state["selected_candidate"] = selected_candidate
    save_json(state_path, state)
    summary = {
        "status": state["status"],
        "candidate_count": len(snap["candidates"]),
        "results": state["candidates"],
        "checkpoint": str(state_path),
        "independent_from_structured_pruning": True,
        "selected_candidate": selected_candidate,
    }
    if comparison is not None:
        summary["vllm_benchmark"] = {
            "status": comparison.get("status"),
            "path": str(destination / "vllm-benchmark.json"),
            "comparison": comparison.get("comparison"),
        }
    save_json(destination / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quantization.json")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = execute(args.config, output=args.output) if args.execute else plan(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
