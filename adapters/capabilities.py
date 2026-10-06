"""Static stage contracts. Environment/kernel/quality evidence remains stage-owned."""

from adapters.base import AdapterError


STAGE_OPERATIONS = {
    "dense": ("dense_load", "processor_load", "prepare_inputs", "generation_timer", "image_token_count"),
    "token": ("token_controller", "prepare_inputs", "token_geometry"),
    "sp": ("sp_apply",),
    "quantization": ("dense_load", "quantization_export"),
    "stack": ("stack_export", "stack_load", "token_controller"),
    "runtime": ("vllm_stack",),
    "kv_cache": ("vllm_stack",),
    "lmms": ("lmms_native", "lmms_vllm"),
}


def check_capabilities(model, checkpoint, platform, *, device=None, stages=None):
    """Return JSON without importing declared entry points or touching hardware."""
    requested = tuple(STAGE_OPERATIONS) if stages is None else tuple(stages)
    rows = {}
    for stage in requested:
        try:
            if stage not in STAGE_OPERATIONS:
                raise AdapterError(f"unknown stage: {stage}")
            model.validate_stage(stage, checkpoint)
            if platform.id not in model.hardware_ids or stage not in platform.stages:
                raise AdapterError(f"{model.id}/{platform.id} has no implemented {stage} path")
            if device is not None:
                platform.validate_device(device)
            missing = [name for name in STAGE_OPERATIONS[stage] if name not in model.entrypoints]
            hardware_ops = ("probe_identity", "launch_worker") + (
                ("vllm_runtime_kwargs",) if stage in {"runtime", "kv_cache", "lmms"} else ()
            )
            missing += [name for name in hardware_ops if name not in platform.entrypoints]
            if missing:
                raise AdapterError(f"{stage} missing entry points: {', '.join(missing)}")
            rows[stage] = {"status": "ready_for_probe", "implementation": "available",
                           "environment": "unverified", "quality": "unverified"}
        except AdapterError as error:
            rows[stage] = {"status": "blocked_preflight", "reason": str(error)}
    return {"model_adapter": model.id, "hardware": platform.id,
            "status": "blocked_preflight" if any(r["status"] == "blocked_preflight" for r in rows.values()) else "ready_for_probe",
            "stages": rows,
            "qualification": "Static implementation checks only; backend environments, kernels and quality need matching stage receipts."}


def require_capabilities(model, checkpoint, platform, *, device=None, stages=None):
    report = check_capabilities(model, checkpoint, platform, device=device, stages=stages)
    if report["status"] == "blocked_preflight":
        raise AdapterError("; ".join(f"{stage}: {row['reason']}" for stage, row in report["stages"].items()
                                     if row["status"] == "blocked_preflight"))
    return report
