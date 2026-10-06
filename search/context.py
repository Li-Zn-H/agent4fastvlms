"""JSON-only model selection and shared result paths; no inference imports."""

import json
from pathlib import Path

from adapters import AdapterError, hardware, models, select_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESULT_ROOT = "results/model/qwen3vl"


def checkpoint_config(model, *, project_root=PROJECT_ROOT):
    return json.loads((project_root / model["path"] / "config.json").read_text())


def model_adapter(model, *, project_root=PROJECT_ROOT):
    checkpoint = checkpoint_config(model, project_root=project_root)
    adapter = select_model(checkpoint)
    if "adapter" in model and models.get(model["adapter"]).id != adapter.id:
        raise AdapterError("model adapter does not match checkpoint model_type")
    return adapter


def hardware_adapter(model):
    # Old descriptors select CUDA through device. Explicit selection must agree.
    device_type = str(model.get("device", "cuda:0")).split(":", 1)[0]
    adapter = hardware.get(model.get("hardware", device_type))
    if device_type != adapter.worker_device.split(":", 1)[0]:
        raise AdapterError("hardware platform differs from the model device")
    return adapter


def result_root(evaluation, *, project_root=PROJECT_ROOT):
    value = evaluation.get("result_root", LEGACY_RESULT_ROOT)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("evaluation.result_root must be a nonempty path")
    path = (project_root / value).resolve()
    if path == project_root.resolve() or project_root.resolve().is_relative_to(path):
        raise AdapterError("result_root must be a dedicated model/experiment directory")
    return path


def search_root(evaluation, *, project_root=PROJECT_ROOT, legacy=None):
    # The optional legacy argument preserves old module-level path overrides.
    if "result_root" not in evaluation and legacy is not None:
        return Path(legacy)
    return result_root(evaluation, project_root=project_root) / "search"


def dense_root(evaluation, *, project_root=PROJECT_ROOT, legacy=None):
    name = evaluation["scoring"]["baseline"]
    if not isinstance(name, str) or not name or ("result_root" in evaluation and (
        name in {".", ".."} or Path(name).name != name
    )):
        raise AdapterError("scoring.baseline must name a directory within the model search root")
    return search_root(evaluation, project_root=project_root, legacy=legacy) / name


def validate_result_scope(model, evaluation, *, project_root=PROJECT_ROOT):
    """Legacy receipts may only be addressed by the original model descriptor."""
    adapter = model_adapter(model, project_root=project_root)
    path = result_root(evaluation, project_root=project_root)
    legacy = (project_root / LEGACY_RESULT_ROOT).resolve()
    original = (project_root / "models/Qwen3-VL-4B-Instruct").resolve()
    if ("result_root" not in evaluation or path == legacy or path.is_relative_to(legacy)) and (
        adapter.id != "qwen3_vl" or (project_root / model["path"]).resolve() != original
        or hardware_adapter(model).id != "cuda"
    ):
        raise AdapterError("a different model/platform requires its own evaluation.result_root")
    return path


def require_stages(model, stages, *, device=None, project_root=PROJECT_ROOT):
    from adapters.capabilities import require_capabilities

    return require_capabilities(model_adapter(model, project_root=project_root),
                                checkpoint_config(model, project_root=project_root),
                                hardware_adapter(model), device=device, stages=stages)


def validate_dense_reference(model, evaluation, *, project_root=PROJECT_ROOT):
    """New experiments need a bound denominator; preserve legacy receipt rules."""
    if "result_root" not in evaluation:
        return
    root = dense_root(evaluation, project_root=project_root)
    paths = (root / "config.json", root / "quality/config.json")
    found = False
    for path in paths:
        if not path.is_file():
            continue
        found = True
        saved = json.loads(path.read_text())
        if saved.get("evaluation") != evaluation or (model is not None and saved.get("model") != model):
            raise AdapterError(f"Dense reference config mismatch: {path}")
    if not found:
        raise AdapterError(f"Dense reference has no model/evaluation receipt: {root}")
