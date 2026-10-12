"""Register the native DeepSeek-VL2 model for the unchanged LMMS task CLI."""

import json
import os
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_native_config(config):
    if (config.get("adapter") != "deepseek_vl2" or config.get("dtype") != "bfloat16"
            or config.get("attn_implementation") != "eager"
            or config.get("execution_backend", "native_transformers") not in {"native_transformers", "vllm"}):
        raise ValueError("DeepSeek-VL2 native LMMS requires BF16 native Transformers with eager MLA")
    if not config.get("stack_adapter"):
        if (config.get("token_keep_ratio", 1) != 1 or config.get("native_worker_python")
                or config.get("execution_backend", "native_transformers") != "native_transformers"):
            raise ValueError("DeepSeek-VL2 Dense LMMS requires its original unmodified descriptor")
        return
    from skills.vision_token import validate_selection
    from skills.deepseek_vl2_artifact import validate_quantized_bindings
    ratio = config.get("token_keep_ratio")
    if config["stack_adapter"] is not True or type(ratio) not in (int, float) or not 0 < ratio < 1:
        raise ValueError("Optimized DeepSeek LMMS requires an explicit Token keep ratio in (0, 1)")
    try:
        validate_selection(config.get("token_selection", {}))
    except (KeyError, TypeError) as error:
        raise ValueError("Optimized DeepSeek LMMS requires its complete Token selection policy") from error
    if config.get("execution_backend") == "vllm":
        from skills.stack_runtime import validate_runtime_settings
        runtime = config.get("runtime", {})
        if runtime.get("vllm_max_model_len") != 4096 or runtime.get("cudagraph_mm_encoder", False):
            raise ValueError("DeepSeek serving Runtime requires native context 4096 and no encoder CUDA Graph")
        validate_runtime_settings(runtime)
    else:
        python = config.get("native_worker_python")
        if not isinstance(python, str) or not python or not (PROJECT_ROOT / python).is_file():
            raise ValueError("Optimized DeepSeek LMMS requires its isolated native_worker_python")
    path = PROJECT_ROOT / config["path"]
    header = json.loads((path / "config.json").read_text())
    if header.get("a4fvlm_stack_widths") is None or header.get("a4fvlm_deepseek_gptq") is None:
        raise ValueError("Optimized DeepSeek LMMS requires a freshly quantized physical Parent")
    validate_quantized_bindings(path, header, json.loads((path / "manifest.json").read_text()))


def main():
    from lmms_eval.models import MODEL_REGISTRY_V2, ModelManifest
    config = json.loads(os.environ["A4FVLM_LMMS_DEEPSEEK_MODEL"])
    validate_native_config(config)
    MODEL_REGISTRY_V2.register_manifest(ModelManifest(
        model_id="deepseek_vl2", simple_class_path="search.deepseek_lmms.DeepSeekVL2LMMS"))
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
