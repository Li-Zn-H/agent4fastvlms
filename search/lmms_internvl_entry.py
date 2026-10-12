"""Native InternVL LMMS, with optional isolated quantized generation."""

import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_dense_config(config):
    if config.get("adapter") != "internvl35" or config.get("dtype") != "bfloat16":
        raise ValueError("Native InternVL Dense LMMS requires a registered BF16 InternVL descriptor")
    if (config.get("stack_adapter") or config.get("token_keep_ratio", 1) != 1
            or config.get("execution_backend", "native_transformers") != "native_transformers"):
        raise ValueError("InternVL LMMS entry currently supports Dense only")


def validate_native_config(config):
    if config.get("execution_backend") == "vllm":
        from skills.stack_runtime import validate_runtime_settings
        from skills.vision_token import validate_selection
        runtime = config.get("runtime", {})
        ratio = config.get("token_keep_ratio")
        if (config.get("adapter") != "internvl35" or config.get("dtype") != "bfloat16"
                or config.get("stack_adapter") is not True or type(ratio) not in (int, float)
                or not 0 < ratio < 1 or runtime.get("vllm_max_model_len", 0) < 4877):
            raise ValueError("InternVL serving LMMS requires its physical Stack, Token and native context envelope")
        validate_runtime_settings(runtime)
        try:
            validate_selection(config.get("token_selection", {}))
        except (KeyError, TypeError) as error:
            raise ValueError("InternVL serving LMMS requires the complete Token policy") from error
        return
    if not config.get("stack_adapter"):
        validate_dense_config(config)
        if config.get("native_worker_python"):
            raise ValueError("Native worker requires an explicit optimized LMMS descriptor")
        return
    from skills.vision_token import validate_selection

    if (config.get("adapter") != "internvl35" or config.get("dtype") != "bfloat16"
            or config.get("execution_backend", "native_transformers") != "native_transformers"
            or config.get("attn_implementation") not in {"eager", "sdpa"}):
        raise ValueError("Optimized native InternVL requires BF16 native Transformers with eager/SDPA")
    ratio = config.get("token_keep_ratio")
    if type(ratio) not in (int, float) or not 0 < ratio < 1:
        raise ValueError("Optimized native InternVL requires an explicit Token keep ratio in (0, 1)")
    try:
        validate_selection(config.get("token_selection", {}))
    except (KeyError, TypeError) as error:
        raise ValueError("Optimized native InternVL requires its complete Token selection policy") from error
    python = config.get("native_worker_python")
    if not isinstance(python, str) or not python or not (PROJECT_ROOT / python).is_file():
        raise ValueError("Optimized native InternVL requires its isolated native_worker_python")


def main():
    from lmms_eval.models.simple import internvl3
    from adapters.internvl35_dense import load_model

    config = json.loads(os.environ["A4FVLM_LMMS_INTERNVL_MODEL"])
    validate_native_config(config)

    def load(pretrained, **kwargs):
        if Path(pretrained).resolve() != (PROJECT_ROOT / config["path"]).resolve():
            raise ValueError("InternVL LMMS pretrained path differs from its descriptor")
        # Native LMMS owns device placement after CUDA_VISIBLE_DEVICES mapping.
        device = kwargs.get("device_map", config["device"])
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise ValueError("InternVL Dense LMMS requires a single explicit CUDA device")
        if config.get("stack_adapter"):
            from search.internvl_native_worker import create_native_chat_proxy
            return create_native_chat_proxy({**config, "device": device})
        model, _ = load_model({**config, "device": device})
        return model

    # Replace only this native wrapper's binding, not AutoModel globally.
    internvl3.AutoModel = SimpleNamespace(from_pretrained=load)
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
