"""Run native LMMS with a physical LLaVA Stack parent.

Current lmms-eval registers the Hugging Face LLaVA wrapper as ``llava_hf``;
the launcher patches the simple base initializer inherited by the chat wrapper, replaces its model class
with the physical Stack class and routes generation through the shared
post-encoder Token controller.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_GENERATION_KEYS = frozenset({
    "eos_token_id", "pad_token_id", "max_new_tokens", "use_cache",
    "do_sample", "temperature", "top_p", "top_k", "num_beams",
})


def restore_native_prefix(result, native_ids, compacted_length):
    """LMMS trims the original prompt length; return its unchanged prefix."""
    import torch

    return torch.cat((native_ids, result[:, compacted_length:]), dim=1)


def main() -> None:
    from lmms_eval.models.simple import llava_hf as simple
    from skills.stack_model import create_pruned_model_class
    from adapters.llava_onevision_dense import LlavaVisionTokenController, quantized_siglip_initialization
    from adapters.llava15_dense import Llava15VisionTokenController, quantized_clip_initialization

    simple.model_map["llava_onevision"] = create_pruned_model_class("llava_onevision")
    simple.model_map["llava"] = create_pruned_model_class("llava15")
    original_init = simple.LlavaHf.__init__

    def init_with_token(self, *args, **kwargs):
        pretrained = kwargs.get("pretrained", args[0] if args else None)
        if pretrained is None:
            raise ValueError("Stack LMMS requires an explicit pretrained artifact")
        model_type = json.loads((Path(pretrained) / "config.json").read_text())["model_type"]
        if model_type not in {"llava", "llava_onevision"}:
            raise ValueError("Unsupported native LLaVA Stack model type")
        llava15 = model_type == "llava"
        initialization = quantized_clip_initialization if llava15 else quantized_siglip_initialization
        with initialization(pretrained):
            original_init(self, *args, **kwargs)
        ratio_text = os.environ.get("A4FVLMSP_TOKEN_KEEP_RATIO")
        if ratio_text is None:
            return
        ratio = float(ratio_text)
        selection_text = os.environ.get("A4FVLMSP_TOKEN_SELECTION")
        selection = json.loads(selection_text) if selection_text else json.loads(
            (PROJECT_ROOT / ("configs/vision_token-llava15.json" if llava15 else "configs/vision_token-llava-onevision.json")).read_text()
        )["selection"]
        controller_type = Llava15VisionTokenController if llava15 else LlavaVisionTokenController
        controller = controller_type(self.model, selection, ratio)
        original_generate = self.model.generate
        receipt_text = os.environ.get("A4FVLMSP_TOKEN_RECEIPT")
        delegating = False

        def routed_generate(*args, **kwargs):
            nonlocal delegating
            if delegating:
                return original_generate(*args, **kwargs)
            if args:
                raise ValueError("LLaVA Stack Token bridge requires keyword model inputs")
            inputs = {key: value for key, value in kwargs.items() if key not in _GENERATION_KEYS}
            options = {key: value for key, value in kwargs.items() if key in _GENERATION_KEYS}
            native_ids = inputs["input_ids"]
            delegating = True
            try:
                result = controller.generate(inputs, options)
                metrics = controller.last_metrics
                if ratio < 1:
                    if not metrics:
                        raise RuntimeError("LLaVA Stack Token controller produced no evidence")
                    if receipt_text:
                        with Path(receipt_text).open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "method": "transformers_score_based_post_encoder",
                                "token_keep_ratio": ratio,
                                "image_tokens_before": metrics["image_tokens_before"],
                                "image_tokens_after": metrics["image_tokens_after"],
                                "decoder_tokens_after": metrics["decoder_tokens_after"],
                                "decode_steps": metrics["decode_steps"],
                                "kv_tokens_after_last_forward": metrics["kv_tokens_after_last_forward"],
                            }) + "\n")
                    result = restore_native_prefix(result, native_ids, metrics["decoder_tokens_after"])
                return result
            finally:
                delegating = False

        self.model.generate = routed_generate

    simple.LlavaHf.__init__ = init_with_token
    # Keep the patch local to this LMMS child. The process exits after one
    # task, so restoring is unnecessary and would race model construction.
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
