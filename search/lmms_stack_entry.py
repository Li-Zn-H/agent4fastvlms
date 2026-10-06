"""Dispatch lmms-eval after binding the physical Qwen3-VL Stack class."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


# When this file is executed directly by the LMMS child process, Python puts
# `search/` on sys.path rather than the project root. Bind the local skills and
# search packages before importing the adapter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_GENERATION_KEYS = frozenset(
    {
        "eos_token_id",
        "pad_token_id",
        "max_new_tokens",
        "use_cache",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
    }
)


def _install_token_controller(qwen3_vl) -> None:
    """Attach the shared Token controller to the native Stack LMMS adapter.

    LMMS calls the upstream adapter's ``model.generate`` directly.  The
    controller is therefore installed as a narrow instance-level generate
    bridge so its hooks run for the same one-image greedy generation used by
    Search/Dev and vLLM.  The physical Stack artifact remains unchanged.
    """

    ratio_text = os.environ.get("A4FVLMSP_TOKEN_KEEP_RATIO")
    if ratio_text is None:
        return
    try:
        ratio = float(ratio_text)
    except ValueError as error:
        raise ValueError("A4FVLMSP_TOKEN_KEEP_RATIO must be numeric") from error
    selection_text = os.environ.get("A4FVLMSP_TOKEN_SELECTION")
    if selection_text:
        selection = json.loads(selection_text)
    else:
        selection = json.loads(
            (PROJECT_ROOT / "configs/vision_token.json").read_text(encoding="utf-8")
        )["selection"]

    from skills.vision_token import VisionTokenController

    original_init = qwen3_vl.Qwen3_VL.__init__

    def init_with_token(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        controller = self._a4fvlm_token_controller = VisionTokenController(
            self.model, selection, ratio
        )
        # Bind the model boundary during initialization: the chat adapter
        # overrides generate_until, but inherits this initialization path.
        model = self.model
        original_generate = model.generate
        delegating = False
        receipt_text = os.environ.get("A4FVLMSP_TOKEN_RECEIPT")

        def routed_generate(*args, **kwargs):
            nonlocal delegating
            if delegating:
                return original_generate(*args, **kwargs)
            if args:
                raise ValueError("LMMS Token bridge requires keyword model inputs")
            inputs = {
                key: value
                for key, value in kwargs.items()
                if key not in _GENERATION_KEYS
            }
            options = {
                key: value
                for key, value in kwargs.items()
                if key in _GENERATION_KEYS
            }
            delegating = True
            try:
                result = controller.generate(inputs, options)
                metrics = controller.last_metrics
                if ratio < 1:
                    if not metrics:
                        raise RuntimeError("LMMS Token controller produced no mechanism evidence")
                    expected = metrics["decoder_tokens_after"] + metrics["decode_steps"]
                    if metrics["kv_tokens_after_last_forward"] != expected:
                        raise RuntimeError("LMMS Token KV length does not match compacted sequence")
                if receipt_text and metrics:
                    with Path(receipt_text).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "method": "transformers_score_based_post_encoder",
                            "token_keep_ratio": ratio,
                            **metrics,
                        }) + "\n")
                return result
            finally:
                delegating = False

        model.generate = routed_generate

    qwen3_vl.Qwen3_VL.__init__ = init_with_token


def main() -> None:
    from transformers import AutoConfig

    from skills.stack_model import STACK_ARCHITECTURE, create_pruned_model_class
    from lmms_eval.models.simple import qwen3_vl

    original = qwen3_vl._resolve_model_class

    def resolve_model_class(pretrained: str, is_moe: bool):
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        if getattr(config, "architectures", None) == [STACK_ARCHITECTURE]:
            return create_pruned_model_class(), "dtype"
        return original(pretrained, is_moe)

    qwen3_vl._resolve_model_class = resolve_model_class
    _install_token_controller(qwen3_vl)
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
