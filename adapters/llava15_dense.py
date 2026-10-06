"""Native LLaVA-1.5 loading and fixed CLIP geometry; backend imports stay lazy."""

from contextlib import contextmanager
from pathlib import Path
import json

from adapters.llava_onevision_dense import (
    PROJECT_ROOT, LlavaVisionTokenController, load_processor,
)


@contextmanager
def quantized_clip_initialization(path):
    """Skip Dense container initializers only for fully packed CLIP linears."""
    config = json.loads((Path(path) / "config.json").read_text())
    if not config.get("a4fvlm_clip_gptq") or not config.get("quantization_config"):
        yield
        return
    from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPAttention, CLIPMLP

    original = CLIPPreTrainedModel._init_weights

    def initialize(self, module):
        names = ("q_proj", "k_proj", "v_proj", "out_proj") if isinstance(module, CLIPAttention) else (
            ("fc1", "fc2") if isinstance(module, CLIPMLP) else ())
        if names and all(hasattr(getattr(module, name), "qweight") for name in names):
            return
        return original(self, module)

    CLIPPreTrainedModel._init_weights = initialize
    try:
        yield
    finally:
        CLIPPreTrainedModel._init_weights = original


def load_model(config):
    import torch
    from transformers import LlavaForConditionalGeneration

    path = str((PROJECT_ROOT / config["path"]).resolve())
    processor = load_processor(config)
    checkpoint = json.loads((Path(path) / "config.json").read_text())
    loading = {"device_map": config["device"]} if checkpoint.get("quantization_config") else {}
    with quantized_clip_initialization(path):
        model, info = LlavaForConditionalGeneration.from_pretrained(
            path, dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attn_implementation"],
            local_files_only=True, output_loading_info=True, **loading,
        )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"LLaVA-1.5 checkpoint did not load exactly: {info}")
    model = model.to(config["device"])
    model.eval()
    return model, processor


def token_geometry(processor, inputs):
    pixels = inputs["pixel_values"]
    if tuple(pixels.shape) != (1, 3, 336, 336):
        raise ValueError("LLaVA-1.5 requires one native 336x336 image")
    if getattr(processor, "patch_size", None) != 14:
        raise ValueError("LLaVA-1.5 requires 14-pixel CLIP patches")
    count = int((inputs["input_ids"] == processor.image_token_id).sum())
    if count != 576:
        raise ValueError("LLaVA-1.5 native prompt must contain 576 image placeholders")
    return {"rows": 24, "columns": 24, "height": 336, "width": 336, "patches": 1}


class Llava15VisionTokenController(LlavaVisionTokenController):
    """Select CLIP patch features with the shared native LLaVA compaction path.

    Transformers exposes the same list of projected image feature streams for
    Llava and LlavaOnevision. LLaVA-1.5 has one 576-patch stream, with its CLS
    feature removed by the native model before this selector executes.
    """

    def _indices(self, features):
        from skills.llava_token import select_clip_indices

        return select_clip_indices(features, self.selection, self.keep_ratio)

    def _keep_count(self, count):
        from skills.llava_token import clip_grid
        from skills.vision_token import keep_count

        rows, columns = clip_grid(count)
        return keep_count(rows, columns, self.keep_ratio, self.selection["coverage_grid"])
