"""Qwen3-VL Dense implementations, imported only inside the selected Worker."""

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_processor(config):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str((PROJECT_ROOT / config["path"]).resolve()), local_files_only=True)


def image_token_count(model, inputs):
    return int((inputs["input_ids"] == model.config.image_token_id).sum())


def token_geometry(processor, inputs):
    if inputs["image_grid_thw"].shape != (1, 3):
        raise ValueError("Expected one still image")
    t, h, w = inputs["image_grid_thw"][0].tolist()
    if t != 1:
        raise ValueError("Expected one still image")
    merge = processor.image_processor.merge_size
    return {"rows": h // merge, "columns": w // merge}


def load_model(config):
    """Load local dense weights and their native processor once per model."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    path = str((PROJECT_ROOT / config["path"]).resolve())
    processor = load_processor(config)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=getattr(torch, config["dtype"]),
        attn_implementation=config["attn_implementation"],
        local_files_only=True,
    ).to(config["device"])
    model.eval()
    return model, processor


def prepare_inputs(processor, sample, media_root):
    """Build the native single-image prompt and CPU processor inputs."""
    from PIL import Image

    image_path = PROJECT_ROOT / media_root / sample["media"]["logical_path"]
    with Image.open(image_path) as source, source.convert("RGB") as image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["logical_question"]},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    return inputs


class GenerationTimer:
    """CUDA event intervals on the model stream, resolved after generation sync.

    Enter after the candidate's compaction hooks are installed. Bracketing those
    hooks separates selection/mask preparation from actual decoder execution.
    No phase boundary synchronizes the GPU; outer wall-clock timing remains the
    measure of complete inference including uninstrumented work and overhead.
    """

    def __init__(self, model, *, pruning=False):
        self.model = model
        self.pruning = pruning
        self.handles = []
        self.intervals = {
            key: []
            for key in (
                "vision_encode_seconds",
                "pruning_seconds",
                "prefill_seconds",
                "decode_seconds",
            )
        }
        self.decoder_calls = 0
        self.vision_start: Any = None
        self.preparation_start: Any = None
        self.decoder_start: Any = None

    def event(self):
        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.model.device))
        return event

    def vision_before(self, module, args):
        self.vision_start = self.event()

    def vision_after(self, module, args, output):
        self.intervals["vision_encode_seconds"].append(
            (self.vision_start, self.event())
        )

    def preparation_before(self, module, args, kwargs):
        self.preparation_start = self.event()

    def decoder_before(self, module, args, kwargs):
        self.decoder_start = self.event()
        if self.pruning:
            self.intervals["pruning_seconds"].append(
                (self.preparation_start, self.decoder_start)
            )

    def decoder_after(self, module, args, kwargs, output):
        key = "prefill_seconds" if self.decoder_calls == 0 else "decode_seconds"
        self.intervals[key].append((self.decoder_start, self.event()))
        self.decoder_calls += 1

    def __enter__(self):
        visual = self.model.model.visual
        decoder = self.model.model.language_model
        try:
            self.handles.append(visual.register_forward_pre_hook(self.vision_before))
            self.handles.append(
                visual.register_forward_hook(self.vision_after, prepend=True)
            )
            if self.pruning:
                self.handles.append(
                    decoder.register_forward_pre_hook(
                        self.preparation_before, with_kwargs=True, prepend=True
                    )
                )
            self.handles.append(
                decoder.register_forward_pre_hook(self.decoder_before, with_kwargs=True)
            )
            self.handles.append(
                decoder.register_forward_hook(
                    self.decoder_after, with_kwargs=True, prepend=True
                )
            )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def metrics(self):
        if len(self.intervals["vision_encode_seconds"]) != 1 or self.decoder_calls < 1:
            raise RuntimeError(
                "Expected one vision encoding and at least one decoder forward"
            )
        result: dict[str, Any] = {
            key: sum(start.elapsed_time(end) for start, end in spans) / 1000
            for key, spans in self.intervals.items()
        }
        result["decode_tokens"] = self.decoder_calls - 1
        result["decode_mean_seconds_per_token"] = (
            result["decode_seconds"] / result["decode_tokens"]
            if result["decode_tokens"]
            else None
        )
        return result

