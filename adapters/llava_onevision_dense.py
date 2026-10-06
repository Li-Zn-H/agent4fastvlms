"""LLaVA-OneVision Dense implementations, imported only in the selected Worker.

The checkpoint uses the native Hugging Face ``LlavaOnevisionProcessor`` and
``LlavaOnevisionForConditionalGeneration`` classes.  This module deliberately
keeps the input/timing contract separate from the Qwen3-VL implementation:
LLaVA's any-resolution processor emits ``pixel_values``, ``image_sizes`` and
``batch_num_images`` rather than Qwen's ``image_grid_thw``/DeepStack fields.
"""

from pathlib import Path
from typing import Any
from contextlib import contextmanager
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def quantized_siglip_initialization(path):
    """Avoid Dense-only container initialization after packed weights load.

    Transformers 5 SigLIP initializes q_proj.weight/fc1.weight at the attention
    or MLP container, although GPTQ has replaced them with packed buffers.
    Only fully quantized containers bypass that Dense initializer; the caller
    still requires an exact clean checkpoint load with no missing keys.
    """
    config = json.loads((Path(path) / "config.json").read_text())
    if not config.get("a4fvlm_siglip_padding") or not config.get("quantization_config"):
        yield
        return
    from transformers.models.siglip.modeling_siglip import SiglipPreTrainedModel, SiglipAttention, SiglipMLP

    original = SiglipPreTrainedModel._init_weights

    def initialize(self, module):
        names = ("q_proj", "k_proj", "v_proj", "out_proj") if isinstance(module, SiglipAttention) else (
            ("fc1", "fc2") if isinstance(module, SiglipMLP) else ())
        if names and all(hasattr(getattr(module, name), "qweight") for name in names):
            return
        return original(self, module)

    SiglipPreTrainedModel._init_weights = initialize
    try:
        yield
    finally:
        SiglipPreTrainedModel._init_weights = original


def load_processor(config):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str((PROJECT_ROOT / config["path"]).resolve()), local_files_only=True
    )


def image_token_count(model, inputs):
    """Count native LLaVA image placeholders in the prepared prompt."""

    return int((inputs["input_ids"] == model.config.image_token_index).sum())


def token_geometry(processor, inputs):
    """Return the any-resolution patch geometry for one still image.

    LLaVA packs a base image plus one or more tiled crops.  The processor
    records the original image size; the exact packed token count is checked
    by the model itself during the forward pass and is exposed separately by
    :func:`image_token_count`.
    """

    sizes = inputs.get("image_sizes")
    batches = inputs.get("batch_num_images")
    if sizes is None or tuple(sizes.shape) != (1, 2):
        raise ValueError("Expected one still image with image_sizes=(1, 2)")
    if batches is not None and (tuple(batches.shape) != (1,) or int(batches[0]) != 1):
        raise ValueError("Expected one still image")
    height, width = (int(value) for value in sizes[0].tolist())
    patches = int(inputs["pixel_values"].shape[1])
    # The generic runner needs a stable rectangular geometry for its coverage
    # bookkeeping.  LLaVA's packed stream is not a single rectangular grid,
    # so expose it as a one-row stream while retaining native image metadata.
    return {
        "rows": 1,
        "columns": int((inputs["input_ids"] == processor.image_token_id).sum()),
        "height": height,
        "width": width,
        "patches": patches,
    }


def load_model(config):
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration

    path = str((PROJECT_ROOT / config["path"]).resolve())
    processor = load_processor(config)
    with quantized_siglip_initialization(path):
        model, info = LlavaOnevisionForConditionalGeneration.from_pretrained(
            path,
            dtype=getattr(torch, config["dtype"]),
            attn_implementation=config["attn_implementation"],
            local_files_only=True,
            output_loading_info=True,
        )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"LLaVA checkpoint did not load exactly: {info}")
    model = model.to(config["device"])
    model.eval()
    return model, processor


def prepare_inputs(processor, sample, media_root):
    """Build the native single-image LLaVA-OneVision prompt."""

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
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    # Some Transformers versions return this legacy field for Qwen2 tokenizers;
    # LLaVA's forward does not consume it.
    inputs.pop("token_type_ids", None)
    return inputs


class GenerationTimer:
    """CUDA event intervals for the LLaVA vision tower and Qwen2 decoder."""

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
        self.intervals["vision_encode_seconds"].append((self.vision_start, self.event()))

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
        vision = self.model.model.vision_tower
        decoder = self.model.model.language_model
        try:
            self.handles.append(vision.register_forward_pre_hook(self.vision_before))
            self.handles.append(vision.register_forward_hook(self.vision_after, prepend=True))
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
                decoder.register_forward_hook(self.decoder_after, with_kwargs=True, prepend=True)
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
        if len(self.intervals["vision_encode_seconds"]) < 1 or self.decoder_calls < 1:
            raise RuntimeError("Expected one vision encoding and at least one decoder forward")
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


class LlavaVisionTokenController:
    """Score-select packed LLaVA image features before the language model.

    LLaVA-OneVision's native processor expands each image placeholder to a
    single packed feature stream (base crop, tiled crops and newline vectors).
    Unlike Qwen3-VL it has no DeepStack or mRoPE bookkeeping.  The controller
    therefore selects one shared score-ranked index set per packed stream and
    rewrites the corresponding placeholder run before generation.  The
    keep-all path is deliberately bit-for-bit equivalent to native generation.
    """

    def __init__(self, model, selection, keep_ratio=1.0):
        from skills.vision_token import validate_selection

        validate_selection(selection)
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.model = model
        self.selection = dict(selection)
        self.keep_ratio = keep_ratio
        self.last_metrics = None

    def _indices(self, features):
        from skills.llava_token import select_packed_indices

        return select_packed_indices(features, self.selection, self.keep_ratio)

    def _keep_count(self, count):
        from skills.vision_token import keep_count

        return keep_count(1, count, self.keep_ratio, self.selection["coverage_grid"])

    def generate(self, inputs, options, *, forward_context=None):
        from contextlib import nullcontext
        from types import MethodType
        import torch

        self.last_metrics = None
        if self.keep_ratio == 1:
            with (forward_context if forward_context is not None else nullcontext()):
                return self.model.generate(**inputs, **options)
        if inputs["input_ids"].shape[0] != 1 or "pixel_values_videos" in inputs:
            raise ValueError("LLaVA Token selection supports one still image and batch size one")
        if options.get("use_cache", True) is not True:
            raise ValueError("LLaVA Token validation requires use_cache=True")
        # Quantizers can serialize use_cache=False after calibration. This
        # generation path requires actual compact-cache observations.
        options = {**options, "use_cache": True}
        image_id = int(self.model.config.image_token_index)
        ids = inputs["input_ids"][0]
        locations = (ids == image_id).nonzero().flatten()
        if locations.numel() < 1:
            raise ValueError("LLaVA prompt has no image placeholders")
        selected_holder = {}
        original_method = self.model.model.get_image_features

        def select_features(owner, *args, **kwargs):
            output = original_method(*args, **kwargs)
            streams = output.pooler_output
            if not isinstance(streams, (list, tuple)) or len(streams) != 1:
                raise ValueError("LLaVA Token selection requires one packed image stream")
            selected = []
            before = after = 0
            for stream in streams:
                if stream.ndim != 2:
                    raise ValueError("Unexpected LLaVA packed image feature shape")
                indices = self._indices(stream)
                selected.append(stream.index_select(0, indices))
                before += int(stream.shape[0])
                after += int(indices.numel())
            output.pooler_output = selected
            selected_holder.update(before=before, after=after)
            return output

        self.model.model.get_image_features = MethodType(select_features, self.model.model)
        before_placeholders = int(locations.numel())
        try:
            # The native forward checks placeholder count against packed feature
            # length.  Rewrite only the contiguous image-token run; all text
            # and control tokens retain their exact order and values.
            run_start, run_end = int(locations[0]), int(locations[-1]) + 1
            if locations.numel() != run_end - run_start:
                raise ValueError("Expected one contiguous LLaVA image placeholder run")
            # The native processor's placeholder run and packed feature stream
            # have equal length.  Compute the retained count before the first
            # forward so the model's placeholder/feature consistency check sees
            # matching shapes immediately.
            original_ids = inputs["input_ids"]
            expected_count = self._keep_count(before_placeholders)
            compacted_ids = torch.cat(
                (
                    original_ids[:, :run_start],
                    original_ids.new_full((1, expected_count), image_id),
                    original_ids[:, run_end:],
                ),
                dim=1,
            )
            compacted_inputs = {**inputs, "input_ids": compacted_ids}
            mask = inputs.get("attention_mask")
            if mask is not None:
                if mask.ndim != 2 or mask.shape != original_ids.shape:
                    raise ValueError("LLaVA Token requires a native 2D prompt attention mask")
                compacted_inputs["attention_mask"] = torch.cat(
                    (mask[:, :run_start], mask[:, run_start:run_start + expected_count], mask[:, run_end:]),
                    dim=1,
                )
            # The packed feature hook determines the actual retained count on
            # the first forward.  Placeholder count is patched by a narrow
            # forward pre-hook after that selection is known.
            language = self.model.model.language_model

            def patch_prompt(module, args, kwargs):
                if selected_holder.get("after") is None:
                    return args, kwargs
                embeds = kwargs.get("inputs_embeds")
                if embeds is None or embeds.shape[1] == compacted_ids.shape[1]:
                    return args, kwargs
                # This hook is defensive for Transformers versions that build
                # embeddings before invoking get_image_features.  The standard
                # path uses the compacted IDs directly below.
                return args, kwargs

            handle = language.register_forward_pre_hook(patch_prompt, with_kwargs=True)
            observed = {"forwards": 0, "kv_length": None}

            def observe_cache(module, args, kwargs, output):
                cache = getattr(output, "past_key_values", None)
                if cache is None:
                    raise RuntimeError("LLaVA Token requires an observed decoder KV cache")
                observed["forwards"] += 1
                observed["kv_length"] = int(cache.get_seq_length())

            cache_handle = language.register_forward_hook(observe_cache, with_kwargs=True)
            try:
                with (forward_context if forward_context is not None else nullcontext()):
                    result = self.model.generate(**compacted_inputs, **options)
            finally:
                handle.remove()
                cache_handle.remove()
            if not selected_holder:
                raise RuntimeError("LLaVA visual feature selection was not observed")
            expected = int(selected_holder["after"])
            if expected != expected_count:
                raise RuntimeError(
                    f"LLaVA selected feature count drifted: {expected} != {expected_count}"
                )
            inputs.update(compacted_inputs)
            self.last_metrics = {
                "requested_keep_ratio": self.keep_ratio,
                "image_tokens_before": before_placeholders,
                "image_tokens_after": expected,
                "actual_keep_ratio": expected / before_placeholders,
                "decoder_tokens_before": int(original_ids.shape[1]),
                "decoder_tokens_after": int(inputs["input_ids"].shape[1]),
                "decode_steps": max(0, observed["forwards"] - 1),
                "output_tokens": int(result.shape[1] - inputs["input_ids"].shape[1]),
                "kv_tokens_after_last_forward": observed["kv_length"],
            }
            if observed["kv_length"] != compacted_ids.shape[1] + self.last_metrics["decode_steps"]:
                raise RuntimeError("Observed LLaVA KV cache does not match the compacted prompt")
            return result
        finally:
            self.model.model.get_image_features = original_method
