"""Native InternVL inputs and all-tile post-encoder Token compaction.

Uses the pinned LMMS image transform and the checkpoint's conversation template.
Generation retains the native suffix-only return convention, also used by chat.
"""

from contextlib import nullcontext
from math import isqrt
from pathlib import Path
from types import MethodType

from adapters.llava_onevision_dense import GenerationTimer as BaseTimer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InternVLProcessor:
    def __init__(self, tokenizer, config, conversation_factory, dtype, *, adapter_id="internvl35"):
        self.tokenizer = tokenizer
        self.config = config
        self.conversation_factory = conversation_factory
        self.dtype = dtype
        self.image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if self.image_token_id is None or self.image_token_id == tokenizer.unk_token_id:
            raise ValueError("InternVL tokenizer has no image context token")
        self.tokens_per_tile = int((config.force_image_size // config.vision_config.patch_size) ** 2
                                  * config.downsample_ratio ** 2)
        self._agent4vlm_adapter_id = adapter_id

    def unexpanded_prompt(self, question):
        if "<image>" not in question:
            question = "<image>\n" + question
        if question.count("<image>") != 1 or any(
            token in question for token in ("<img>", "</img>", "<IMG_CONTEXT>")
        ):
            raise ValueError("InternVL supports exactly one native image placeholder")
        conversation = self.conversation_factory(self.config.template)
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    def prompt(self, question, tiles):
        tokens = "<img>" + "<IMG_CONTEXT>" * (tiles * self.tokens_per_tile) + "</img>"
        return self.unexpanded_prompt(question).replace("<image>", tokens, 1)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, destination):
        """Preserve tokenizer and original custom code beside an artifact."""
        import shutil

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        source = Path(self.config._name_or_path)
        for path in source.glob("*.py"):
            shutil.copyfile(path, destination / path.name)


def load_processor(config, *, artifact=False):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = str((PROJECT_ROOT / config["path"]).resolve())
    native = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    from adapters import AdapterError, select_model
    descriptor = native.to_dict()
    if artifact:
        from skills.internvl35_artifact import validate_artifact_config
        validate_artifact_config(descriptor)
    adapter = select_model(descriptor)
    if config.get("adapter", adapter.id) != adapter.id:
        raise AdapterError("InternVL processor adapter differs from checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    conversation = get_class_from_dynamic_module("conversation.get_conv_template", path, local_files_only=True)
    return InternVLProcessor(tokenizer, native, conversation, getattr(torch, config["dtype"]),
                             adapter_id=adapter.id)


def native_model_class(path):
    """Finish the legacy custom class lifecycle required by Transformers 5.

    The checkpoint's chat/vision constructors predate composite post_init
    metadata. Use a local subclass, leaving checkpoint and cached code intact.
    Exact loading diagnostics below still reject every absent/extra weight.
    """
    from transformers import PreTrainedModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    native = get_class_from_dynamic_module(
        "modeling_internvl_chat.InternVLChatModel", str(path), local_files_only=True
    )

    class InternVL35Model(native):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            for module in reversed(list(self.modules())):
                if isinstance(module, PreTrainedModel) and not hasattr(module, "all_tied_weights_keys"):
                    module.post_init()

    return InternVL35Model


def load_model(config):
    import json
    import torch

    path = str((PROJECT_ROOT / config["path"]).resolve())
    header = json.loads((Path(path) / "config.json").read_text())
    if header.get("a4fvlm_stack_widths") is not None or header.get("a4fvlm_internvl_gptq") is not None:
        from skills.internvl35_artifact import load_artifact
        return load_artifact(config)
    if config["attn_implementation"] not in {"eager", "sdpa"}:
        raise ValueError("InternVL preparation supports eager/SDPA decoder attention only")
    processor = load_processor(config)
    model, info = native_model_class(path).from_pretrained(
        path,
        local_files_only=True, dtype=getattr(torch, config["dtype"]),
        use_flash_attn=False, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"InternVL checkpoint did not load exactly: {info}")
    # Native __init__ forces eager when FlashAttention is absent. Set the
    # decoder explicitly afterwards; vision keeps the native eager route.
    model.language_model.set_attn_implementation(config["attn_implementation"])
    if model.language_model.config._attn_implementation != config["attn_implementation"]:
        raise ValueError("InternVL decoder attention implementation did not take effect")
    model.img_context_token_id = processor.image_token_id
    separator = processor.conversation_factory(processor.config.template).sep.strip()
    model.language_model.generation_config.eos_token_id = processor.tokenizer.convert_tokens_to_ids(separator)
    model = model.to(config["device"]).eval()
    return model, processor


def prepare_image(processor, image, question):
    import importlib.util
    from transformers import BatchFeature
    if importlib.util.find_spec("lmms_eval") is None:
        from adapters.internvl35_processing import load_image
    else:
        from lmms_eval.models.simple.internvl3 import load_image

    pixels = load_image(image.convert("RGB"), input_size=processor.config.force_image_size,
                        max_num=processor.config.max_dynamic_patch).to(dtype=processor.dtype)
    tokens = processor.tokenizer(processor.prompt(question, len(pixels)),
                                 return_tensors="pt")
    tokens.pop("token_type_ids", None)
    inputs = BatchFeature({**tokens, "pixel_values": pixels})
    token_geometry(processor, inputs)
    return inputs


def prepare_inputs(processor, sample, media_root):
    from PIL import Image
    with Image.open(PROJECT_ROOT / media_root / sample["media"]["logical_path"]) as source:
        return prepare_image(processor, source, sample["logical_question"])


def token_geometry(processor, inputs):
    pixels, ids = inputs["pixel_values"], inputs["input_ids"]
    size = processor.config.force_image_size
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, size, size) or ids.shape[0] != 1:
        raise ValueError("InternVL requires one image with native square tiles")
    tiles = int(pixels.shape[0])
    if not 1 <= tiles <= processor.config.max_dynamic_patch + 1:
        raise ValueError("InternVL tile count is outside its native envelope")
    if int((ids == processor.image_token_id).sum()) != tiles * processor.tokens_per_tile:
        raise ValueError("InternVL prompt and visual tile counts disagree")
    side = isqrt(processor.tokens_per_tile)
    if side * side != processor.tokens_per_tile:
        raise ValueError("InternVL requires square post-shuffle tile grids")
    return {"rows": side, "columns": side, "repeats": tiles,
            "height": size, "width": size, "patches": tiles}


def image_token_count(model, inputs):
    return int((inputs["input_ids"] == model.img_context_token_id).sum())


class GenerationTimer(BaseTimer):
    """Native vision, decoder prefill/decode and post-projector preparation.

    pruning_seconds includes selection and preparation between the projector
    and first decoder forward; it is not isolated selector-kernel timing.
    """

    def projected(self, module, args, output):
        self.preparation_start = self.event()

    def decoder_before(self, module, args, kwargs):
        self.decoder_start = self.event()
        if self.pruning and self.decoder_calls == 0:
            self.intervals["pruning_seconds"].append((self.preparation_start, self.decoder_start))

    def __enter__(self):
        try:
            self.handles.append(self.model.vision_model.register_forward_pre_hook(self.vision_before))
            self.handles.append(self.model.vision_model.register_forward_hook(self.vision_after, prepend=True))
            if self.pruning:
                self.handles.append(self.model.mlp1.register_forward_hook(self.projected))
            decoder = self.model.language_model.model
            self.handles.append(decoder.register_forward_pre_hook(self.decoder_before, with_kwargs=True))
            self.handles.append(decoder.register_forward_hook(self.decoder_after, with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self


class InternVLTokenController:
    """Select spatial representatives in EVERY tile, including the thumbnail."""

    def __init__(self, model, selection, keep_ratio=1.0):
        from skills.vision_token import validate_selection
        validate_selection(selection)
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.model, self.selection, self.keep_ratio = model, dict(selection), keep_ratio
        self.last_metrics = None

    def generate(self, inputs, options, *, forward_context=None):
        import torch
        from skills.vision_token import keep_count, select_indices

        self.last_metrics = None
        context = forward_context if forward_context is not None else nullcontext()
        if self.keep_ratio == 1:
            with context:
                return self.model.generate(**inputs, **options)
        if options.get("do_sample") or options.get("use_cache", True) is not True or options.get("num_beams", 1) != 1:
            raise ValueError("InternVL Token requires greedy single-beam cached generation")
        if set(inputs) - {"input_ids", "attention_mask", "pixel_values"}:
            raise ValueError("InternVL Token requires native single-image inputs")
        ids, pixels = inputs["input_ids"], inputs["pixel_values"]
        if ids.ndim != 2 or ids.shape[0] != 1 or pixels.ndim != 4 or pixels.shape[0] < 1:
            raise ValueError("InternVL Token requires batch size one with native tiles")
        image_id = self.model.img_context_token_id
        locations = (ids[0] == image_id).nonzero().flatten()
        tiles, count = int(pixels.shape[0]), self.model.num_image_token
        side = isqrt(count)
        if side * side != count or locations.numel() != tiles * count:
            raise ValueError("InternVL feature grid and placeholders disagree")
        start, end = int(locations[0]), int(locations[-1]) + 1
        if end - start != locations.numel():
            raise ValueError("InternVL Token requires one contiguous image placeholder run")
        kept = keep_count(side, side, self.keep_ratio, self.selection["coverage_grid"])
        after = tiles * kept
        positions = torch.cat((torch.arange(start + after, device=ids.device),
                               torch.arange(end, ids.shape[1], device=ids.device)))
        compacted = {**inputs, "input_ids": ids.index_select(1, positions)}
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"]
            if mask.shape != ids.shape or not bool((mask == 1).all()):
                raise ValueError("InternVL Token requires a native unpadded prompt mask")
            compacted["attention_mask"] = mask.index_select(1, positions)
        original = self.model.extract_feature
        owned = "extract_feature" in self.model.__dict__
        observed = {"selection_calls": 0, "forwards": 0, "kv_length": None}

        def extract(owner, pixel_values):
            features = original(pixel_values)
            if features.ndim != 3 or tuple(features.shape[:2]) != (tiles, count):
                raise ValueError("InternVL native visual features changed shape")
            observed["selection_calls"] += 1
            return torch.stack([tile.index_select(0, select_indices(tile, side, side, self.keep_ratio, self.selection))
                                for tile in features])

        def cache(module, args, kwargs, output):
            value = getattr(output, "past_key_values", None)
            if value is None:
                raise RuntimeError("InternVL Token requires an observed decoder KV cache")
            observed["forwards"] += 1
            observed["kv_length"] = int(value.get_seq_length())

        self.model.extract_feature = MethodType(extract, self.model)
        handle = None
        try:
            handle = self.model.language_model.model.register_forward_hook(cache, with_kwargs=True)
            options = dict(options)
            # The native generate passes use_cache=True itself.
            options.pop("use_cache", None)
            with context:
                result = self.model.generate(**compacted, **options)
            length = int(compacted["input_ids"].shape[1])
            steps = observed["forwards"] - 1
            if observed["selection_calls"] != 1 or steps < 0 or observed["kv_length"] != length + steps:
                raise RuntimeError("InternVL selection/cache did not match the compacted prompt")
            self.last_metrics = {
                "requested_keep_ratio": self.keep_ratio,
                "image_tokens_before": tiles * count, "image_tokens_after": after,
                "actual_keep_ratio": kept / count, "tiles_selected": tiles,
                "tokens_per_tile_before": count, "tokens_per_tile_after": kept,
                "decoder_tokens_before": int(ids.shape[1]), "decoder_tokens_after": length,
                "decode_steps": steps, "output_tokens": int(result.shape[1]),
                "kv_tokens_after_last_forward": observed["kv_length"],
            }
            inputs.update(compacted)
            return result
        finally:
            if handle is not None:
                handle.remove()
            if owned:
                self.model.extract_feature = original
            else:
                del self.model.extract_feature
