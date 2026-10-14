"""Unmodified checkpoint code and native Instruct preprocessing for Kimi-VL."""

import hashlib
import json
from pathlib import Path

from adapters.llava_onevision_dense import GenerationTimer as BaseTimer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def native_path(config):
    path = (PROJECT_ROOT / config["path"]).resolve()
    if config.get("protocol") != "instruct" or "Thinking" in path.name:
        raise ValueError("Kimi execution requires the native non-thinking Instruct protocol")
    sources = config.get("native_source_sha256", {})
    required = {"configuration_kimi_vl.py", "modeling_kimi_vl.py", "processing_kimi_vl.py",
                "image_processing_kimi_vl.py", "tokenization_moonshot.py", "chat_template.jinja",
                "preprocessor_config.json", "tokenizer_config.json", "generation_config.json"}
    if set(sources) != required:
        raise ValueError("Kimi descriptor must authenticate all native source/protocol files")
    for name, expected in sources.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Kimi native source changed: {name}")
    return path


def load_processor(config):
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = native_path(config)
    # Transformers 4.50 passes the nested ``image_processor`` entry from
    # processor_config.json in addition to the already materialized image
    # processor positional argument.  The native Kimi processor's legacy
    # constructor then receives the same argument twice.  Materialize the
    # two native components explicitly and construct the authenticated class;
    # this preserves the checkpoint's preprocessing while avoiding that
    # version-specific ProcessorMixin conflict.
    image_processor = AutoImageProcessor.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True, use_fast=False
    )
    processor_class = get_class_from_dynamic_module(
        "processing_kimi_vl.KimiVLProcessor", str(path), local_files_only=True
    )
    processor = processor_class(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=(path / "chat_template.jinja").read_text(),
    )
    processor._agent4vlm_adapter_id = "kimi_vl"
    processor._agent4vlm_model_config = dict(config)
    saved = json.loads((path / "preprocessor_config.json").read_text())
    for key in ("patch_size", "in_token_limit", "pad_input", "image_mean", "image_std"):
        if getattr(processor.image_processor, key) != saved[key]:
            raise ValueError(f"Kimi native image preprocessing changed: {key}")
    if processor.tokenizer.convert_tokens_to_ids("<|media_pad|>") != 163605:
        raise ValueError("Kimi image placeholder differs from checkpoint")
    if processor.chat_template != (path / "chat_template.jinja").read_text():
        raise ValueError("Kimi native Instruct template differs from checkpoint")
    return processor


def unexpanded_prompt(processor, question):
    """Use the original single-image Instruct template before image expansion."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt.count(processor.image_token) != 1:
        raise ValueError("Kimi single-image prompt requires exactly one native image placeholder")
    return prompt


def prompt_token_ids(processor, question, image_size):
    """Preflight the native expanded prompt without decoding or resizing pixels."""
    from skills.kimi_vl_vllm import native_grid
    rows, columns = native_grid(*image_size, processor.image_processor)
    prompt = unexpanded_prompt(processor, question).replace(processor.image_token,
                                                          processor.image_token * (rows * columns))
    return processor.tokenizer([prompt], padding=True, truncation=False)["input_ids"][0]


def prepare_image(processor, image, question):
    prompt = unexpanded_prompt(processor, question)
    inputs = processor(images=[image.convert("RGB")], text=[prompt], return_tensors="pt",
                       padding=True, truncation=False)
    inputs.pop("token_type_ids", None)
    if inputs["input_ids"].shape[0] != 1 or inputs["image_grid_hws"].shape != (1, 2):
        raise ValueError("Kimi local path requires exactly one native still image")
    if inputs["input_ids"].shape[1] > 131072:
        raise ValueError("Kimi input exceeds native context; truncation is forbidden")
    return inputs


def prepare_inputs(processor, sample, media_root):
    from PIL import Image
    with Image.open(PROJECT_ROOT / media_root / sample["media"]["logical_path"]) as image:
        return prepare_image(processor, image, sample["logical_question"])


def image_token_count(model, inputs):
    return int((inputs["input_ids"] == model.config.media_placeholder_token_id).sum())


def native_model_class(config):
    import transformers
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = native_path(config)
    header = AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    if int(transformers.__version__.split(".")[0]) >= 5:
        from adapters.kimi_vl_compat import legacy_imports, modern_model_class
        with legacy_imports():
            cls = get_class_from_dynamic_module(header.auto_map["AutoModelForCausalLM"], str(path), local_files_only=True)
        return modern_model_class(cls)
    return get_class_from_dynamic_module(header.auto_map["AutoModelForCausalLM"], str(path), local_files_only=True)


def load_model(config):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    path = native_path(config)
    if config["attn_implementation"] != "eager":
        raise ValueError("Kimi original Dense requires its native eager MLA reference")
    header = json.loads((path / "config.json").read_text())
    if header.get("quantization_config") or header.get("a4fvlm_stack_physical_compaction"):
        from skills.kimi_vl_artifact import load_artifact
        return load_artifact(config)
    processor = load_processor(config)
    loader = native_model_class(config) if int(transformers.__version__.split(".")[0]) >= 5 else AutoModelForCausalLM
    model, info = loader.from_pretrained(
        str(path), torch_dtype=getattr(torch, config["dtype"]), trust_remote_code=True,
        local_files_only=True, attn_implementation="eager", low_cpu_mem_usage=True,
        output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Kimi weights did not load exactly: {info}")
    info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
    if int(transformers.__version__.split(".")[0]) >= 5:
        from adapters.kimi_vl_compat import restore_native_rotary_buffers
        restore_native_rotary_buffers(model)
    # Preserve correction biases in their checkpoint precision. Rounding to
    # BF16 before casting back can change noaux_tc expert routing.
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    routing = {}
    for name, parameter in model.named_parameters():
        if name.endswith("mlp.gate.e_score_correction_bias"):
            with safe_open(path / index[name], framework="pt", device="cpu") as shard:
                original = shard.get_tensor(name)
            if original.shape != parameter.shape:
                raise ValueError("Kimi native routing bias shape changed")
            parameter.data = original.to(device=parameter.device)
            routing[name] = str(original.dtype)
    if len(routing) != 26:
        raise ValueError("Kimi requires all 26 native MoE routing biases")
    original_generate = model.generate

    def generate(*args, **options):
        ids = options.get("input_ids", args[0] if args else None)
        if ids is None or ids.shape[1] + options.get("max_new_tokens", 0) > 131072:
            raise ValueError("Kimi input plus generation exceeds native context")
        return original_generate(*args, **options)

    model.generate = generate
    model._agent4vlm_loading_info = info
    model._agent4vlm_router_dtypes = routing
    return model.to(config["device"]).eval(), processor


class GenerationTimer(BaseTimer):
    """Measure post-projector selection/preparation once, before prefill."""

    def projected(self, module, args, output):
        self.preparation_start = self.event()

    def decoder_before(self, module, args, kwargs):
        self.decoder_start = self.event()
        if self.pruning and self.decoder_calls == 0:
            self.intervals["pruning_seconds"].append(
                (self.preparation_start, self.decoder_start)
            )

    def __enter__(self):
        try:
            self.handles.append(self.model.vision_tower.register_forward_pre_hook(self.vision_before))
            self.handles.append(self.model.vision_tower.register_forward_hook(self.vision_after, prepend=True))
            if self.pruning:
                self.handles.append(self.model.multi_modal_projector.register_forward_hook(self.projected))
            self.handles.append(self.model.language_model.model.register_forward_pre_hook(
                self.decoder_before, with_kwargs=True))
            self.handles.append(self.model.language_model.model.register_forward_hook(
                self.decoder_after, with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self
