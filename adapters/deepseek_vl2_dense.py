"""Original DeepSeek-VL2 model and processor in an isolated Transformers Worker."""

from pathlib import Path
from contextlib import nullcontext
import json
import subprocess
import sys

from adapters.llava_onevision_dense import GenerationTimer as BaseTimer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "ef9f91e2b6426536b83294c11742c27be66361b1"


def native_source(config):
    """Authenticate the pinned source before importing model or processor code."""
    source = (PROJECT_ROOT / config["model_code"]).resolve()
    expected = config["model_code_revision"]
    if expected != UPSTREAM_REVISION:
        raise ValueError("DeepSeek-VL2 model code revision has not been validated")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError("DeepSeek-VL2 model code checkout differs from the descriptor")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "deepseek_vl2"],
                   check=True, stdout=subprocess.DEVNULL)
    return source


def processor_module(config):
    """Load unmodified preprocessing without executing the model package init.

    Upstream models/__init__.py eagerly imports the CUDA model. A private
    namespace lets its processor and relative conversation import stay intact
    without adding native xFormers/model dependencies to the vLLM interpreter.
    """
    import importlib
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    source = native_source(config) / "deepseek_vl2" / "models"
    name = "_a4fvlm_deepseek_processor"
    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(source)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
    elif package.__path__ != [str(source)]:
        raise ValueError("DeepSeek processor imported from a different source directory")
    return importlib.import_module(name + ".processing_deepseek_vl_v2")


def native_classes(config):
    """Import a pinned, unmodified upstream checkout without touching weights."""
    source = native_source(config)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import transformers
    modern = int(transformers.__version__.split('.')[0]) >= 5
    if modern:
        from adapters.deepseek_vl2_compat import legacy_imports, bind_native_generation
    with legacy_imports() if modern else nullcontext():
        import deepseek_vl2
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    if not Path(deepseek_vl2.__file__).resolve().is_relative_to(source):
        raise ValueError("DeepSeek-VL2 imported from a different model code directory")
    if modern:
        bind_native_generation()
    return DeepseekVLV2ForCausalLM, DeepseekVLV2Processor


class DeepSeekProcessor:
    def __init__(self, native, config):
        self.native = native
        self.tokenizer = native.tokenizer
        self.config = dict(config)
        self._agent4vlm_adapter_id = "deepseek_vl2"

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, destination):
        return self.native.save_pretrained(destination)

    @property
    def image_token_id(self):
        return self.native.image_token_id

    def unexpanded_prompt(self, question):
        # Match format_messages_v2's single user turn before image expansion.
        conversation = self.native.new_chat_template()
        conversation.append_message(conversation.roles[0], image_question(question).strip())
        conversation.append_message(conversation.roles[1], "")
        return conversation.get_prompt()

    def prompt_token_ids(self, question, image_size):
        """Original context count without an extra image transform in serving."""
        from math import ceil
        from importlib import import_module
        from skills.deepseek_vl2_token import image_layout
        native = self.native
        select_best_resolution = import_module(type(native).__module__).select_best_resolution
        width, height = select_best_resolution(image_size, native.candidate_resolutions)
        side = ceil((native.image_size // native.patch_size) / native.downsample_ratio)
        count = image_layout(side, width // native.image_size, height // native.image_size)[2]
        before, after = self.unexpanded_prompt(question).split(native.image_token)
        return ([native.bos_id] + native.encode(before, bos=False, eos=False)
                + [native.image_token_id] * count + native.encode(after, bos=False, eos=False))


def load_processor(config):
    processor_class = processor_module(config).DeepseekVLV2Processor
    native = processor_class.from_pretrained(str((PROJECT_ROOT / config["path"]).resolve()),
                                            local_files_only=True)
    saved = json.loads((PROJECT_ROOT / config["path"] / "processor_config.json").read_text())
    for name in ("candidate_resolutions", "patch_size", "downsample_ratio", "sft_format"):
        if getattr(native, name) != saved[name]:
            raise ValueError(f"DeepSeek-VL2 processor did not preserve {name}")
    return DeepSeekProcessor(native, config)


def image_question(question):
    if "<image>" not in question:
        question = "<image>\n" + question
    if question.count("<image>") != 1:
        raise ValueError("DeepSeek-VL2 local evaluation requires exactly one image")
    return question


def prepare_image(processor, image, question):
    import torch
    from transformers import BatchFeature

    question = image_question(question)
    conversation = [{"role": "<|User|>", "content": question, "images": [image]},
                    {"role": "<|Assistant|>", "content": ""}]
    native = processor.native(conversations=conversation, images=[image.convert("RGB")],
                              force_batchify=True, system_prompt="")
    inputs = BatchFeature({key: native[key] for key in (
        "input_ids", "attention_mask", "images", "images_seq_mask", "images_spatial_crop")})
    inputs["images"] = inputs["images"].to(dtype=getattr(torch, processor.config["dtype"]))
    if inputs["input_ids"].shape[0] != 1 or inputs["images_spatial_crop"].shape != (1, 1, 2):
        raise ValueError("DeepSeek-VL2 requires a native single-image batch")
    return inputs


def prepare_inputs(processor, sample, media_root):
    from PIL import Image
    with Image.open(PROJECT_ROOT / media_root / sample["media"]["logical_path"]) as image:
        return prepare_image(processor, image, sample["logical_question"])


def native_model_class(config):
    native, _ = native_classes(config)

    class DeepSeekVL2Dense(native):
        def __init__(self, model_config):
            super().__init__(model_config)
            import transformers
            if int(transformers.__version__.split('.')[0]) >= 5:
                from adapters.deepseek_vl2_compat import finish_model
                finish_model(self)
            # Upstream selects inputs_embeds only when the initial cache is
            # None. Transformers 4.46 otherwise injects an empty DynamicCache
            # before prefill. Retain upstream's legacy generation convention;
            # its decoder still uses the original native MLA cache internally.
            self.language._supports_cache_class = False

        def generate(self, input_ids, attention_mask, images, images_seq_mask,
                     images_spatial_crop, **options):
            length = input_ids.shape[1]
            if length + options.get("max_new_tokens", 0) > self.config.language_config.max_position_embeddings:
                raise ValueError("DeepSeek-VL2 input plus generation exceeds the native context; no truncation is allowed")
            embeddings = self.prepare_inputs_embeds(
                input_ids=input_ids, images=images, images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop)
            return self.language.generate(inputs_embeds=embeddings,
                                          attention_mask=attention_mask, **options)

    return DeepSeekVL2Dense


def load_model(config):
    import torch

    header = json.loads((PROJECT_ROOT / config["path"] / "config.json").read_text())
    if header.get("a4fvlm_stack_widths") is not None or header.get("a4fvlm_deepseek_gptq") is not None:
        from skills.deepseek_vl2_artifact import load_artifact
        return load_artifact(config)

    if config["attn_implementation"] != "eager":
        raise ValueError("DeepSeek-VL2 original Dense currently requires native eager MLA")
    processor = load_processor(config)
    model, info = native_model_class(config).from_pretrained(
        str((PROJECT_ROOT / config["path"]).resolve()), local_files_only=True,
        torch_dtype=getattr(torch, config["dtype"]), attn_implementation="eager",
        low_cpu_mem_usage=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"DeepSeek-VL2 weights did not load exactly: {info}")
    # Transformers 4.46's keep-in-fp32 path only applies to float16 loading,
    # not bfloat16. Restore the small native F32 router tensors explicitly
    # before device placement; casting rounded BF16 values back is insufficient.
    from safetensors import safe_open
    path = (PROJECT_ROOT / config["path"]).resolve()
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    router_dtypes = {}
    for name, parameter in model.named_parameters():
        if name.endswith("mlp.gate.e_score_correction_bias"):
            with safe_open(path / index[name], framework="pt", device="cpu") as shard:
                original = shard.get_tensor(name)
            if original.dtype != torch.float32 or original.shape != parameter.shape:
                raise ValueError("DeepSeek native routing bias differs from its original F32 checkpoint")
            parameter.data = original.to(device=parameter.device)
            router_dtypes[name] = str(parameter.dtype)
    if len(router_dtypes) != 29:
        raise ValueError("DeepSeek full model requires all 29 native routing biases")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(model.language.generation_config, key, getattr(processor.tokenizer, key))
    model.language.generation_config.use_cache = True
    model._agent4vlm_loading_info = {
        key: sorted(value) if isinstance(value, set) else value for key, value in info.items()
    }
    model._agent4vlm_router_dtypes = router_dtypes
    return model.to(config["device"]).eval(), processor


def image_token_count(model, inputs):
    return int(inputs["images_seq_mask"].sum())


class GenerationTimer(BaseTimer):
    def projected(self, module, args, output):
        self.preparation_start = self.event()

    def __enter__(self):
        try:
            self.handles.append(self.model.vision.register_forward_pre_hook(self.vision_before))
            self.handles.append(self.model.vision.register_forward_hook(self.vision_after, prepend=True))
            if self.pruning:
                self.handles.append(self.model.projector.register_forward_hook(self.projected))
            self.handles.append(self.model.language.model.register_forward_pre_hook(self.decoder_before, with_kwargs=True))
            self.handles.append(self.model.language.model.register_forward_hook(self.decoder_after, with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def decoder_before(self, module, args, kwargs):
        self.decoder_start = self.event()
        if self.pruning and self.decoder_calls == 0:
            self.intervals["pruning_seconds"].append((self.preparation_start, self.decoder_start))
