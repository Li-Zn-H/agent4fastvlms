"""Lazy vLLM compatibility for Transformers 5 SigLIP checkpoint names."""

import sys
import re


def normalize_siglip_weight_name(name):
    prefix = "model.vision_tower."
    if name.startswith(prefix):
        suffix = name[len(prefix):]
        if suffix.startswith(("embeddings.", "encoder.", "post_layernorm.", "head.")):
            return prefix + "vision_model." + suffix
    return name


def siglip_weights_mapper(base):
    """Map both checkpoint tensors and GPTQ module selection prefixes."""
    from vllm.model_executor.models.utils import WeightsMapper

    return base | WeightsMapper(orig_to_new_regex={
        re.compile(r"^model\.vision_tower\.(?!vision_model\.)"): "model.vision_tower.vision_model."
    })


def create_model_class():
    from vllm.model_executor.models.llava_onevision import (
        LlavaOnevisionForConditionalGeneration, LlavaOnevisionMultiModalProcessor,
        LlavaOnevisionProcessingInfo, LlavaOnevisionDummyInputsBuilder,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY

    @MULTIMODAL_REGISTRY.register_processor(
        LlavaOnevisionMultiModalProcessor, info=LlavaOnevisionProcessingInfo,
        dummy_inputs=LlavaOnevisionDummyInputsBuilder,
    )
    class A4FVLMLlavaOnevisionForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
        hf_to_vllm_mapper = siglip_weights_mapper(LlavaOnevisionForConditionalGeneration.hf_to_vllm_mapper)
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                                  "gate_up_proj": ["gate_proj", "up_proj"]}

        def load_weights(self, weights):
            return super().load_weights((normalize_siglip_weight_name(name), tensor)
                                        for name, tensor in weights)

    return A4FVLMLlavaOnevisionForConditionalGeneration


if "vllm" in sys.modules:
    A4FVLMLlavaOnevisionForConditionalGeneration = create_model_class()
