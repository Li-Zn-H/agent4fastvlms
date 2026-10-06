"""Lazy vLLM LLaVA-1.5 compatibility for Transformers 5 CLIP names."""
import re
import sys


def normalize_clip_weight_name(name):
    prefix = "model.vision_tower."
    if name.startswith(prefix) and not name.startswith(prefix + "vision_model."):
        return prefix + "vision_model." + name[len(prefix):]
    return name


def clip_weights_mapper(base):
    from vllm.model_executor.models.utils import WeightsMapper

    return base | WeightsMapper(orig_to_new_regex={
        re.compile(r"^model\.vision_tower\.(?!vision_model\.)"): "model.vision_tower.vision_model."
    })


def create_model_class():
    from vllm.model_executor.models.llava import (
        LlavaForConditionalGeneration, LlavaMultiModalProcessor,
        LlavaProcessingInfo, LlavaDummyInputsBuilder,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY

    @MULTIMODAL_REGISTRY.register_processor(
        LlavaMultiModalProcessor, info=LlavaProcessingInfo, dummy_inputs=LlavaDummyInputsBuilder,
    )
    class A4FVLMLlava15ForConditionalGeneration(LlavaForConditionalGeneration):
        hf_to_vllm_mapper = clip_weights_mapper(LlavaForConditionalGeneration.hf_to_vllm_mapper)

        def load_weights(self, weights):
            return super().load_weights((normalize_clip_weight_name(name), tensor)
                                        for name, tensor in weights)

    return A4FVLMLlava15ForConditionalGeneration


if "vllm" in sys.modules:
    A4FVLMLlava15ForConditionalGeneration = create_model_class()
