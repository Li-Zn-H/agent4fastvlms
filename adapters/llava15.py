"""Lazy registration for the fixed-resolution Hugging Face LLaVA-1.5 model."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class Llava15Adapter(ModelAdapter):
    id = "llava15"
    model_types = ("llava",)
    pipeline_ids = ("llava15",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    text_defaults = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 11008,
                     "num_attention_heads": 32, "num_key_value_heads": 32}
    hardware_ids = ("cuda",)
    entrypoints = MappingProxyType({
        "dense_load": "adapters.llava15_dense:load_model",
        "processor_load": "adapters.llava_onevision_dense:load_processor",
        "prepare_inputs": "adapters.llava_onevision_dense:prepare_inputs",
        "generation_timer": "adapters.llava_onevision_dense:GenerationTimer",
        "image_token_count": "adapters.llava_onevision_dense:image_token_count",
        "token_geometry": "adapters.llava15_dense:token_geometry",
        "token_controller": "adapters.llava15_dense:Llava15VisionTokenController",
        "sp_apply": "skills.structured_pruning:apply_recipe",
        "quantization_export": "skills.quantization_adapters:export_candidate",
        "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
        "stack_load": "skills.stack_model:load_stack_model",
        "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
        "lmms_native": "search.lmms_llava_stack_entry:main",
        "lmms_vllm": "search.lmms_llava_vllm_entry:install_bridge",
    })

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("model_type") != "llava":
            raise AdapterError("LLaVA-1.5 requires model_type=llava")
        text, vision = checkpoint.get("text_config"), checkpoint.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise AdapterError("LLaVA-1.5 requires text_config and vision_config")
        # Older HF checkpoints omit LlamaConfig defaults. Resolve those known
        # defaults without importing Transformers during static registration.
        expected_text = {"model_type": "llama", **self.text_defaults}
        for key, expected in expected_text.items():
            value = text.get(key, expected)
            if type(value) is not type(expected) or value != expected:
                raise AdapterError("LLaVA-1.5 adapter requires the 7B Llama topology")
        for key, expected in {"model_type": "clip_vision_model", "image_size": 336,
                              "patch_size": 14, "hidden_size": 1024,
                              "intermediate_size": 4096, "num_hidden_layers": 24}.items():
            value = vision.get(key)
            if type(value) is not type(expected) or value != expected:
                raise AdapterError("LLaVA-1.5 adapter requires CLIP ViT-L/14 at 336 pixels")
        for key, expected in {"image_token_index": 32000, "vision_feature_layer": -2,
                              "vision_feature_select_strategy": "default"}.items():
            value = checkpoint.get(key)
            if type(value) is not type(expected) or value != expected:
                raise AdapterError("LLaVA-1.5 native image feature contract differs")

    def describe(self):
        return {"id": self.id, "model_types": list(self.model_types),
                "pipeline_ids": list(self.pipeline_ids), "entrypoints": dict(self.entrypoints),
                "implemented_stages": list(self.stages), "hardware_ids": list(self.hardware_ids),
                "scope": {"modality": "single_image_fixed_336", "text_architecture": "llama",
                          "visual_tokens": 576},
                "qualification": "The local 7B CUDA path has independent Skill, Stack, Runtime/KV, final LMMS and matched end-to-end evidence. Registration describes implementation; every job still requires matching stage receipts."}
