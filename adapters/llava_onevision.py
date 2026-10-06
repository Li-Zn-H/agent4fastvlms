"""Register the model-specific LLaVA-OneVision optimization interfaces."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class LlavaOneVisionAdapter(ModelAdapter):
    id = "llava_onevision"
    model_types = ("llava_onevision",)
    pipeline_ids = ("llava_onevision",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    hardware_ids = ("cuda",)
    entrypoints = MappingProxyType(
        {
            "dense_load": "adapters.llava_onevision_dense:load_model",
            "processor_load": "adapters.llava_onevision_dense:load_processor",
            "prepare_inputs": "adapters.llava_onevision_dense:prepare_inputs",
            "generation_timer": "adapters.llava_onevision_dense:GenerationTimer",
            "image_token_count": "adapters.llava_onevision_dense:image_token_count",
            "token_geometry": "adapters.llava_onevision_dense:token_geometry",
            "token_controller": "adapters.llava_onevision_dense:LlavaVisionTokenController",
            "sp_apply": "skills.structured_pruning:apply_recipe",
            "quantization_export": "skills.quantization_adapters:export_candidate",
            "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
            "stack_load": "skills.stack_model:load_stack_model",
            "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
            "lmms_native": "search.lmms_llava_stack_entry:main",
            "lmms_vllm": "search.lmms_llava_vllm_entry:install_bridge",
        }
    )

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("model_type") not in self.model_types:
            raise AdapterError("LLaVA-OneVision adapter received a different model_type")
        text = checkpoint.get("text_config")
        vision = checkpoint.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise AdapterError("LLaVA-OneVision checkpoint is missing text_config or vision_config")
        for key in ("num_hidden_layers", "hidden_size", "intermediate_size"):
            if type(text.get(key)) is not int or text[key] <= 0:
                raise AdapterError(f"LLaVA-OneVision text_config.{key} is invalid")
        for key in ("image_size", "patch_size"):
            if type(vision.get(key)) is not int or vision[key] <= 0:
                raise AdapterError(f"LLaVA-OneVision vision_config.{key} is invalid")
        if pipeline:
            # Full orchestration starts from this Dense checkpoint. Stage checks
            # still admit its physically pruned and SigLIP-padded descendants.
            scopes = (
                (text, {"model_type": "qwen2", "num_hidden_layers": 28,
                        "hidden_size": 3584, "intermediate_size": 18944,
                        "num_attention_heads": 28, "num_key_value_heads": 4}),
                (vision, {"model_type": "siglip_vision_model", "num_hidden_layers": 26,
                          "hidden_size": 1152, "intermediate_size": 4304,
                          "num_attention_heads": 16, "image_size": 384, "patch_size": 14}),
                (checkpoint, {"image_token_index": 151646, "vision_feature_layer": -1,
                              "vision_feature_select_strategy": "full",
                              "vision_aspect_ratio": "anyres_max_9"}),
            )
            if any(type(actual.get(key)) is not type(value) or actual[key] != value
                   for actual, expected in scopes for key, value in expected.items()):
                raise AdapterError("LLaVA-OneVision pipeline requires the registered 7B Qwen2/SigLIP topology")

    def validate_stage(self, stage, checkpoint):
        super().validate_stage(stage, checkpoint)
        self.validate_checkpoint(checkpoint, pipeline=False)

    def describe(self):
        return {
            "id": self.id,
            "model_types": list(self.model_types),
            "pipeline_ids": list(self.pipeline_ids),
            "entrypoints": dict(self.entrypoints),
            "implemented_stages": list(self.stages),
            "hardware_ids": list(self.hardware_ids),
            "scope": {
                "modality": "single_image_anyres",
                "text_architecture": "qwen2",
                "optimization": "dense_token_sp_quantization_stack_runtime_kv_lmms_port",
            },
            "qualification": "The local 7B CUDA path has independent Skill, Stack, Runtime/KV, final LMMS and matched end-to-end evidence. Registration describes implementation; every job still requires matching stage receipts.",
        }
