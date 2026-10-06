"""Register the existing Qwen3-VL implementation without importing Torch or vLLM."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class Qwen3VLAdapter(ModelAdapter):
    id = "qwen3_vl"
    model_types = ("qwen3_vl",)
    pipeline_ids = ("qwen3vl",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    hardware_ids = ("cuda",)
    # The current physical vLLM Stack implementation has this exact topology.
    # This is an implementation boundary, not a default for future models.
    pipeline_dimensions = MappingProxyType({
        "num_hidden_layers": 36, "hidden_size": 2560, "intermediate_size": 9728,
    })
    entrypoints = MappingProxyType({
        "dense_load": "adapters.qwen3vl_dense:load_model",
        "processor_load": "adapters.qwen3vl_dense:load_processor",
        "prepare_inputs": "adapters.qwen3vl_dense:prepare_inputs",
        "generation_timer": "adapters.qwen3vl_dense:GenerationTimer",
        "image_token_count": "adapters.qwen3vl_dense:image_token_count",
        "token_geometry": "adapters.qwen3vl_dense:token_geometry",
        "token_controller": "skills.vision_token:VisionTokenController",
        "sp_apply": "skills.structured_pruning:apply_recipe",
        "quantization_export": "skills.quantization_adapters:export_candidate",
        "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
        "stack_load": "skills.stack_model:load_stack_model",
        "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
        "lmms_native": "search.lmms_stack_entry:main",
        "lmms_vllm": "search.lmms_vllm_entry:install_bridge",
    })

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("model_type") not in self.model_types:
            raise AdapterError("Qwen3-VL adapter received a different model_type")
        if pipeline:
            text = checkpoint.get("text_config")
            if not isinstance(text, dict) or any(
                type(text.get(key)) is not int or text[key] != value
                for key, value in self.pipeline_dimensions.items()
            ):
                raise AdapterError("Qwen3-VL pipeline currently requires the registered 4B topology; port the Stack adapter first")

    def validate_stage(self, stage, checkpoint):
        super().validate_stage(stage, checkpoint)
        # Other sizes can establish Dense first, but optimized paths need a port.
        self.validate_checkpoint(checkpoint, pipeline=stage != "dense")

    def describe(self):
        return {
            "id": self.id, "model_types": list(self.model_types),
            "pipeline_ids": list(self.pipeline_ids), "entrypoints": dict(self.entrypoints),
            "implemented_stages": list(self.stages), "hardware_ids": list(self.hardware_ids),
            "scope": {"modality": "single_image", "pipeline_dimensions": dict(self.pipeline_dimensions),
                      "stack_route": "gptqmodel_w4a16", "stack_group_size": 128},
            "qualification": "Registration describes implementation; each job still requires its own evidence.",
        }
