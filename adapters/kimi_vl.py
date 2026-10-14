"""Lazy Kimi-VL-A3B-Instruct registration; receipts qualify each stage."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class KimiVLAdapter(ModelAdapter):
    id = "kimi_vl"
    model_types = ("kimi_vl",)
    hardware_ids = ("cuda",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    pipeline_ids = ()
    entrypoints = MappingProxyType({
        "dense_load": "adapters.kimi_vl_dense:load_model",
        "processor_load": "adapters.kimi_vl_dense:load_processor",
        "prepare_inputs": "adapters.kimi_vl_dense:prepare_inputs",
        "generation_timer": "adapters.kimi_vl_dense:GenerationTimer",
        "image_token_count": "adapters.kimi_vl_dense:image_token_count",
        "token_geometry": "skills.kimi_vl_token:token_geometry",
        "token_controller": "skills.kimi_vl_token:KimiTokenController",
        "sp_apply": "skills.kimi_vl_pruning:apply_recipe",
        "quantization_export": "skills.quantization_adapters:export_candidate",
        "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
        "stack_load": "skills.stack_model:load_stack_model",
        "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
        "lmms_native": "search.lmms_kimi_entry:main",
        "lmms_vllm": "search.lmms_kimi_entry:main",
    })

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("model_type") != self.id:
            raise AdapterError("Kimi-VL requires its native checkpoint")
        expected = {
            "text_config": {"num_hidden_layers": 27, "hidden_size": 2048,
                            "intermediate_size": 11264, "moe_intermediate_size": 1408,
                            "n_routed_experts": 64, "n_shared_experts": 2,
                            "num_experts_per_tok": 6, "first_k_dense_replace": 1,
                            "topk_method": "noaux_tc", "scoring_func": "sigmoid",
                            "max_position_embeddings": 131072, "vocab_size": 163840},
            "vision_config": {"patch_size": 14, "hidden_size": 1152,
                              "num_hidden_layers": 27, "merge_kernel_size": [2, 2]},
        }
        for section, fields in expected.items():
            actual = checkpoint.get(section, {})
            for name, value in fields.items():
                if type(actual.get(name)) is not type(value) or actual[name] != value:
                    raise AdapterError(f"Kimi native topology differs: {section}.{name}")
        if checkpoint.get("media_placeholder_token_id") != 163605:
            raise AdapterError("Kimi native image token differs")
        if pipeline:
            raise AdapterError("Kimi full pipeline is under validation")

    def describe(self):
        return {"id": self.id, "model_types": list(self.model_types),
                "pipeline_ids": list(self.pipeline_ids), "hardware_ids": list(self.hardware_ids),
                "implemented_stages": list(self.stages), "entrypoints": dict(self.entrypoints),
                "scope": {"modality": "single_image_native_resolution", "protocol": "instruct",
                          "text_architecture": "deepseek_v3_moe_mla"},
                "qualification": "Individual stage interfaces are ready for probing; CUDA reload, serving and quality require matching receipts. The complete pipeline remains under validation."}
