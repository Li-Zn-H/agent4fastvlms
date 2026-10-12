"""Lazy native InternVL3.5-14B pipeline registration; no implied quality pass."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class InternVL35Adapter(ModelAdapter):
    id = "internvl35"
    model_types = ("internvl_chat",)
    pipeline_ids = ("internvl35",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    hardware_ids = ("cuda",)
    generation_includes_prompt = False
    entrypoints = MappingProxyType({
        "dense_load": "adapters.internvl35_dense:load_model",
        "processor_load": "adapters.internvl35_dense:load_processor",
        "prepare_inputs": "adapters.internvl35_dense:prepare_inputs",
        "generation_timer": "adapters.internvl35_dense:GenerationTimer",
        "image_token_count": "adapters.internvl35_dense:image_token_count",
        "token_geometry": "adapters.internvl35_dense:token_geometry",
        "token_controller": "adapters.internvl35_dense:InternVLTokenController",
        "sp_apply": "skills.structured_pruning:apply_recipe",
        "quantization_export": "skills.internvl35_quantization:export_multimodal_gptq",
        "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
        "stack_load": "skills.stack_model:load_stack_model",
        "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
        "lmms_native": "search.lmms_internvl_entry:main",
        "lmms_vllm": "search.lmms_internvl_vllm_entry:install_bridge",
    })

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("a4fvlm_stack_widths") is not None or checkpoint.get("a4fvlm_internvl_gptq") is not None:
            from skills.internvl35_artifact import validate_artifact_config
            try:
                validate_artifact_config(checkpoint)
            except (ValueError, KeyError, TypeError) as error:
                raise AdapterError(str(error)) from error
            checkpoint = {**checkpoint, "architectures": ["InternVLChatModel"]}
        expected = {
            None: {"model_type": "internvl_chat", "template": "internvl2_5",
                   "force_image_size": 448, "downsample_ratio": 0.5,
                   "select_layer": -1, "ps_version": "v2",
                   "dynamic_image_size": True, "min_dynamic_patch": 1,
                   "max_dynamic_patch": 12, "use_thumbnail": True},
            "llm_config": {"model_type": "qwen3", "num_hidden_layers": 40,
                           "hidden_size": 5120, "intermediate_size": 17408,
                           "num_attention_heads": 40, "num_key_value_heads": 8,
                           "head_dim": 128},
            "vision_config": {"hidden_size": 1024, "intermediate_size": 4096,
                              "num_hidden_layers": 24, "num_attention_heads": 16,
                              "image_size": 448, "patch_size": 14},
        }
        for section, fields in expected.items():
            actual = checkpoint if section is None else checkpoint.get(section, {})
            if not isinstance(actual, dict):
                raise AdapterError("InternVL3.5 requires native nested configs")
            for key, value in fields.items():
                if type(actual.get(key)) is not type(value) or actual.get(key) != value:
                    raise AdapterError(f"InternVL3.5-14B topology differs: {section or 'config'}.{key}")
        if checkpoint.get("architectures") != ["InternVLChatModel"] or checkpoint["llm_config"].get(
            "architectures"
        ) != ["Qwen3ForCausalLM"]:
            raise AdapterError("InternVL3.5 requires native InternVLChatModel with Qwen3")

    def describe(self):
        return {"id": self.id, "model_types": list(self.model_types),
                "pipeline_ids": list(self.pipeline_ids), "entrypoints": dict(self.entrypoints),
                "implemented_stages": list(self.stages), "hardware_ids": list(self.hardware_ids),
                "scope": {"modality": "single_image_dynamic_tiles", "tile_size": 448,
                          "visual_tokens_per_tile": 256, "maximum_tiles_including_thumbnail": 13,
                          "token_selection": "all_tiles_including_thumbnail", "text_architecture": "qwen3"},
                "qualification": "The native CUDA pipeline has complete local execution evidence through final LMMS, image inference and matched Dense timing. Registration enables orchestration, not quality qualification; the selected reference remains below the 95% target."}
