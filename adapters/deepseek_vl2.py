"""Lazy full DeepSeek-VL2 CUDA pipeline registration; no implied quality pass."""

from types import MappingProxyType

from adapters.base import AdapterError, ModelAdapter


class DeepSeekVL2Adapter(ModelAdapter):
    id = "deepseek_vl2"
    model_types = ("deepseek_vl_v2",)
    hardware_ids = ("cuda",)
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    pipeline_ids = ("deepseek_vl2",)
    generation_includes_prompt = False
    entrypoints = MappingProxyType({
        "dense_load": "adapters.deepseek_vl2_dense:load_model",
        "processor_load": "adapters.deepseek_vl2_dense:load_processor",
        "prepare_inputs": "adapters.deepseek_vl2_dense:prepare_inputs",
        "generation_timer": "adapters.deepseek_vl2_dense:GenerationTimer",
        "image_token_count": "adapters.deepseek_vl2_dense:image_token_count",
        "token_geometry": "skills.deepseek_vl2_token:token_geometry",
        "token_controller": "skills.deepseek_vl2_token:DeepSeekTokenController",
        "sp_apply": "skills.deepseek_vl2_pruning:apply_recipe",
        "quantization_export": "skills.quantization_adapters:export_candidate",
        "stack_export": "skills.stack_adapters:export_pruned_gptqmodel",
        "stack_load": "skills.stack_model:load_stack_model",
        "vllm_stack": "skills.stack_vllm:prepare_vllm_artifact",
        "lmms_native": "search.lmms_deepseek_entry:main",
        "lmms_vllm": "search.lmms_deepseek_entry:main",
    })

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        expected = {
            None: {"model_type": "deepseek_vl_v2", "tile_tag": "2D", "global_view_pos": "head"},
            "language_config": {"model_type": "deepseek_v2", "hidden_size": 2560,
                                "intermediate_size": 12288, "moe_intermediate_size": 1536,
                                "n_routed_experts": 72, "n_shared_experts": 2,
                                "num_experts_per_tok": 6, "first_k_dense_replace": 1,
                                "max_position_embeddings": 4096, "vocab_size": 129280},
            "vision_config": {"patch_size": 14, "width": 1152, "layers": 27},
            "projector_config": {"n_embed": 2560},
        }
        for section, fields in expected.items():
            actual = checkpoint if section is None else checkpoint.get(section, {})
            if not isinstance(actual, dict):
                raise AdapterError("DeepSeek-VL2 requires native nested configs")
            for key, value in fields.items():
                if type(actual.get(key)) is not type(value) or actual[key] != value:
                    raise AdapterError(f"Full DeepSeek-VL2 topology differs: {section or 'config'}.{key}")
        text = checkpoint["language_config"]
        if text.get("num_hidden_layers", 30) != 30 or text.get("use_mla", True) is not True:
            raise AdapterError("DeepSeek-VL2 requires the native 30-layer MLA decoder")

    def describe(self):
        return {"id": self.id, "model_types": list(self.model_types),
                "pipeline_ids": list(self.pipeline_ids), "hardware_ids": list(self.hardware_ids),
                "implemented_stages": list(self.stages), "entrypoints": dict(self.entrypoints),
                "scope": {"modality": "single_image_dynamic_tiles", "text_architecture": "deepseek_v2_moe_mla"},
                "qualification": "The full native CUDA pipeline has complete local evidence through final serving LMMS, image inference and matched original Dense timing. Registration enables orchestration; every job still requires its own matching model/hardware evidence and quality gates."}
