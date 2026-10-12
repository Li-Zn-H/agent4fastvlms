"""Lazy native InternVL vLLM topology, native tiles and per-tile selection."""
import hashlib
import json
import os
from pathlib import Path

from skills.vllm_token import configured_runtime

_CLASSES = None


def tile_keep_count(runtime=None):
    from skills.vision_token import keep_count
    return 256 if runtime is None else keep_count(16, 16, runtime["keep_ratio"], runtime["selection"]["coverage_grid"])


def select_tiles(features, runtime):
    """Select every native tile independently, then retain original order."""
    import torch
    from skills.vision_token import select_indices

    if features.ndim != 2 or features.shape[0] % 256:
        raise ValueError("InternVL features must contain complete native 16x16 tiles")
    selected = [select_indices(tile, 16, 16, runtime["keep_ratio"], runtime["selection"]) + index * 256
                for index, tile in enumerate(features.reshape(-1, 256, features.shape[-1]))]
    return torch.cat(selected)


def processor_classes():
    """Preserve pinned native pixels and prompt expansion inside vLLM."""
    from adapters.internvl35_processing import load_image, native_grid
    from vllm.transformers_utils.processors.internvl import InternVLImageProcessor, InternVLProcessor
    from vllm.model_executor.models.internvl import InternVLProcessingInfo
    from vllm.multimodal.processing import PromptUpdateDetails

    class NativeImageProcessor(InternVLImageProcessor):
        def _images_to_pixel_values_lst(self, images, min_dynamic_patch=None,
                                        max_dynamic_patch=None, dynamic_image_size=None):
            if (self.image_size != 448 or self.min_dynamic_patch != 1 or self.max_dynamic_patch != 12
                    or not self.dynamic_image_size or not self.use_thumbnail
                    or min_dynamic_patch not in (None, 1) or max_dynamic_patch not in (None, 12)
                    or dynamic_image_size not in (None, True)):
                raise ValueError("InternVL vLLM must retain its native 448/12+thumbnail image protocol")
            return [load_image(image.convert("RGB")) for image in images]

    class NativeProcessor(InternVLProcessor):
        def get_num_image_tokens(self, *, image_width, image_height):
            columns, rows = native_grid(image_width, image_height)
            tiles = columns * rows
            return (tiles + int(tiles > 1)) * tile_keep_count(configured_runtime())

        def get_image_repl(self, num_patches, num_features=None):
            if num_patches is None:
                raise ValueError("InternVL vLLM requires actual native image pixels")
            count = int(num_patches) * tile_keep_count(configured_runtime())
            full = self.start_image_token + self.ctx_image_token * count + self.end_image_token
            return PromptUpdateDetails.select_text(full, self.ctx_image_token)

    class NativeProcessingInfo(InternVLProcessingInfo):
        @property
        def ctx_video_token(self):
            return None

        def get_supported_mm_limits(self):
            return {"image": 1}

        def get_image_processor(self, **kwargs):
            original = super().get_image_processor(**kwargs)
            return NativeImageProcessor(image_size=original.image_size,
                                        min_dynamic_patch=original.min_dynamic_patch,
                                        max_dynamic_patch=original.max_dynamic_patch,
                                        dynamic_image_size=original.dynamic_image_size,
                                        use_thumbnail=original.use_thumbnail)

        def get_hf_processor(self, **kwargs):
            return NativeProcessor(image_processor=self.get_image_processor(**kwargs),
                                   tokenizer=self.get_tokenizer(), image_seq_length=256)

    return NativeImageProcessor, NativeProcessor, NativeProcessingInfo


def create_model_class():
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from adapters import select_model
    from vllm.model_executor.models.internvl import InternVLChatModel, InternVLMultiModalProcessor, InternVLDummyInputsBuilder
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.multimodal import MULTIMODAL_REGISTRY

    _, _, ProcessingInfo = processor_classes()

    @MULTIMODAL_REGISTRY.register_processor(
        InternVLMultiModalProcessor, info=ProcessingInfo, dummy_inputs=InternVLDummyInputsBuilder,
    )
    class A4FVLMInternVLChatModel(InternVLChatModel):
        # Encoder graph replay bypasses _process_vision_input and its measured
        # selector. Reject that unsupported mode explicitly; decoder graph and
        # compilation controls remain owned by Runtime.
        supports_encoder_cudagraph = False
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                                  "gate_up_proj": ["gate_proj", "up_proj"], "qkv": ["qkv"]}

        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.model_config.hf_config
            header = json.loads((Path(vllm_config.model_config.model) / "config.json").read_text())
            select_model(header)
            if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
                raise ValueError("InternVL serving currently requires its single-GPU topology")
            if vllm_config.compilation_config.cudagraph_mm_encoder:
                raise ValueError("InternVL Token selection has no validated encoder CUDA Graph replay")
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            runtime = configured_runtime()
            self.a4fvlm_token_runtime = runtime
            self.a4fvlm_token_pruning = bool(runtime and runtime["keep_ratio"] < 1)
            widths = header.get("a4fvlm_stack_widths")
            layers = self.language_model.model.layers
            if widths is not None:
                quant = getattr(self.language_model, "quant_config", None)
                for index, (layer, width) in enumerate(zip(layers, widths, strict=True)):
                    layer.mlp = Qwen2MLP(hidden_size=config.text_config.hidden_size,
                                        intermediate_size=width, hidden_act=config.text_config.hidden_act,
                                        quant_config=quant,
                                        prefix=(prefix + "." if prefix else "") + f"language_model.model.layers.{index}.mlp")
            receipt = os.environ.get("A4FVLM_VLLM_MODEL_RECEIPT")
            if receipt:
                Path(receipt).write_text(json.dumps({
                    "attention_backends": sorted({layer.self_attn.attn.attn_backend.get_name() for layer in layers}),
                    "kv_cache_dtypes": sorted({layer.self_attn.attn.kv_cache_dtype for layer in layers}),
                    "physical_widths": [layer.mlp.gate_up_proj.output_size // 2 for layer in layers],
                    "token_runtime": runtime, "token_pruning_enabled": self.a4fvlm_token_pruning,
                }) + "\n")

        def _process_vision_input(self, image_input):
            if image_input["type"] != "pixel_values":
                raise ValueError("InternVL optimization requires native image pixels")
            outputs = super()._process_vision_input(image_input)
            if not self.a4fvlm_token_pruning:
                return outputs
            selected_outputs = []
            for features in outputs:
                indices = select_tiles(features, self.a4fvlm_token_runtime)
                selected = features.index_select(0, indices)
                trace = os.environ.get("A4FVLM_VLLM_SELECTION_TRACE")
                if trace:
                    with Path(trace).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "method": "vllm_score_based_post_encoder",
                            "token_keep_ratio": self.a4fvlm_token_runtime["keep_ratio"],
                            "image_tokens_before": int(features.shape[0]),
                            "image_tokens_after": int(selected.shape[0]),
                            "tiles_selected": int(features.shape[0]) // 256,
                            "position_type": "1d_compacted", "positions_after": int(selected.shape[0]),
                            "selected_indices_sha256": hashlib.sha256(json.dumps(indices.tolist()).encode()).hexdigest(),
                        }) + "\n")
                selected_outputs.append(selected)
            return tuple(selected_outputs)

        def get_num_mm_encoder_tokens(self, num_image_tokens):
            return num_image_tokens // tile_keep_count(self.a4fvlm_token_runtime) * (self.patch_tokens + 1)

        def get_num_mm_connector_tokens(self, num_vision_tokens):
            return num_vision_tokens // (self.patch_tokens + 1) * tile_keep_count(self.a4fvlm_token_runtime)

    _CLASSES = A4FVLMInternVLChatModel
    return _CLASSES


def __getattr__(name):
    if name == "A4FVLMInternVLChatModel":
        return create_model_class()
    raise AttributeError(name)
