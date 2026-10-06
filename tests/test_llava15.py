"""LLaVA-1.5 admission and native fixed-resolution processor contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from adapters import AdapterError, models, select_model
from adapters.llava15_dense import token_geometry


class Llava15RegistrationTests(unittest.TestCase):
    def checkpoint(self):
        return {"model_type": "llava", "text_config": {"model_type": "llama"},
                "vision_config": {"model_type": "clip_vision_model", "image_size": 336,
                                  "patch_size": 14, "hidden_size": 1024,
                                  "intermediate_size": 4096, "num_hidden_layers": 24},
                "image_token_index": 32000, "vision_feature_layer": -2,
                "vision_feature_select_strategy": "default"}

    def test_native_hf_defaults_are_admitted_for_the_model_specific_pipeline(self):
        checkpoint = self.checkpoint()
        adapter = select_model(checkpoint)
        self.assertEqual(adapter.id, "llava15")
        adapter.validate_stage("dense", checkpoint)
        adapter.validate_stage("token", checkpoint)
        self.assertIs(select_model(checkpoint, pipeline_id="llava15"), adapter)
        with self.assertRaisesRegex(AdapterError, "not integrated"):
            select_model(checkpoint, pipeline_id="llava_onevision")
        adapter.validate_stage("quantization", checkpoint)

    def test_incompatible_text_vision_and_feature_geometry_are_rejected(self):
        for section, key, value in (("text_config", "hidden_size", 5120),
                                    ("text_config", "num_hidden_layers", True),
                                    ("vision_config", "image_size", 384),
                                    ("vision_config", "model_type", "siglip_vision_model"),
                                    (None, "vision_feature_select_strategy", "full")):
            with self.subTest(key=key):
                checkpoint = copy.deepcopy(self.checkpoint())
                (checkpoint[section] if section else checkpoint)[key] = value
                with self.assertRaises(AdapterError):
                    models.get("llava15").validate_checkpoint(checkpoint)

    def test_fixed_geometry_rejects_tiled_pixels_or_wrong_placeholder_counts(self):
        class IDs:
            def __init__(self, count): self.count = count
            def __eq__(self, other): return self
            def sum(self): return self.count

        processor = SimpleNamespace(patch_size=14, image_token_id=32000)
        inputs = {"pixel_values": SimpleNamespace(shape=(1, 3, 336, 336)), "input_ids": IDs(576)}
        self.assertEqual(token_geometry(processor, inputs)["rows"], 24)
        for shape, count in (((1, 2, 3, 336, 336), 576), ((1, 3, 336, 336), 577)):
            with self.assertRaises(ValueError):
                token_geometry(processor, {"pixel_values": SimpleNamespace(shape=shape), "input_ids": IDs(count)})

    def test_clip_selection_covers_all_spatial_cells_and_matches_prompt_budget(self):
        import torch
        from adapters.llava15_dense import Llava15VisionTokenController
        from skills.llava_token import select_clip_indices

        selection = {"score_dtype": "float32", "coverage_grid": {"rows": 6, "columns": 6}}
        features = torch.arange(576, dtype=torch.float32).reshape(576, 1)
        for ratio, expected in ((.1, 58), (.001, 36), (.85, 490), (1., 576)):
            controller = Llava15VisionTokenController(None, selection, ratio)
            indices = controller._indices(features)
            self.assertEqual(indices.numel(), expected)
            self.assertEqual(controller._keep_count(576), expected)
            torch.testing.assert_close(indices, select_clip_indices(features, selection, ratio))
            cells = (indices // 24 // 4) * 6 + (indices % 24 // 4)
            self.assertEqual(cells.unique().numel(), 36)
            self.assertTrue(bool((indices[1:] > indices[:-1]).all()))
        with self.assertRaisesRegex(ValueError, "24x24"):
            select_clip_indices(torch.zeros(577, 1), selection, .85)

    def test_native_lmms_uses_hf_llava_with_explicit_dtype(self):
        from search.lmms import build_command

        model = {"adapter": "llava15", "path": "models/LLaVA-1.5-7B", "device": "cuda:0",
                 "dtype": "float16", "attn_implementation": "sdpa"}
        command = build_command(model, "chartqa", Path("output"), limit=2)
        self.assertEqual(command[command.index("--model") + 1], "llava_hf")
        self.assertIn("dtype=float16", command[command.index("--model_args") + 1])
        stack = build_command({**model, "stack_adapter": True}, "chartqa", Path("output"))
        self.assertTrue(stack[1].endswith("lmms_llava_stack_entry.py"))
        serving = build_command({**model, "stack_adapter": True, "execution_backend": "vllm"}, "chartqa", Path("output"))
        self.assertTrue(serving[1].endswith("lmms_llava_vllm_entry.py"))

    def test_clip_quantization_scope_preserves_projector_and_rejects_padding(self):
        from skills.quantization import assignment, validate_module_inventory
        from skills.quantization_evidence import llava15_gptq_preflight

        names = ["model.language_model.layers.0.mlp.gate_proj",
                 "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                 "model.vision_tower.encoder.layers.0.mlp.fc1",
                 "model.multi_modal_projector.linear_1", "lm_head"]
        precision = assignment(names, model_adapter="llava15")
        self.assertEqual(list(precision.values()).count("int4_w4a16"), 3)
        self.assertEqual(precision["model.multi_modal_projector.linear_1"], "bf16")
        self.assertEqual(precision["lm_head"], "bf16")
        self.assertTrue(validate_module_inventory(names, model_adapter="llava15")["valid"])
        config = self.checkpoint()
        self.assertEqual(llava15_gptq_preflight(config, 128)["status"], "ready_for_probe")
        config["vision_config"]["intermediate_size"] = 4304
        self.assertEqual(llava15_gptq_preflight(config, 128)["status"], "blocked_preflight")

    def test_clip_initializer_only_skips_packed_containers_and_restores_on_failure(self):
        import torch
        from transformers import CLIPVisionConfig
        from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP, CLIPPreTrainedModel
        from adapters.llava15_dense import quantized_clip_initialization

        config = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2)
        attention, mlp = CLIPAttention(config), CLIPMLP(config)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            packed = torch.nn.Module()
            packed.register_buffer("qweight", torch.zeros((2, 2), dtype=torch.int32))
            setattr(attention, name, packed)
        original = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "a4fvlm_clip_gptq": {"method": "clip_gptq_v1"}, "quantization_config": {"bits": 4}}))
            with patch.object(CLIPPreTrainedModel, "_init_weights", original):
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    with quantized_clip_initialization(root):
                        CLIPPreTrainedModel._init_weights(None, attention)
                        original.assert_not_called()
                        CLIPPreTrainedModel._init_weights(None, mlp)
                        original.assert_called_once_with(None, mlp)
                        attention.out_proj = torch.nn.Linear(16, 16)
                        CLIPPreTrainedModel._init_weights(None, attention)
                        self.assertEqual(original.call_count, 2)
                        raise RuntimeError("load failed")
                self.assertIs(CLIPPreTrainedModel._init_weights, original)

    def test_clip_weight_alias_is_narrow_and_idempotent(self):
        from skills.llava15_vllm import normalize_clip_weight_name

        for suffix in ("pre_layrnorm.weight", "embeddings.class_embedding", "encoder.layers.0.mlp.fc1.qweight"):
            raw = "model.vision_tower." + suffix
            expected = "model.vision_tower.vision_model." + suffix
            self.assertEqual(normalize_clip_weight_name(raw), expected)
            self.assertEqual(normalize_clip_weight_name(expected), expected)
        raw = "model.language_model.layers.0.mlp.gate_proj.qweight"
        self.assertEqual(normalize_clip_weight_name(raw), raw)

    def test_physical_llama_topology_matches_each_nonuniform_width(self):
        import torch
        from transformers import LlavaConfig, CLIPVisionConfig, LlamaConfig
        from skills.stack_model import create_pruned_model_class, LLAVA15_STACK_ARCHITECTURE
        from skills.stack_vllm import stack_widths

        config = LlavaConfig(text_config=LlamaConfig(), vision_config=CLIPVisionConfig())
        config.a4fvlm_stack_widths = [11008 - index % 3 * 128 for index in range(32)]
        config.a4fvlm_stack_physical_compaction = True
        config.architectures = [LLAVA15_STACK_ARCHITECTURE]
        self.assertEqual(list(stack_widths(config)), config.a4fvlm_stack_widths)
        with torch.device("meta"):
            model = create_pruned_model_class("llava15")(config)
        for layer, width in zip(model.model.language_model.layers, config.a4fvlm_stack_widths):
            self.assertEqual(tuple(layer.mlp.gate_proj.weight.shape), (width, 4096))
            self.assertEqual(tuple(layer.mlp.up_proj.weight.shape), (width, 4096))
            self.assertEqual(tuple(layer.mlp.down_proj.weight.shape), (4096, width))
        config.a4fvlm_stack_widths[0] = 11007
        with self.assertRaisesRegex(ValueError, "Group-128"):
            stack_widths(config)

    @unittest.skipUnless(Path("models/LLaVA-1.5-7B/processor_config.json").is_file(), "Local HF model required")
    def test_real_native_processor_keeps_fixed_geometry_for_different_aspect_ratios(self):
        from PIL import Image
        from adapters.llava_onevision_dense import load_processor

        processor = load_processor({"path": "models/LLaVA-1.5-7B"})
        for size in ((640, 375), (1152, 1152), (2048, 1536)):
            with self.subTest(size=size), Image.new("RGB", size) as image:
                messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                              {"type": "text", "text": "Describe the image."}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                self.assertTrue(prompt.startswith("USER: <image>\n"))
                self.assertTrue(prompt.endswith("ASSISTANT:"))
                inputs = processor(text=[prompt], images=[image], return_tensors="pt")
                geometry = token_geometry(processor, inputs)
                self.assertEqual((geometry["rows"], geometry["columns"], geometry["patches"]), (24, 24, 1))


if __name__ == "__main__":
    unittest.main()
