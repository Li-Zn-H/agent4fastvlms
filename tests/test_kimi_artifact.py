"""Physical Kimi export/reload preserves cached outputs and rejects corruption."""

import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Kimi-VL-A3B-Instruct"
try:
    import torch
    import transformers
    NATIVE = transformers.__version__ == "4.50.3"
except ImportError:
    NATIVE = False


@unittest.skipUnless(NATIVE and (MODEL / "config.json").is_file(), "native Kimi environment required")
class ParentTests(unittest.TestCase):
    def setUp(self):
        from transformers import AutoConfig
        from adapters.kimi_vl_dense import load_processor, native_model_class
        from skills.kimi_vl_pruning import dimensions, units, make_recipe, apply_recipe

        torch.manual_seed(19)
        self.descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        self.processor = load_processor(self.descriptor)
        config = AutoConfig.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
        for key, value in {"vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
                           "moe_intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
                           "num_key_value_heads": 4, "n_routed_experts": 4, "n_shared_experts": 1,
                           "num_experts_per_tok": 2, "kv_lora_rank": 16, "qk_rope_head_dim": 8,
                           "qk_nope_head_dim": 16, "v_head_dim": 16, "bos_token_id": 1,
                           "eos_token_id": 2, "pad_token_id": 0}.items():
            setattr(config.text_config, key, value)
        for key, value in {"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 4,
                           "num_hidden_layers": 1, "init_pos_emb_height": 2, "init_pos_emb_width": 2}.items():
            setattr(config.vision_config, key, value)
        config.media_placeholder_token_id = 100
        config.vocab_size = 256
        config._attn_implementation = "eager"
        cls = native_model_class(self.descriptor)
        self.model = cls(config).to(torch.bfloat16).eval()
        self.model.language_model.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        dims = dimensions(config.text_config)
        domain = units(dims)
        scores = [[[float(i + 1) for i in range(unit["original_width"])]
                   for _ in range(unit["experts"])] for unit in domain]
        self.recipe = make_recipe(scores, [unit["original_width"] // 2 for unit in domain], dims,
                                  {"channel_group_size": 16, "protected_layers": []})
        apply_recipe(self.model, self.recipe)
        ids = torch.tensor([[1, 100, 25]])
        self.inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                       "pixel_values": torch.randn(4, 3, 14, 14, dtype=torch.bfloat16),
                       "image_grid_hws": torch.tensor([[2, 2]])}
        self.options = {"do_sample": False, "max_new_tokens": 3, "eos_token_id": None,
                        "pad_token_id": 0, "use_cache": True}

    def test_round_trip_preserves_physical_weights_routing_and_cached_outputs(self):
        from skills.kimi_vl_artifact import write_parent, load_artifact
        from skills.kimi_vl_pruning import audit_recipe
        with torch.inference_mode():
            expected = self.model.generate(**self.inputs, **self.options)
            logits = self.model(**self.inputs).logits
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parent"
            manifest = write_parent(self.model, self.processor, self.recipe, output)
            descriptor = {**self.descriptor, "path": str(output), "device": "cpu"}
            restored, processor = load_artifact(descriptor)
            with torch.inference_mode():
                observed = restored.generate(**self.inputs, **self.options)
                actual_logits = restored(**self.inputs).logits
            self.assertTrue(torch.equal(expected, observed))
            self.assertTrue(torch.equal(logits, actual_logits))
            self.assertEqual(audit_recipe(restored, self.recipe), manifest["mechanism"])
            self.assertEqual(restored.generation_config.eos_token_id, [163585])
            self.assertEqual(processor.chat_template, self.processor.chat_template)
            self.assertEqual(len(restored.language_model.model.layers[1].mlp.experts), 4)
            for name in self.descriptor["native_source_sha256"]:
                self.assertEqual((output / name).read_bytes(), (MODEL / name).read_bytes())
            for name, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_missing_weights_and_changed_router_precision_are_rejected(self):
        from safetensors.torch import load_file, save_file
        from skills.kimi_vl_artifact import write_parent, load_physical_model
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parent"
            write_parent(self.model, self.processor, self.recipe, output)
            descriptor = {**self.descriptor, "path": str(output), "device": "cpu"}
            shard = next(output.glob("*.safetensors"))
            state = load_file(str(shard))
            key = "language_model.model.layers.1.mlp.gate.e_score_correction_bias"
            original = state[key]
            state[key] = original.float()
            save_file(state, str(shard))
            with self.assertRaisesRegex(ValueError, "router shape or native precision"):
                load_physical_model(descriptor)
            state[key] = original
            del state["language_model.model.layers.1.mlp.experts.0.down_proj.weight"]
            save_file(state, str(shard))
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_physical_model(descriptor)

    def test_config_and_recipe_tampering_are_rejected_before_reload(self):
        from skills.kimi_vl_artifact import write_parent, read_parent
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parent"
            write_parent(self.model, self.processor, self.recipe, output)
            header_path = output / "config.json"
            header = json.loads(header_path.read_text())
            header["text_config"]["num_experts_per_tok"] = 1
            header_path.write_text(json.dumps(header))
            with self.assertRaises(ValueError):
                read_parent(output)

    def test_quantization_inventory_and_internal_vision_padding_preserve_native_features(self):
        from skills.kimi_vl_quantization import expected_inventory, high_precision_exceptions, pad_vision
        header = self.model.config.to_dict()
        policy = expected_inventory(header)
        names = {name for name, module in self.model.named_modules() if isinstance(module, torch.nn.Linear)}
        self.assertEqual(set(policy), names)
        self.assertEqual(policy["language_model.model.layers.1.self_attn.kv_b_proj"], "int4_w4a16")
        self.assertEqual(policy["multi_modal_projector.linear_1"], "bf16")
        correction = "language_model.model.layers.1.mlp.gate.e_score_correction_bias"
        self.assertEqual(high_precision_exceptions(header)[correction]["dtype"], "bfloat16")
        with torch.inference_mode():
            expected = self.model._extract_image_features(self.inputs["pixel_values"], self.inputs["image_grid_hws"])
            padding = pad_vision(self.model)
            observed = self.model._extract_image_features(self.inputs["pixel_values"], self.inputs["image_grid_hws"])
        self.assertEqual(padding["original_width"], 64)
        self.assertEqual(padding["padded_width"], 128)
        for a, b in zip(expected, observed, strict=True):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.inputs["image_grid_hws"].tolist(), [[2, 2]])
        with self.assertRaisesRegex(ValueError, "fresh native"):
            pad_vision(self.model)

    def test_packed_vision_capture_replays_native_positional_sequence_offsets(self):
        from skills.kimi_vl_quantization import capture_inputs
        block = self.model.vision_tower.encoder.blocks[0]
        hooks = len(block._forward_pre_hooks)
        with torch.inference_mode():
            rows = capture_inputs(block, [0], lambda _: self.model._extract_image_features(
                self.inputs["pixel_values"], self.inputs["image_grid_hws"]), vision=True)
            hidden, kwargs = rows[0]
            self.assertIn("cu_seqlens", kwargs)
            self.assertEqual(kwargs["cu_seqlens"].tolist(), [0, 4])
            replay = block(hidden, **kwargs)
        self.assertEqual(replay.shape, hidden.shape)
        self.assertTrue(torch.isfinite(replay).all())
        self.assertEqual(len(block._forward_pre_hooks), hooks)

    def test_uninitialized_derived_rotary_buffers_can_be_rebuilt_without_changing_weights(self):
        from adapters.kimi_vl_compat import restore_native_rotary_buffers
        parameters = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        with torch.inference_mode():
            expected = self.model.generate(**self.inputs, **self.options)
            for layer in self.model.language_model.model.layers:
                layer.self_attn.rotary_emb.inv_freq.fill_(float('nan'))
            restore_native_rotary_buffers(self.model)
            observed = self.model.generate(**self.inputs, **self.options)
        self.assertTrue(torch.equal(expected, observed))
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]), name)
