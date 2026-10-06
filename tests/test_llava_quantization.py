"""LLaVA FFN alignment must preserve the dense function before quantization."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from skills.llava_quantization import pad_siglip_mlp, padded_width


class LlavaPaddingTests(unittest.TestCase):
    def test_auxiliary_checkpoint_map_supports_single_file_and_shards(self):
        import torch
        from safetensors.torch import save_file
        from skills.quantization_adapters import _llava_checkpoint_weight_map, _llava_checkpoint_prefix

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "model.vision_tower.encoder.layers.0.mlp.fc1.weight"
            save_file({key: torch.ones((2, 2))}, str(root / "model.safetensors"))
            weights = _llava_checkpoint_weight_map(root)
            selected, strip = _llava_checkpoint_prefix(weights, "vision_tower.", "vision_tower.vision_model.")
            self.assertEqual(selected, {key: "model.safetensors"})
            self.assertEqual(key[len(strip):], "encoder.layers.0.mlp.fc1.weight")
            (root / "model.safetensors").rename(root / "model-00001-of-00001.safetensors")
            sharded = {key: "model-00001-of-00001.safetensors"}
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": sharded}))
            self.assertEqual(_llava_checkpoint_weight_map(root), sharded)
            (root / "model.safetensors.index.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                _llava_checkpoint_weight_map(root)

    def test_siglip_initialization_skips_only_fully_packed_containers_and_restores(self):
        import torch
        from transformers import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipMLP, SiglipPreTrainedModel
        from adapters.llava_onevision_dense import quantized_siglip_initialization

        config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2)
        attention, mlp = SiglipAttention(config), SiglipMLP(config)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            packed = torch.nn.Module()
            packed.register_buffer("qweight", torch.zeros((2, 2), dtype=torch.int32))
            setattr(attention, name, packed)
        original = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "a4fvlm_siglip_padding": {"padded_width": 32}, "quantization_config": {"bits": 4}}))
            with patch.object(SiglipPreTrainedModel, "_init_weights", original):
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    with quantized_siglip_initialization(root):
                        SiglipPreTrainedModel._init_weights(None, attention)
                        original.assert_not_called()
                        SiglipPreTrainedModel._init_weights(None, mlp)
                        original.assert_called_once_with(None, mlp)
                        attention.out_proj = torch.nn.Linear(16, 16)
                        SiglipPreTrainedModel._init_weights(None, attention)
                        self.assertEqual(original.call_count, 2)
                        raise RuntimeError("load failure")
                self.assertIs(SiglipPreTrainedModel._init_weights, original)

    def test_auxiliary_keys_load_both_original_and_physical_parent(self):
        from skills.quantization_adapters import _llava_checkpoint_prefix

        cases = [
            ("vision_tower.", "vision_tower.vision_model.",
             "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
             "model.vision_tower.encoder.layers.0.mlp.fc1.weight", "encoder.layers.0.mlp.fc1.weight"),
            ("multi_modal_projector.", "multi_modal_projector.",
             "multi_modal_projector.linear_1.weight", "model.multi_modal_projector.linear_1.weight", "linear_1.weight"),
            ("image_newline", "", "image_newline", "model.image_newline", "image_newline"),
            ("language_model.model.embed_tokens.", "language_model.model.embed_tokens.",
             "language_model.model.embed_tokens.weight", "model.language_model.embed_tokens.weight", "weight"),
            ("language_model.model.norm.", "language_model.model.norm.",
             "language_model.model.norm.weight", "model.language_model.norm.weight", "weight"),
            ("language_model.model.layers.3.", "language_model.model.layers.3.",
             "language_model.model.layers.3.mlp.up_proj.weight", "model.language_model.layers.3.mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("language_model.lm_head.", "language_model.lm_head.",
             "language_model.lm_head.weight", "lm_head.weight", "weight"),
        ]
        for prefix, strip, original, physical, local in cases:
            for key in (original, physical):
                with self.subTest(key=key):
                    selected, resolved_strip = _llava_checkpoint_prefix({key: "shard", "unrelated": "other"}, prefix, strip)
                    self.assertEqual(selected, {key: "shard"})
                    self.assertEqual(key[len(resolved_strip):], local)
        with self.assertRaisesRegex(RuntimeError, "no vision_tower"):
            _llava_checkpoint_prefix({}, "vision_tower.", "vision_tower.vision_model.")

    def test_vllm_siglip_mapping_is_narrow_and_idempotent(self):
        from skills.llava_vllm import normalize_siglip_weight_name

        for suffix in ("embeddings.patch_embedding.weight", "encoder.layers.0.mlp.fc1.qweight"):
            name = "model.vision_tower." + suffix
            mapped = "model.vision_tower.vision_model." + suffix
            self.assertEqual(normalize_siglip_weight_name(name), mapped)
            self.assertEqual(normalize_siglip_weight_name(mapped), mapped)
        for name in ("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                     "model.language_model.layers.0.mlp.gate_proj.qweight",
                     "model.visual.blocks.0.mlp.linear_fc1.weight"):
            self.assertEqual(normalize_siglip_weight_name(name), name)

    def test_zero_extension_preserves_function_and_bias(self):
        import torch

        for bias in (False, True):
            with self.subTest(bias=bias):
                torch.manual_seed(7)
                class Mlp(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = torch.nn.Linear(16, 19, bias=bias, dtype=torch.float64)
                        self.fc2 = torch.nn.Linear(19, 16, bias=bias, dtype=torch.float64)

                    def forward(self, x):
                        return self.fc2(torch.nn.functional.gelu(self.fc1(x), approximate="tanh"))

                mlp = Mlp()
                original = copy.deepcopy(mlp)
                x = torch.randn(3, 9, 16, dtype=torch.float64)
                receipt = pad_siglip_mlp(mlp, 32)
                self.assertEqual(receipt["added_channels"], 13)
                torch.testing.assert_close(mlp(x), original(x), rtol=1e-12, atol=1e-12)
                self.assertEqual(torch.count_nonzero(mlp.fc1.weight[19:]).item(), 0)
                self.assertEqual(torch.count_nonzero(mlp.fc2.weight[:, 19:]).item(), 0)
                if bias:
                    self.assertEqual(torch.count_nonzero(mlp.fc1.bias[19:]).item(), 0)
                self.assertEqual(pad_siglip_mlp(mlp, 32)["added_channels"], 0)
                with self.assertRaises(ValueError):
                    pad_siglip_mlp(mlp, 19)

    def test_native_siglip_width_gets_group_and_kernel_alignment(self):
        self.assertEqual(padded_width(4304), 4352)
        self.assertEqual(padded_width(4352), 4352)
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                padded_width(width)


if __name__ == "__main__":
    unittest.main()
