"""CPU calibration/packing checks; solver tests use the isolated quantizer."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import unittest


class CaptureTests(unittest.TestCase):
    def test_native_capture_clears_cache_and_always_removes_its_hook(self):
        import torch
        from skills.internvl35_quantization import capture_inputs

        class Block(torch.nn.Module):
            def forward(self, hidden, **kwargs):
                raise AssertionError("The first block must not execute during capture")
        layer = Block()
        inputs = [torch.randn(1, 5, 128), torch.randn(1, 7, 128)]
        rows = capture_inputs(layer, inputs, lambda value: layer(value, past_key_values=object(), use_cache=True))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], {"past_key_values": None, "use_cache": False})
        torch.testing.assert_close(rows[1][0], inputs[1], rtol=0, atol=0)
        self.assertFalse(layer._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, "input error"):
            capture_inputs(layer, inputs, lambda value: (_ for _ in ()).throw(RuntimeError("input error")))
        self.assertFalse(layer._forward_pre_hooks)

    def test_unpack_rejects_wrong_zero_convention(self):
        import torch
        from skills.internvl35_quantization import unpack_gptq

        tensors = {"qweight": torch.zeros(16, 128, dtype=torch.int32),
                   "qzeros": torch.full((1, 16), 0x77777777, dtype=torch.int32),
                   "g_idx": torch.zeros(128, dtype=torch.int32),
                   "scales": torch.ones(1, 128, dtype=torch.bfloat16)}
        torch.testing.assert_close(unpack_gptq(tensors), torch.full((128, 128), -8.0), rtol=0, atol=0)
        tensors["qzeros"][0, 0] = 0x66666666
        with self.assertRaisesRegex(ValueError, "zero convention"):
            unpack_gptq(tensors)


@unittest.skipUnless(importlib.util.find_spec("gptqmodel"), "isolated GPTQModel interpreter required")
class SolverTests(unittest.TestCase):
    def test_stack_export_releases_parent_after_first_backend_import(self):
        import os
        import subprocess
        import sys
        import textwrap

        # A fresh interpreter is essential: importing GPTQModel after creating
        # the parent can retain that parent through dependency import frames.
        script = textwrap.dedent('''
            import gc, json, tempfile, weakref
            from collections import OrderedDict
            from pathlib import Path
            import torch
            import skills.internvl35_quantization as quant
            from skills.stack_adapters import export_pruned_gptqmodel

            torch.set_num_threads(2)
            references = []
            def export(**kwargs):
                parent = torch.nn.Sequential(OrderedDict([
                    ("linear", torch.nn.Linear(128, 128, bias=False)),
                ])).to(torch.bfloat16).eval()
                references.append(weakref.ref(parent))
                packed, reports = {}, []
                rows = [(torch.randn(1, 32, 128, dtype=torch.bfloat16), {})]
                with torch.inference_mode():
                    quant.quantize_blocks([parent], "parent", rows,
                                          (("linear",),), packed, reports)
                assert len(packed) == len(reports) == 1
                return {"manifest": "fixture"}

            quant.export_multimodal_gptq = export
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "stack-parent.json").write_text(json.dumps({"recipe_digest": "recipe"}))
                result = export_pruned_gptqmodel(
                    candidate={"route": "gptqmodel_w4a16"}, parent_dir=root,
                    processor=None, calibration={}, output_dir=root / "output",
                    model={"adapter": "internvl35"}, module_names=[], recipe_digest="recipe")
            assert result == {"manifest": "fixture"}
            gc.collect()
            assert len(references) == 1
            assert references[0]() is None, "Backend import retained the completed physical parent"
        ''')
        result = subprocess.run([sys.executable, "-B", "-c", script], text=True, capture_output=True,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
                                timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bf16_outliers_preserve_exact_codes_and_reject_corrupt_packing(self):
        import torch
        from unittest.mock import patch
        from gptqmodel.nn_modules.qlinear.torch import TorchLinear
        from skills.llava_quantization import new_gptq_task, pack_gptq_task
        prior = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, prior)
        torch.manual_seed(123)
        layer = torch.nn.Linear(128, 128, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            layer.weight.mul_(100)
        inputs = torch.randn(2, 40, 128, dtype=torch.bfloat16)
        original_pack = TorchLinear.pack
        for corrupt in (False, True):
            def pack(module, *args, **kwargs):
                original_pack(module, *args, **kwargs)
                if corrupt:
                    module.qweight[0, 0] ^= 1
            with self.subTest(corrupt=corrupt), torch.inference_mode(), patch.object(TorchLinear, 'pack', pack):
                task = new_gptq_task(layer, bits=4, group_size=128)
                task.add_batch(inputs, layer(inputs))
                if corrupt:
                    with self.assertRaises(AssertionError):
                        pack_gptq_task(task)
                else:
                    tensors, report = pack_gptq_task(task)
                    self.assertEqual(report['packing_validation'], 'exact_codes_zeros_groups_and_fp32_dequantization')
                    self.assertTrue(bool((tensors['qzeros'] == 0x77777777).all()))

    @unittest.skipUnless(Path("models/InternVL3_5-14B/config.json").exists(), "native InternVL code fixture required")
    def test_actual_native_towers_calibrate_physical_widths_and_preserve_mapper(self):
        self.check_native_towers("models/InternVL3_5-14B")

    def check_native_towers(self, model_path):
        import torch
        from transformers import AutoConfig, AutoModel
        from skills.internvl35_quantization import quantize_native_modules, unpack_gptq
        from skills.structured_pruning import apply_recipe, make_recipe

        prior = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, prior)
        torch.manual_seed(812)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        for key, value in {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                           "layer_types": ["full_attention"] * 2, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 128,
                           "eos_token_id": 2, "bos_token_id": 1, "pad_token_id": 0}.items():
            setattr(config.llm_config, key, value)
        for key, value in {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                           "num_attention_heads": 4, "image_size": 112}.items():
            setattr(config.vision_config, key, value)
        config.force_image_size = 112
        native = AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=False).to(torch.bfloat16).eval()
        native.language_model.set_attn_implementation("sdpa")
        native.img_context_token_id = 98
        dimensions = {"num_hidden_layers": 2, "hidden_size": 128, "intermediate_size": 256}
        recipe = make_recipe([list(range(256))] * 2, [128, 256], dimensions,
                             {"channel_group_size": 128, "protected_layers": []})
        apply_recipe(native, recipe)
        preserved = {key: value.clone() for key, value in native.state_dict().items()
                     if key.startswith("mlp1.") or key.startswith("language_model.lm_head.")}
        samples = []
        for tiles in (1, 3):
            ids = torch.tensor([[10] + [98] * (tiles * 16) + [11, 12]])
            samples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                            "pixel_values": torch.randn(tiles, 3, 112, 112, dtype=torch.bfloat16)})
        packed, reports = quantize_native_modules(native, samples, lambda row: dict(row))
        self.assertEqual(len(packed), 22)
        self.assertTrue(all(row["module"].startswith("vision_model.") for row in reports[:8]))
        self.assertTrue(all(row["module"].startswith("language_model.") for row in reports[8:]))
        self.assertTrue(all(row["samples"] > 0 and row["fallback"] is None for row in reports))
        for key, tensors in packed.items():
            torch.testing.assert_close(native.get_submodule(key).weight,
                                       unpack_gptq(tensors).to(torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(packed["language_model.model.layers.0.mlp.down_proj"]["g_idx"].numel(), 128)
        self.assertEqual(packed["language_model.model.layers.1.mlp.down_proj"]["g_idx"].numel(), 256)
        for key, original in preserved.items():
            torch.testing.assert_close(native.state_dict()[key], original, rtol=0, atol=0)
        self.assertTrue(all(not block._forward_pre_hooks for block in
                            [native.vision_model.encoder.layers[0], native.language_model.model.layers[0]]))
        with torch.inference_mode():
            output = native.generate(**samples[-1], max_new_tokens=2, do_sample=False,
                                     eos_token_id=None, pad_token_id=0)
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_sequential_groups_propagate_portable_quantized_weights(self):
        import torch
        from skills.internvl35_quantization import quantize_blocks, unpack_gptq

        prior = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(14)
            block = torch.nn.Sequential(OrderedDict([
                ("first", torch.nn.Linear(128, 256)), ("act", torch.nn.SiLU()),
                ("second", torch.nn.Linear(256, 128)),
            ])).to(torch.bfloat16).eval()
            rows = [(torch.randn(1, 40, 128, dtype=torch.bfloat16), {}) for _ in range(4)]
            packed, reports = {}, []
            with torch.inference_mode():
                result = quantize_blocks([block], "test", rows, (("first",), ("second",)), packed, reports)
                for (actual, _), (hidden, _) in zip(result, rows):
                    torch.testing.assert_close(actual, block(hidden), rtol=0, atol=0)
            self.assertEqual(set(packed), {"test.0.first", "test.0.second"})
            self.assertTrue(all(row["samples"] == 160 and row["fallback"] is None for row in reports))
            for name in ("first", "second"):
                module = block.get_submodule(name)
                torch.testing.assert_close(module.weight, unpack_gptq(packed["test.0." + name]).to(torch.bfloat16),
                                           rtol=0, atol=0)
        finally:
            torch.set_num_threads(prior)


if __name__ == "__main__":
    unittest.main()
