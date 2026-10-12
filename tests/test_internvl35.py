"""Native CPU contract checks. No checkpoint weights or GPU are loaded."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters import AdapterError, models, select_model
from adapters.capabilities import check_capabilities
from adapters.cuda import CudaAdapter
from skills.vision_token import geometry_keep_count


MODEL = Path(os.environ.get("INTERNVL_TEST_MODEL", "models/InternVL3_5-14B"))
CONFIG = {"adapter": "internvl35", "hardware": "cuda", "path": str(MODEL.resolve()),
          "device": "cpu", "dtype": "float32", "attn_implementation": "sdpa"}
SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}


class PureTests(unittest.TestCase):
    def test_repeated_geometry_counts_each_tile_coverage_floor(self):
        grid = {"rows": 6, "columns": 6}
        self.assertEqual(geometry_keep_count({"rows": 16, "columns": 16, "repeats": 13}, .5, grid), 1664)
        self.assertEqual(geometry_keep_count({"rows": 16, "columns": 16, "repeats": 13}, .1, grid), 468)
        self.assertEqual(geometry_keep_count({"rows": 24, "columns": 24}, .5, grid), 288)
        for repeats in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                geometry_keep_count({"rows": 16, "columns": 16, "repeats": repeats}, .5, grid)

    def test_dense_lmms_has_explicit_route_and_rejects_unported_serving(self):
        from search.lmms import build_command, cache_environment
        config = {**CONFIG, "device": "cuda:0", "dtype": "bfloat16"}
        command = build_command(config, "mmstar", Path("unused"))
        self.assertIn("internvl3_5", command)
        self.assertTrue(command[1].endswith("lmms_internvl_entry.py"))
        self.assertEqual(json.loads(cache_environment(config)["A4FVLM_LMMS_INTERNVL_MODEL"]), config)
        for changes in ({"token_keep_ratio": .5}, {"stack_adapter": True}, {"execution_backend": "vllm"},
                        {"dtype": "float16"}, {"adapter": "unknown"}):
            with self.assertRaises(ValueError):
                build_command({**config, **changes}, "mmstar", Path("unused"))


@unittest.skipUnless((MODEL / "config.json").is_file(), "local native InternVL code/config fixture required")
class NativeCPUTests(unittest.TestCase):
    model_path = MODEL
    model_config = CONFIG
    adapter_id = "internvl35"
    ready_stages = {"dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms"}
    pipeline_id = "internvl35"

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, AutoModel
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(123)
        c = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True, local_files_only=True)
        c.llm_config.hidden_size = 64
        c.llm_config.intermediate_size = 128
        c.llm_config.num_hidden_layers = 2
        c.llm_config.layer_types = ["full_attention"] * 2
        c.llm_config.num_attention_heads = 4
        c.llm_config.num_key_value_heads = 2
        c.llm_config.head_dim = 16
        c.llm_config.vocab_size = 128
        c.llm_config.eos_token_id = 2
        c.llm_config.bos_token_id = 1
        c.llm_config.pad_token_id = 0
        c.vision_config.hidden_size = 32
        c.vision_config.intermediate_size = 64
        c.vision_config.num_hidden_layers = 2
        c.vision_config.num_attention_heads = 4
        c.vision_config.image_size = 112
        c.force_image_size = 112
        cls.template_model = AutoModel.from_config(c, trust_remote_code=True, use_flash_attn=False).float().eval()
        cls.template_model.language_model.set_attn_implementation("sdpa")
        cls.template_model.img_context_token_id = 98
        cls.options = {"max_new_tokens": 3, "do_sample": False, "eos_token_id": None,
                       "pad_token_id": 0, "return_dict_in_generate": False}

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def setUp(self):
        self.model = copy.deepcopy(self.template_model)

    def inputs(self, tiles=3):
        from transformers import BatchFeature
        torch = self.torch
        count = self.model.num_image_token * tiles
        ids = torch.tensor([[10] + [98] * count + [11, 12]])
        return BatchFeature({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                             "pixel_values": torch.randn(tiles, 3, 112, 112)})

    def test_registration_and_pipeline_require_native_topology(self):
        checkpoint = json.loads((self.model_path / "config.json").read_text())
        adapter = select_model(checkpoint)
        self.assertEqual(adapter.id, self.adapter_id)
        report = check_capabilities(adapter, checkpoint, CudaAdapter(), device=0)
        self.assertEqual({name for name, row in report["stages"].items() if row["status"] == "ready_for_probe"},
                         self.ready_stages)
        self.assertTrue(all(row["quality"] == "unverified" for row in report["stages"].values()
                            if row["status"] == "ready_for_probe"))
        if self.pipeline_id:
            self.assertEqual(select_model(checkpoint, pipeline_id=self.pipeline_id).id, self.adapter_id)
        else:
            with self.assertRaisesRegex(AdapterError, "not integrated"):
                select_model(checkpoint, pipeline_id=self.adapter_id)
        other_type = "qwen2" if checkpoint["llm_config"]["model_type"] == "qwen3" else "qwen3"
        for section, key, value in (("llm_config", "model_type", other_type),
                                    (None, "max_dynamic_patch", 6), (None, "downsample_ratio", .25)):
            changed = copy.deepcopy(checkpoint)
            (changed[section] if section else changed)[key] = value
            with self.assertRaises(AdapterError):
                select_model(changed)
            with self.assertRaises(AdapterError):
                select_model(changed, pipeline_id=self.adapter_id)

    def test_native_checkpoint_reload_uses_exact_weights_and_requested_attention(self):
        from adapters.internvl35_dense import load_model
        processor = SimpleNamespace(image_token_id=98, config=self.model.config,
                                    conversation_factory=lambda template: SimpleNamespace(sep="eos"),
                                    tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 2))
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            with patch("adapters.internvl35_dense.load_processor", return_value=processor):
                reloaded, _ = load_model({**self.model_config, "path": directory})
        for key, value in self.model.state_dict().items():
            self.torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(reloaded.language_model.config._attn_implementation, "sdpa")

    def test_native_suffix_and_keep_all_equivalence(self):
        from adapters.internvl35_dense import InternVLTokenController
        inputs = self.inputs()
        with self.torch.inference_mode():
            native = self.model.generate(**inputs, **self.options)
            kept = InternVLTokenController(self.model, SELECTION, 1).generate(inputs, self.options)
        self.assertEqual(tuple(native.shape), (1, 3))
        self.torch.testing.assert_close(native, kept, rtol=0, atol=0)
        self.assertEqual(self.model.language_model.config._attn_implementation, "sdpa")

    def test_framework_does_not_strip_native_suffix(self):
        from search.model import generate_outputs
        from transformers import BatchFeature
        inputs = self.inputs(1)
        decoded = []
        def decode(ids, **kwargs):
            decoded.append(ids.clone())
            return ["native answer"]
        processor = SimpleNamespace(_agent4vlm_adapter_id=self.adapter_id, batch_decode=decode)
        with patch("adapters.internvl35_dense.prepare_inputs", return_value=BatchFeature(dict(inputs))):
            outputs = generate_outputs(self.model, processor, [{"stable_row_id": "one"}], media_root=".",
                                       generation={"max_new_tokens": 3, "do_sample": False, "ignore_eos": True})
        self.assertEqual(decoded[0].shape[1], 3)
        self.assertEqual(outputs[0]["output_text"], "native answer")

    def test_all_tiles_compact_text_and_observed_cache(self):
        from adapters.internvl35_dense import InternVLTokenController
        controller = InternVLTokenController(self.model, SELECTION, .5)
        inputs = self.inputs(3)
        original = inputs["input_ids"].clone()
        with self.torch.inference_mode():
            result = controller.generate(inputs, self.options)
        metrics = controller.last_metrics
        self.assertEqual((metrics["image_tokens_before"], metrics["image_tokens_after"]), (48, 24))
        self.assertEqual(metrics["tiles_selected"], 3)
        self.assertEqual(metrics["decoder_tokens_after"], 27)
        self.assertEqual(metrics["kv_tokens_after_last_forward"], 29)
        self.assertEqual(metrics["output_tokens"], result.shape[1])
        self.torch.testing.assert_close(inputs["input_ids"][inputs["input_ids"] != 98], original[original != 98])
        self.assertNotIn("extract_feature", self.model.__dict__)
        self.assertFalse(self.model.language_model.model._forward_hooks)

    def test_compacted_generation_matches_manual_native_feature_selection(self):
        from adapters.internvl35_dense import InternVLTokenController
        from skills.vision_token import select_indices
        inputs = self.inputs(3)
        with self.torch.inference_mode():
            features = self.model.extract_feature(inputs["pixel_values"])
            selected = self.torch.stack([tile.index_select(0, select_indices(tile, 4, 4, .5, SELECTION))
                                         for tile in features])
            ids = self.torch.tensor([[10] + [98] * 24 + [11, 12]])
            expected = self.model.generate(pixel_values=inputs["pixel_values"], visual_features=selected,
                                            input_ids=ids, attention_mask=self.torch.ones_like(ids), **self.options)
            actual = InternVLTokenController(self.model, SELECTION, .5).generate(inputs, self.options)
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_generation_failure_cleans_hooks_without_mutating_inputs(self):
        from adapters.internvl35_dense import InternVLTokenController
        controller = InternVLTokenController(self.model, SELECTION, .5)
        inputs = self.inputs()
        original = inputs["input_ids"].clone()
        with patch.object(self.model.language_model, "generate", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                with self.torch.inference_mode():
                    controller.generate(inputs, self.options)
        self.assertIsNone(controller.last_metrics)
        self.assertNotIn("extract_feature", self.model.__dict__)
        self.assertFalse(self.model.language_model.model._forward_hooks)
        self.torch.testing.assert_close(inputs["input_ids"], original)

    def test_sp_slice_matches_zero_mask_reference(self):
        from skills.structured_pruning import apply_recipe, audit_recipe, make_recipe
        torch = self.torch
        dimensions = {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128}
        recipe = make_recipe([list(range(128))] * 2, [64, 96], dimensions,
                             {"channel_group_size": 32, "protected_layers": []})
        reference = copy.deepcopy(self.model)
        with torch.no_grad():
            for layer, channels in zip(reference.language_model.model.layers, recipe["retained_channels"]):
                removed = sorted(set(range(128)) - set(channels))
                layer.mlp.down_proj.weight[:, removed] = 0
        apply_recipe(self.model, recipe)
        audit = audit_recipe(self.model, recipe)
        self.assertEqual(audit["remaining_ffn_parameters"], (64 + 96) * 3 * 64)
        value = torch.randn(1, 9, 64)
        with torch.inference_mode():
            expected = reference.language_model(inputs_embeds=value).logits
            actual = self.model.language_model(inputs_embeds=value).logits
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "fresh Dense"):
            apply_recipe(self.model, recipe)

    def test_native_image_processing_and_chat_prompt_match(self):
        from adapters.internvl35_dense import load_processor, prepare_inputs, token_geometry
        from lmms_eval.models.simple.internvl3 import load_image
        from PIL import Image
        processor = load_processor(self.model_config)
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            return self.torch.tensor([[processor.tokenizer.eos_token_id]])
        native = SimpleNamespace(template=processor.config.template,
                                 system_message=processor.conversation_factory(processor.config.template).system_message,
                                 num_image_token=256, device=self.torch.device("cpu"), generate=generate)
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (896, 448), (30, 120, 210))
            image.save(Path(directory) / "test.png")
            inputs = prepare_inputs(processor, {"media": {"logical_path": "test.png"}, "logical_question": "Describe."}, Path(directory))
            self.torch.testing.assert_close(inputs["pixel_values"], load_image(image), rtol=0, atol=0)
            type(self.model).chat(native, processor.tokenizer, inputs["pixel_values"], "Describe.", {})
        self.torch.testing.assert_close(inputs["input_ids"], captured["input_ids"], rtol=0, atol=0)
        geometry = token_geometry(processor, inputs)
        self.assertGreater(geometry["repeats"], 1)
        self.assertEqual(int((inputs["input_ids"] == processor.image_token_id).sum()), geometry["repeats"] * 256)


if __name__ == "__main__":
    unittest.main()
