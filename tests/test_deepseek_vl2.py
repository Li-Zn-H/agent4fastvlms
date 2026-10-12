"""DeepSeek-VL2 static registration and native CPU input contracts."""

import copy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from adapters import AdapterError, models, select_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/model-deepseek-vl2.json"
CHECKPOINT = ROOT / "models/DeepSeek-VL2"
RESULT_ROOT = ROOT / "results/model/deepseek_vl2/rtx_pro6000_gpu072a4781"


class RegistrationTests(unittest.TestCase):
    def test_native_lmms_dispatch_and_unsupported_artifacts(self):
        from search.lmms import build_command, cache_environment
        config = json.loads(CONFIG.read_text())
        command = build_command(config, "chartqa", Path("unused"), limit=2)
        self.assertTrue(command[1].endswith("lmms_deepseek_entry.py"))
        self.assertEqual(command[command.index("--model") + 1], "deepseek_vl2")
        self.assertEqual(json.loads(cache_environment(config)["A4FVLM_LMMS_DEEPSEEK_MODEL"]), config)
        for change in ({"stack_adapter": True}, {"execution_backend": "vllm"},
                       {"token_keep_ratio": .5}, {"dtype": "float16"}):
            with self.assertRaises(ValueError):
                build_command({**config, **change}, "chartqa", Path("unused"))

    def test_layout_partitions_every_feature_and_separator(self):
        from skills.deepseek_vl2_token import image_layout
        from skills.vision_token import geometry_keep_count
        for nx, ny in ((1, 1), (2, 3), (9, 1), (1, 9)):
            tiles, protected, count = image_layout(14, nx, ny)
            flattened = [x for tile in tiles for x in tile] + protected
            self.assertEqual(sorted(flattened), list(range(count)))
            geometry = {"rows": 14, "columns": 14, "repeats": 1 + nx * ny,
                        "protected_tokens": len(protected)}
            self.assertEqual(geometry_keep_count(geometry, 1, {"rows": 6, "columns": 6}), count)
            self.assertEqual(geometry_keep_count(geometry, .5, {"rows": 6, "columns": 6}),
                             (1 + nx * ny) * 98 + len(protected))
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                geometry_keep_count({"rows": 14, "columns": 14, "protected_tokens": value},
                                    .5, {"rows": 6, "columns": 6})

    def test_catalog_does_not_import_backends(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import adapters, sys; assert not any(x in sys.modules for x in "
            "('torch', 'transformers', 'deepseek_vl2', 'vllm')); "
            "assert adapters.models.get('deepseek_vl2').pipeline_ids == ('deepseek_vl2',)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless((CHECKPOINT / "config.json").is_file(), "local checkpoint metadata required")
    def test_variant_and_stage_scope(self):
        config = json.loads((CHECKPOINT / "config.json").read_text())
        adapter = select_model(config)
        self.assertEqual(adapter.id, "deepseek_vl2")
        adapter.validate_stage("dense", config)
        adapter.validate_stage("sp", config)
        adapter.validate_stage("quantization", config)
        adapter.validate_stage("stack", config)
        from adapters.capabilities import check_capabilities
        from adapters import hardware
        report = check_capabilities(adapter, config, hardware.get("cuda"),
                                    stages=("quantization", "stack", "runtime", "kv_cache", "lmms"))
        self.assertEqual(report["status"], "ready_for_probe")
        for stage in report["stages"]:
            self.assertEqual(report["stages"][stage]["quality"], "unverified")
            self.assertEqual(report["stages"][stage]["environment"], "unverified")
        small = copy.deepcopy(config)
        small["language_config"]["n_routed_experts"] = 64
        with self.assertRaises(AdapterError):
            adapter.validate_checkpoint(small)
        adapter.validate_checkpoint(config, pipeline=True)
        self.assertIs(select_model(config, pipeline_id="deepseek_vl2"), adapter)
        with self.assertRaises(AdapterError):
            select_model(small, pipeline_id="deepseek_vl2")

    @unittest.skipUnless((RESULT_ROOT / "stack-serving-lmms/final/serving-freeze.json").is_file()
                         and importlib.util.find_spec("safetensors") is not None,
                         "complete local DeepSeek serving evidence and safetensors required")
    def test_completed_pipeline_reuses_evidence_without_gpu_work(self):
        from search.pipeline import Pipeline, execute, plan

        pipeline = Pipeline("configs/pipeline-deepseek-vl2.json")
        with tempfile.TemporaryDirectory() as folder:
            pipeline.root = Path(folder) / "handoff"
            with patch("subprocess.Popen", side_effect=AssertionError("no child process")), \
                    patch("search.optimize.hardware", side_effect=AssertionError("no GPU query")), \
                    patch("search.optimize.launch", side_effect=AssertionError("no Worker")):
                planned = plan(pipeline)
                self.assertEqual(planned["status"], "completed")
                self.assertFalse(pipeline.root.exists())
                state = execute(pipeline)
            self.assertEqual(state["stage_invocations"], 0)
            self.assertTrue(all(row["action"] == "reused" for row in state["stages"].values()))
            self.assertGreaterEqual(state["stages"]["serving"]["retention"], .95)


def native_environment_available():
    try:
        return (importlib.metadata.version("transformers") == "4.46.3"
                and (CHECKPOINT / "processor_config.json").is_file()
                and (ROOT / ".cache/deepseek-vl2/upstream/deepseek_vl2").is_dir())
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(native_environment_available(), "isolated DeepSeek native environment required")
class NativeInputTests(unittest.TestCase):
    def test_processor_load_does_not_import_native_model_dependencies(self):
        script = (
            "import json, sys; from pathlib import Path; "
            "from adapters.deepseek_vl2_dense import load_processor, prepare_image; "
            "from PIL import Image; "
            "p = load_processor(json.loads(Path('configs/model-deepseek-vl2.json').read_text())); "
            "assert not any(name in sys.modules for name in ('deepseek_vl2', 'xformers', 'attrdict')); "
            "batch = prepare_image(p, Image.new('RGB', (384, 384)), 'Describe.'); "
            "assert p.prompt_token_ids('Describe.', (384, 384)) == batch['input_ids'][0].tolist()"
        )
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_generation_preserves_task_budget_and_stops(self):
        from search.deepseek_lmms import generation_options
        original = {"max_new_tokens": 32, "temperature": 0, "do_sample": False, "until": ["ASSISTANT:"]}
        before = copy.deepcopy(original)
        options, stops = generation_options(original)
        self.assertEqual(original, before)
        self.assertEqual(options["max_new_tokens"], 32)
        self.assertFalse(options["do_sample"])
        self.assertEqual(stops, ["ASSISTANT:"])
        for invalid in ({"use_cache": False}, {"num_beams": 2}, {"max_new_tokens": 0}):
            with self.assertRaises(ValueError):
                generation_options(invalid)

    @classmethod
    def setUpClass(cls):
        import torch
        from adapters.deepseek_vl2_dense import load_processor
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.processor = load_processor(json.loads(CONFIG.read_text()))

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_native_global_local_layout(self):
        from PIL import Image
        from adapters.deepseek_vl2_dense import prepare_image
        for size, grid in (((384, 384), (1, 1)), ((768, 384), (2, 1)), ((384, 768), (1, 2))):
            with self.subTest(size=size):
                inputs = prepare_image(self.processor, Image.new("RGB", size), "Describe this image.")
                self.assertEqual(inputs["images_spatial_crop"].tolist(), [[list(grid)]])
                self.assertEqual(inputs["images"].shape[1], 1 + grid[0] * grid[1])
                count = 14 * 15 + 1 + grid[1] * 14 * (grid[0] * 14 + 1)
                self.assertEqual(int(inputs["images_seq_mask"].sum()), count)
                self.assertEqual(int((inputs["input_ids"] == self.processor.native.image_token_id).sum()), count)
                self.assertEqual(str(inputs["images"].dtype), "torch.bfloat16")

    def test_rejects_multiple_placeholders(self):
        from PIL import Image
        from adapters.deepseek_vl2_dense import prepare_image
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_image(self.processor, Image.new("RGB", (384, 384)), "<image> and <image>")

    def test_selected_indices_preserve_layout_and_each_tile(self):
        import torch
        from skills.deepseek_vl2_token import image_layout, select_image_indices
        selection = {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}
        tiles, protected, count = image_layout(14, 3, 2)
        features = torch.arange(count, dtype=torch.float32).unsqueeze(1)
        selected = select_image_indices(features, 14, 3, 2, .5, selection).tolist()
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(set(protected).issubset(selected))
        for tile in tiles:
            kept = set(tile).intersection(selected)
            self.assertEqual(len(kept), 98)
            cells = {(tile.index(x) // 14 * 2 // 14, tile.index(x) % 14 * 2 // 14) for x in kept}
            self.assertEqual(len(cells), 4)
        self.assertEqual(select_image_indices(features, 14, 3, 2, 1., selection).tolist(), list(range(count)))

    def test_native_decoder_keep_all_and_physical_cache_compaction(self):
        import torch
        from adapters.deepseek_vl2_dense import native_model_class
        model_class = native_model_class(json.loads(CONFIG.read_text()))
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        from skills.deepseek_vl2_token import DeepSeekTokenController, image_layout
        from transformers import BatchFeature

        config = DeepseekVLV2Config(tile_tag="2D", global_view_pos="head",
            vision_config={"image_size": 56, "patch_size": 14, "width": 32, "layers": 1,
                           "heads": 4, "mlp_ratio": 2},
            projector_config={"input_dim": 32, "n_embed": 32},
            language_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                "moe_intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 2, "kv_lora_rank": 16, "q_lora_rank": None,
                "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 8,
                "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "first_k_dense_replace": 1, "n_group": 1, "topk_group": 1,
                "topk_method": "noaux_tc", "scoring_func": "sigmoid",
                "max_position_embeddings": 128, "bos_token_id": 0, "eos_token_id": 1})
        torch.manual_seed(7)
        model = model_class(config).float().eval()

        class TinyVision(torch.nn.Module):
            def forward(self, pixels):
                # Deterministic synthetic vision input; real native projector,
                # assembly, MoE/MLA decoder and generation/cache remain active.
                return torch.arange(16 * 32, dtype=pixels.dtype).reshape(1, 16, 32).expand(len(pixels), -1, -1) / 512

        model.vision = TinyVision()
        model.language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        count = image_layout(2, 2, 1)[2]
        ids = torch.tensor([[3] + [98] * count + [4, 5]])
        original = BatchFeature({"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "images": torch.zeros(1, 3, 3, 56, 56), "images_seq_mask": ids == 98,
            "images_spatial_crop": torch.tensor([[[2, 1]]])})
        options = {"max_new_tokens": 3, "do_sample": False, "eos_token_id": None,
                   "pad_token_id": 2, "return_dict_in_generate": False}
        selection = {"score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}}
        with torch.inference_mode():
            dense = model.generate(**original, **options)
            keep_all = DeepSeekTokenController(model, selection, 1.)
            self.assertTrue(torch.equal(dense, keep_all.generate(copy.deepcopy(original), options)))
            controller = DeepSeekTokenController(model, selection, .5)
            compacted = copy.deepcopy(original)
            result = controller.generate(compacted, options)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(controller.last_metrics["image_tokens_after"], 11)
        self.assertEqual(controller.last_metrics["decoder_tokens_after"], 14)
        self.assertEqual(controller.last_metrics["kv_tokens_after_last_forward"], 16)
        self.assertTrue(torch.equal(original["input_ids"][~original["images_seq_mask"]],
                                    compacted["input_ids"][~compacted["images_seq_mask"]]))


if __name__ == "__main__":
    unittest.main()
