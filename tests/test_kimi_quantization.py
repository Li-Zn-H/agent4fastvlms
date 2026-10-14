"""Real CPU Hessian export/replay; CUDA Marlin remains a separate GPU gate."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
    import transformers
    AVAILABLE = (transformers.__version__ == "5.14.1"
                 and importlib.util.find_spec("gptqmodel") is not None
                 and (ROOT / "models/Kimi-VL-A3B-Instruct/config.json").is_file())
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "isolated Kimi GPTQ environment and native source required")
class QuantizedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the same native, physically pruned fixture as Parent replay.
        from test_kimi_artifact import ParentTests
        from adapters.kimi_vl_dense import load_processor
        from skills.stack_model import write_pruned_parent
        from skills.kimi_vl_quantization import expected_inventory, export_multimodal_gptq
        cls.fixture = ParentTests()
        cls.fixture.setUp()
        # Use Group-128 and CUDA-Marlin-aligned shapes, so the saved fixture
        # can subsequently undergo a real GPU reload without changing it.
        from adapters.kimi_vl_dense import native_model_class
        from skills.kimi_vl_pruning import dimensions, units, make_recipe, apply_recipe
        config = cls.fixture.model.config
        for key, value in {"hidden_size": 128, "intermediate_size": 512,
                           "moe_intermediate_size": 256, "kv_lora_rank": 128,
                           "qk_rope_head_dim": 64, "qk_nope_head_dim": 32, "v_head_dim": 32}.items():
            setattr(config.text_config, key, value)
        config.vision_config.hidden_size = 128
        config.vision_config.intermediate_size = 160
        torch.manual_seed(19)
        cls.fixture.model = native_model_class(cls.fixture.descriptor)(config).to(torch.bfloat16).eval()
        cls.fixture.model.language_model.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        dims = dimensions(config.text_config)
        domain = units(dims)
        scores = [[[float(i + 1) for i in range(u["original_width"])] for _ in range(u["experts"])] for u in domain]
        cls.fixture.recipe = make_recipe(scores, [u["original_width"] // 2 for u in domain], dims,
                                        {"channel_group_size": 128, "protected_layers": []})
        apply_recipe(cls.fixture.model, cls.fixture.recipe)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.parent = Path(cls.directory.name) / "parent"
        cls.artifact = Path(cls.directory.name) / "artifact"
        write_pruned_parent(cls.fixture.model, cls.fixture.processor, cls.fixture.recipe, cls.parent)
        cls.descriptor = {**cls.fixture.descriptor, "path": str(cls.parent), "device": "cpu"}
        processor = load_processor(cls.descriptor)
        rows = [{"stable_row_id": f"actual-native-input-{i}"} for i in range(8)]
        generator = torch.Generator().manual_seed(783)
        batches = {}
        from transformers import BatchFeature
        for row in rows:
            ids = torch.randint(101, 256, (1, 64), generator=generator)
            ids[0, 0], ids[0, 1] = 1, 100
            batches[row["stable_row_id"]] = BatchFeature({
                "input_ids": ids, "attention_mask": torch.ones_like(ids),
                # Real native preprocessing supplies FP32 pixels. Both direct
                # vision calibration and decoder replay must cast like Dense.
                "pixel_values": torch.randn(4, 3, 14, 14, generator=generator, dtype=torch.float32),
                "image_grid_hws": torch.tensor([[2, 2]])})
        calibration = {"data": "unused-test-input.jsonl", "media_root": "data/evaluation", "limit": len(rows)}
        candidate = json.loads((ROOT / "configs/quantization-kimi-vl.json").read_text())["candidates"][0]
        policy = expected_inventory(json.loads((cls.parent / "config.json").read_text()))
        # Only substitute sample I/O. Every native forward, router decision,
        # Hessian solve, packed export and precision check is real.
        with patch("search.evaluate.load_samples", return_value=rows), patch(
                "adapters.kimi_vl_dense.prepare_inputs", side_effect=lambda _, row, __: batches[row["stable_row_id"]]):
            cls.export = export_multimodal_gptq(
                candidate=candidate, model=cls.descriptor, processor=processor,
                calibration=calibration, output_dir=cls.artifact, module_names=policy,
                loaded_model=cls.fixture.model)
        cls.reload_descriptor = {**cls.descriptor, "path": str(cls.artifact)}
        cls.rows, cls.batches, cls.calibration, cls.candidate = rows, batches, calibration, candidate

    def test_real_hessian_export_and_packed_cpu_reload_preserve_native_outputs(self):
        from gptqmodel.nn_modules.qlinear.torch import TorchLinear
        from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
        from skills.kimi_vl_artifact import (
            _load_packed_model, audit_quantized_artifact, router_fingerprints,
        )
        from skills.kimi_vl_quantization import expected_inventory
        manifest = json.loads((self.artifact / "manifest.json").read_text())
        audit = audit_quantized_artifact(self.artifact, manifest)
        self.assertEqual(audit["status"], "passed", audit)
        self.assertEqual(audit["verified_router_tensors"], 2)
        policy = expected_inventory(self.fixture.model.config.to_dict())
        count = sum(precision == "int4_w4a16" for precision in policy.values())
        self.assertEqual(audit["verified_int4_modules"], count)
        self.assertEqual(audit["verified_packed_geometries"], count)
        def initialize(name, kernel):
            # Torch requires GPTQ-v2 zero codes. Conversion is explicit only
            # in this CPU oracle; production always uses v1 CUDA Marlin.
            convert_gptq_v1_to_v2_format_module(kernel, bits=4, pack_dtype=torch.int32)
            kernel.optimize = lambda *args, **kwargs: None
            kernel.post_init()
        restored = _load_packed_model(self.reload_descriptor, TorchLinear, initialize)
        self.assertTrue(all(b.attn_implementation == "eager" for b in restored.vision_tower.encoder.blocks))
        with torch.inference_mode():
            expected = self.fixture.model(**self.fixture.inputs).logits
            actual = restored(**self.fixture.inputs).logits
            reference_ids = self.fixture.model.generate(**self.fixture.inputs, **self.fixture.options)
            restored_ids = restored.generate(**self.fixture.inputs, **self.fixture.options)
        self.assertTrue(torch.equal(expected, actual), (expected - actual).abs().max().item())
        self.assertTrue(torch.equal(reference_ids, restored_ids))
        self.assertEqual(router_fingerprints(restored), router_fingerprints(self.fixture.model))
        self.assertEqual(restored.generation_config.eos_token_id, [163585])
        self.assertEqual(len(restored.language_model.model.layers[1].mlp.experts), 4)
        receipt = json.loads((self.artifact / "vision-quantization.json").read_text())
        self.assertTrue(all(r["observed_tokens"] > 0 and r["fallback"] is None for r in receipt["modules"]))
        self.assertIn("language_model.model.layers.1.self_attn.kv_b_proj", {r["module"] for r in receipt["modules"]})
        for name in self.fixture.descriptor["native_source_sha256"]:
            self.assertEqual((self.artifact / name).read_bytes(), (self.parent / name).read_bytes())

    def test_router_and_group_index_corruption_fail_actual_tensor_audit(self):
        from safetensors.torch import load_file, save_file
        from skills.kimi_vl_artifact import audit_quantized_artifact
        manifest = json.loads((self.artifact / "manifest.json").read_text())
        shard = next(self.artifact.glob("*.safetensors"))
        original = shard.read_bytes()
        try:
            state = load_file(str(shard))
            key = "language_model.model.layers.1.mlp.gate.e_score_correction_bias"
            state[key] = state[key] + 1
            save_file(state, str(shard))
            audit = audit_quantized_artifact(self.artifact, manifest)
            self.assertEqual(audit["status"], "blocked_preflight")
            self.assertTrue(any(row.get("error") == "changed_native_router" for row in audit["errors"]))
            shard.write_bytes(original)
            state = load_file(str(shard))
            key = "language_model.model.layers.1.mlp.experts.0.down_proj.g_idx"
            state[key][0] = 1
            save_file(state, str(shard))
            audit = audit_quantized_artifact(self.artifact, manifest)
            self.assertTrue(any(row.get("error") == "invalid_symmetric_group128_packing" for row in audit["errors"]))
        finally:
            shard.write_bytes(original)

    def test_changed_source_parent_or_missing_expert_calibration_are_rejected(self):
        from skills.kimi_vl_artifact import read_parent, validate_quantized_bindings
        from skills.quantization import digest
        source_path = self.artifact / "source-config.json"
        original = source_path.read_bytes()
        header = json.loads((self.artifact / "config.json").read_text())
        manifest = json.loads((self.artifact / "manifest.json").read_text())
        try:
            changed = json.loads(original)
            changed["text_config"]["num_experts_per_tok"] = 1
            source_path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                read_parent(self.artifact)
        finally:
            source_path.write_bytes(original)
        receipt_path = self.artifact / "vision-quantization.json"
        original = receipt_path.read_bytes()
        try:
            receipt = json.loads(original)
            receipt["modules"][0]["observed_tokens"] = 0
            receipt_path.write_text(json.dumps(receipt))
            manifest["visual_quantization"]["calibration_receipt_digest"] = digest(receipt)
            manifest["artifact_digest"] = digest({k: v for k, v in manifest.items() if k != "artifact_digest"})
            with self.assertRaisesRegex(ValueError, "complete observed native"):
                validate_quantized_bindings(self.artifact, header, manifest)
        finally:
            receipt_path.write_bytes(original)

    def test_public_loader_rejects_cpu_instead_of_falling_back(self):
        from skills.kimi_vl_artifact import load_quantized_model
        with self.assertRaisesRegex(ValueError, "CUDA Marlin"):
            load_quantized_model(self.reload_descriptor)

    def test_independent_dense_export_has_no_sp_ancestry(self):
        import shutil
        from adapters.kimi_vl_dense import load_processor, native_model_class
        from skills.internvl35_artifact import save_tensors
        from skills.kimi_vl_artifact import validate_quantized_bindings, audit_quantized_artifact
        from skills.kimi_vl_quantization import expected_inventory
        from skills.quantization_adapters import export_candidate
        source = Path(self.directory.name) / "dense"
        source.mkdir()
        header = json.loads((self.parent / "config.json").read_text())
        header = {key: value for key, value in header.items() if not key.startswith("a4fvlm_stack_")}
        header["architectures"] = ["KimiVLForConditionalGeneration"]
        (source / "config.json").write_text(json.dumps(header))
        for name in (*self.descriptor["native_source_sha256"], "tiktoken.model"):
            shutil.copyfile(self.parent / name, source / name)
        descriptor = {**self.descriptor, "path": str(source)}
        cls = native_model_class(descriptor)
        torch.manual_seed(19)
        model = cls(cls.config_class(**header)).to(torch.bfloat16).eval()
        model.language_model.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        save_tensors(model.state_dict(), source)
        original = {f.name: f.read_bytes() for f in source.glob("*.safetensors")}
        output = Path(self.directory.name) / "independent"
        processor = load_processor(descriptor)
        with patch("search.evaluate.load_samples", return_value=self.rows), patch(
                "adapters.kimi_vl_dense.prepare_inputs", side_effect=lambda _, row, __: self.batches[row["stable_row_id"]]), patch(
                "adapters.kimi_vl_dense.load_model", return_value=(model, processor)) as load:
            export_candidate(candidate=self.candidate, model=descriptor, processor=processor,
                             calibration=self.calibration, output_dir=output,
                             module_names=expected_inventory(header))
            load.assert_called_once_with(descriptor)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertNotIn("stack", manifest)
        self.assertFalse((output / "sp-recipe.json").exists())
        self.assertFalse((output / "stack-parent.json").exists())
        validate_quantized_bindings(output, json.loads((output / "config.json").read_text()), manifest)
        self.assertEqual(audit_quantized_artifact(output, manifest)["status"], "passed")
        self.assertEqual(original, {f.name: f.read_bytes() for f in source.glob("*.safetensors")})

    def test_stack_rejects_changed_recipe_before_starting_quantization(self):
        from skills.stack_adapters import StackAdapterError, export_pruned_gptqmodel
        from skills.stack_model import stack_architecture_for_config, load_stack_model
        from skills.kimi_vl_artifact import STACK_ARCHITECTURE, read_parent
        from skills.kimi_vl_pruning import audit_recipe
        header, recipe, parent = read_parent(self.parent)
        self.assertEqual(stack_architecture_for_config(header), STACK_ARCHITECTURE)
        restored, _ = load_stack_model(self.descriptor)
        self.assertEqual(audit_recipe(restored, recipe), parent["mechanism"])
        with self.assertRaisesRegex(StackAdapterError, "differs from selected recipe"):
            export_pruned_gptqmodel(candidate=self.candidate, parent_dir=self.parent,
                                    processor=self.fixture.processor, calibration=self.calibration,
                                    output_dir=Path(self.directory.name) / "stale-parent-attempt",
                                    model=self.descriptor, module_names=[], recipe_digest="stale-recipe")

    def test_vllm_preparation_authenticates_artifact_without_model_or_gpu_construction(self):
        from skills.stack_vllm import prepare_vllm_artifact, validate_stack_artifact
        from skills.vllm_token import configure_runtime
        from search.lmms_kimi_entry import validate_native_config
        import sys
        self.addCleanup(configure_runtime, None)
        runtime = {"keep_ratio": .5, "selection": {"score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}}}
        result = prepare_vllm_artifact(self.artifact, token_runtime=runtime)
        self.assertTrue(result["custom_stack"])
        self.assertEqual(result["model_type"], "kimi_vl")
        self.assertEqual(list(result["model_class_overrides"].values()), ["skills.kimi_vl_vllm:A4FVLMKimiVLForConditionalGeneration"])
        self.assertEqual(result["vision_token"]["runtime"], runtime)
        self.assertEqual(validate_stack_artifact(self.artifact)["widths"], self.fixture.recipe["widths"])
        validate_native_config({**self.reload_descriptor, "stack_adapter": True,
                                "native_worker_python": sys.executable,
                                "token_keep_ratio": .5, "token_selection": runtime["selection"]})
        validate_native_config({**self.reload_descriptor, "stack_adapter": True,
                                "execution_backend": "vllm", "runtime": {"vllm_max_model_len": 131072},
                                "token_keep_ratio": .5, "token_selection": runtime["selection"]})
        self.assertFalse(torch.cuda.is_initialized())
