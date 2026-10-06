"""Registration is discoverable without activating backends or certifying support."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters import AdapterError, HardwareAdapter, ModelAdapter, hardware, models, select_model
from adapters.cuda import CudaAdapter
from adapters.qwen3vl import Qwen3VLAdapter
from adapters.registry import Registry
from search.registry import catalog


class RegistryTests(unittest.TestCase):
    def checkpoint(self):
        return {"model_type": "qwen3_vl", "text_config": dict(Qwen3VLAdapter.pipeline_dimensions)}

    def test_catalog_is_json_and_has_no_backend_imports_probes_or_writes(self):
        script = """
import json, sys
from search.registry import catalog
value = catalog()
assert value['models'][0]['id'] == 'qwen3_vl'
assert value['hardware'][0]['id'] == 'cuda'
assert not {'torch', 'transformers', 'vllm', 'gptqmodel', 'PIL'} & set(sys.modules)
print(json.dumps(value))
"""
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["hardware"][0]["worker_device"], "cuda:0")
        self.assertIn("unprobed", value["hardware"][0]["capabilities"])

    def test_registry_rejects_duplicates_wrong_types_and_unknown_entries(self):
        registry = Registry(ModelAdapter)
        registry.register(Qwen3VLAdapter())
        with self.assertRaisesRegex(ValueError, "Duplicate adapter"):
            registry.register(Qwen3VLAdapter())
        with self.assertRaises(TypeError):
            registry.register(CudaAdapter())
        with self.assertRaisesRegex(AdapterError, "blocked_preflight.*unknown"):
            registry.get("unimplemented")
        other = Qwen3VLAdapter()
        other.id = "another_qwen"
        with self.assertRaisesRegex(ValueError, "model_type ownership"):
            registry.register(other)
        other.model_types = ("other_model",)
        with self.assertRaisesRegex(ValueError, "pipeline model_id ownership"):
            registry.register(other)

    def test_unknown_models_and_unported_dimensions_stop_before_pipeline_work(self):
        checkpoint = self.checkpoint()
        self.assertIs(select_model(checkpoint, pipeline_id="qwen3vl"), models.get("qwen3_vl"))
        with self.assertRaisesRegex(AdapterError, "model_type"):
            select_model({"model_type": "unregistered"}, pipeline_id="qwen3vl")
        with self.assertRaisesRegex(AdapterError, "not integrated"):
            select_model(checkpoint, pipeline_id="different_job")
        for value in (48, True, "36"):
            changed = copy.deepcopy(checkpoint)
            changed["text_config"]["num_hidden_layers"] = value
            with self.assertRaisesRegex(AdapterError, "topology"):
                select_model(changed, pipeline_id="qwen3vl")

    def test_registration_does_not_implicitly_enable_pipeline(self):
        class FutureAdapter(ModelAdapter):
            id = "future"
            model_types = ("future_type",)

            def describe(self):
                return {"id": self.id}

            def validate_checkpoint(self, checkpoint, *, pipeline=False):
                pass

        registry = Registry(ModelAdapter)
        registry.register(FutureAdapter())
        with patch("adapters.registry.models", registry):
            self.assertEqual(select_model({"model_type": "future_type"}).id, "future")
            with self.assertRaisesRegex(AdapterError, "not integrated"):
                select_model({"model_type": "future_type"}, pipeline_id="future")

    def test_llava_pipeline_scope_rejects_unvalidated_architectures(self):
        checkpoint = json.loads(Path("models/LLaVA-OneVision-7B/config.json").read_text())
        self.assertEqual(select_model(checkpoint, pipeline_id="llava_onevision").id, "llava_onevision")
        for section, field, value in (
            ("text_config", "model_type", "llama"),
            ("text_config", "num_key_value_heads", 8),
            ("vision_config", "patch_size", 16),
            ("vision_config", "num_hidden_layers", 24),
            (None, "vision_aspect_ratio", "square"),
            (None, "vision_feature_layer", True),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(checkpoint)
                (changed[section] if section else changed)[field] = value
                with self.assertRaisesRegex(AdapterError, "topology"):
                    select_model(changed, pipeline_id="llava_onevision")

    def test_llava_pipeline_dense_guard_does_not_reject_padded_stage_artifacts(self):
        checkpoint = json.loads(Path("models/LLaVA-OneVision-7B/config.json").read_text())
        checkpoint["vision_config"]["intermediate_size"] = 4352
        adapter = select_model(checkpoint)
        for stage in ("quantization", "stack", "runtime", "kv_cache", "lmms"):
            adapter.validate_stage(stage, checkpoint)
        with self.assertRaisesRegex(AdapterError, "topology"):
            select_model(checkpoint, pipeline_id="llava_onevision")

    def test_bindings_are_lazy_and_missing_operations_do_not_fall_back(self):
        adapter = models.get("qwen3_vl")
        loader = lambda: None
        with patch("adapters.base.import_module", return_value=SimpleNamespace(load_model=loader)) as load:
            catalog()
            load.assert_not_called()
            self.assertIs(adapter.resolve("dense_load"), loader)
            load.assert_called_once_with("adapters.qwen3vl_dense")
            with self.assertRaisesRegex(AdapterError, "no registered"):
                adapter.resolve("unsupported_skill")
            load.assert_called_once()

    def test_builtin_bindings_point_to_real_top_level_symbols(self):
        for adapter in (*models.entries(), *hardware.entries()):
            for target in adapter.entrypoints.values():
                module, symbol = target.split(":")
                path = Path(*module.split(".")).with_suffix(".py")
                tree = ast.parse(path.read_text())
                names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                self.assertIn(symbol, names, target)

    def test_cuda_registration_preserves_current_selector_contract(self):
        adapter = hardware.get("cuda")
        for value in (0, 1):
            adapter.validate_device(value)
        for value in (-1, 2, True, "cuda:0"):
            with self.assertRaises(AdapterError):
                adapter.validate_device(value)

    def test_pipeline_uses_registration_checks_without_altering_its_snapshot(self):
        from search.pipeline import Pipeline, plan
        adapter = models.get("qwen3_vl")
        with patch.object(adapter, "validate_checkpoint", wraps=adapter.validate_checkpoint) as validate, \
                patch("subprocess.Popen", side_effect=AssertionError("no GPU Worker")), \
                patch("subprocess.check_output", side_effect=AssertionError("no GPU query")):
            pipeline = Pipeline()
            snapshot = pipeline.snapshot()
            saved = (pipeline.root / "config.json").read_bytes()
            self.assertEqual(plan(pipeline)["status"], "completed")
            # Planning/registration must not mutate current inputs or the saved
            # historical job; source config migration can change their hashes.
            self.assertEqual(pipeline.snapshot(), snapshot)
            self.assertEqual((pipeline.root / "config.json").read_bytes(), saved)
        self.assertTrue(any(call.kwargs.get("pipeline") for call in validate.call_args_list))


if __name__ == "__main__":
    unittest.main()
