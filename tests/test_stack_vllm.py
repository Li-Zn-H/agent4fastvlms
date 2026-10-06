"""CPU contracts for the optional vLLM Stack adapter."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skills.stack_model import STACK_ARCHITECTURE
from skills.stack_vllm import prepare_vllm_artifact, stack_widths, validate_stack_artifact


class StackVllmContractTests(unittest.TestCase):
    def test_runtime_family_comes_from_artifact_when_descriptor_only_has_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({
                "model_type": "llava_onevision", "architectures": ["LlavaOnevisionForConditionalGeneration"]}))
            with patch("skills.stack_vllm.register_stack_vllm_model", side_effect=AssertionError("no GPU import")):
                result = prepare_vllm_artifact(root)
            self.assertEqual(result["model_type"], "llava_onevision")
            self.assertIn("LlavaOnevisionForConditionalGeneration", result["model_class_overrides"])

    def _artifact(self, folder: str) -> Path:
        root = Path(folder) / "artifact"
        root.mkdir()
        widths = [8960] * 36
        config = {
            "architectures": [STACK_ARCHITECTURE],
            "a4fvlm_stack_widths": widths,
            "a4fvlm_stack_physical_compaction": True,
            "text_config": {
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "intermediate_size": 9728,
            },
            "quantization_config": {"bits": 4, "group_size": 128},
        }
        parent = {
            "schema": "a4fvlm.stack-parent.v1",
            "architecture": STACK_ARCHITECTURE,
            "widths": widths,
            "physical_compaction": True,
        }
        (root / "config.json").write_text(json.dumps(config))
        (root / "stack-parent.json").write_text(json.dumps(parent))
        (root / "manifest.json").write_text(
            json.dumps({"route": "gptqmodel_w4a16"})
        )
        return root

    def test_width_vector_is_strict_and_group_aligned(self):
        config = {
            "architectures": [STACK_ARCHITECTURE],
            "a4fvlm_stack_widths": [8960] * 36,
            "a4fvlm_stack_physical_compaction": True,
            "text_config": {
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "intermediate_size": 9728,
            },
        }
        self.assertEqual(len(stack_widths(config)), 36)
        config["a4fvlm_stack_widths"][0] = 8961
        with self.assertRaises(ValueError):
            stack_widths(config)

    def test_artifact_validation_requires_parent_and_quant_route(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = self._artifact(folder)
            report = validate_stack_artifact(artifact)
            self.assertEqual(report["route"], "gptqmodel_w4a16")
            parent = json.loads((artifact / "stack-parent.json").read_text())
            parent["widths"][0] -= 128
            (artifact / "stack-parent.json").write_text(json.dumps(parent))
            with self.assertRaises(ValueError):
                validate_stack_artifact(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("vllm") is not None,
        "the isolated quantization environment provides the vLLM smoke test",
    )
    def test_registration_api_smoke(self):
        from skills.stack_vllm import register_stack_vllm_model
        from vllm.model_executor.models.registry import ModelRegistry

        result = register_stack_vllm_model()
        self.assertEqual(result["architecture"], STACK_ARCHITECTURE)
        registered = ModelRegistry.models[STACK_ARCHITECTURE]
        self.assertEqual(registered.module_name, "skills.stack_vllm")
        self.assertEqual(registered.class_name, STACK_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
