"""Exported storage, rather than assignment intent, certifies quantization."""

from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

from skills.quantization_evidence import audit_gptq_export, llava_gptq_preflight


class QuantizationEvidenceTests(unittest.TestCase):
    def test_direct_export_blocks_before_backend_import(self):
        from skills.quantization_adapters import export_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"vision_config": {
                "hidden_size": 1152, "intermediate_size": 4304}}))
            with patch("skills.quantization_adapters.importlib.import_module",
                       side_effect=AssertionError("backend must stay lazy")):
                with self.assertRaisesRegex(ValueError, "blocked_preflight"):
                    export_candidate(candidate={"route": "gptqmodel_w4a16", "group_size": 128},
                                     model={"adapter": "llava_onevision", "path": str(root)},
                                     processor=None, calibration={}, output_dir=root / "out")

    def test_siglip_4304_is_a_kernel_block_not_quality_failure(self):
        result = llava_gptq_preflight({"vision_config": {
            "hidden_size": 1152, "intermediate_size": 4304}}, 128)
        self.assertEqual(result["status"], "blocked_preflight")
        self.assertEqual(len(result["blockers"]), 2)
        aligned = llava_gptq_preflight({"vision_config": {
            "hidden_size": 1152, "intermediate_size": 4352}}, 128)
        self.assertEqual(aligned["status"], "ready_for_probe")

    def test_dense_visual_tensor_cannot_certify_int4_and_fp16_is_not_bf16(self):
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"module_assignment": {"language": "int4_w4a16",
                         "vision": "int4_w4a16", "mapper": "bf16"}}
            tensors = {"language." + name: np.zeros((2,), dtype=dtype)
                       for name, dtype in (("qweight", np.int32), ("qzeros", np.int32),
                                           ("scales", np.float16), ("g_idx", np.int32))}
            tensors.update({"vision.weight": np.zeros((2, 2), dtype=np.float16),
                            "mapper.weight": np.zeros((2, 2), dtype=np.float16)})
            save_file(tensors, str(root / "model.safetensors"))
            result = audit_gptq_export(root, manifest)
            self.assertEqual(result["status"], "blocked_preflight")
            self.assertEqual(result["declared_int4_modules"], 2)
            self.assertEqual(result["verified_int4_modules"], 1)
            self.assertEqual({row["module"] for row in result["errors"]}, {"vision", "mapper"})
            result = audit_gptq_export(root, {"module_assignment": {"language": "int4_w4a16"}})
            self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
