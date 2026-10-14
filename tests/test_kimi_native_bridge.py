"""Actual isolated native process: Token compaction, physical replay and stops."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
    import transformers
    NATIVE = transformers.__version__ == "4.50.3"
except ImportError:
    NATIVE = False


@unittest.skipUnless(NATIVE and (ROOT / "models/Kimi-VL-A3B-Instruct/config.json").is_file(),
                     "native Kimi environment required")
class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        from test_kimi_artifact import ParentTests
        from skills.kimi_vl_artifact import write_parent
        self.fixture = ParentTests()
        self.fixture.setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_parent(self.fixture.model, self.fixture.processor, self.fixture.recipe, self.root / "parent")
        self.config = {**self.fixture.descriptor, "path": str(self.root / "parent"), "device": "cpu",
                       "native_worker_python": sys.executable, "native_worker_timeout_seconds": 60,
                       "token_keep_ratio": .5, "token_selection": {
                           "score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}}}
        ids = torch.tensor([[1, 100, 100, 100, 100, 25]])
        self.inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                       "pixel_values": torch.randn(16, 3, 14, 14, dtype=torch.bfloat16),
                       "image_grid_hws": torch.tensor([[4, 4]])}

    def test_isolated_process_matches_physical_generation_cache_and_native_stop_strings(self):
        from adapters.kimi_vl_dense import load_model
        from search.kimi_native_worker import NativeBackend
        from skills.kimi_vl_token import KimiTokenController
        from search.lmms_gate import _mechanism_file
        backend = NativeBackend(self.config)
        self.addCleanup(backend.close)
        self.assertEqual(backend.observed_model["physical_widths"], self.fixture.recipe["widths"])
        self.assertEqual(backend.observed_model["routed_experts"], [4])
        reference, processor = load_model(self.config)
        controller = KimiTokenController(reference, self.config["token_selection"], .5)
        options = {**self.fixture.options, "max_new_tokens": 5, "return_dict_in_generate": False}
        receipt = self.root / "fixture-token-mechanism.jsonl"
        with torch.inference_mode(), patch.dict(os.environ, {"A4FVLMSP_TOKEN_RECEIPT": str(receipt)}):
            inputs = dict(self.inputs)
            expected = controller.generate(inputs, options)[:, inputs["input_ids"].shape[1]:]
            actual = backend.generate(**self.inputs, **options)
            self.assertTrue(torch.equal(expected, actual))
            record = json.loads(receipt.read_text().splitlines()[-1])
            self.assertEqual(record["image_tokens_before"], 4)
            self.assertEqual(record["image_tokens_after"], 2)
            self.assertEqual(record["decoder_tokens_after"], 4)
            self.assertEqual(record["kv_tokens_after_last_forward"], 8)
            token = processor.tokenizer.encode("a", add_special_tokens=False)[0]
            self.assertLess(token, 256)
            forced = {**options, "suppress_tokens": [i for i in range(256) if i != token]}
            self.assertEqual(backend.generate(**self.inputs, **forced).shape[1], 5)
            stopped = {**forced, "stop_strings": ["aa"]}
            inputs = dict(self.inputs)
            expected = controller.generate(inputs, {**stopped, "tokenizer": processor.tokenizer})[:, inputs["input_ids"].shape[1]:]
            actual = backend.generate(**self.inputs, **stopped)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(actual.tolist(), [[token, token]])
            record = json.loads(receipt.read_text().splitlines()[-1])
            self.assertEqual(record["output_tokens"], 2)
            self.assertEqual(record["decode_steps"], 1)
            self.assertEqual(_mechanism_file(self.root, "fixture", 3, .5), receipt)
            with self.assertRaisesRegex(RuntimeError, "greedy cached"):
                backend.generate(**self.inputs, **{**options, "do_sample": True})
            self.assertIsNone(backend.connection)
            self.assertEqual(len(receipt.read_text().splitlines()), 3)

    def test_readiness_rejects_changed_expert_widths_routing_and_missing_kernels(self):
        from search.kimi_native_worker import NativeBackend, observed_topology
        ready = observed_topology(self.fixture.model)
        # The fixture's in-memory generation defaults differ from the exported
        # native EOS file, so inspect the actual restored Parent for readiness.
        from adapters.kimi_vl_dense import load_model
        restored, _ = load_model(self.config)
        ready = observed_topology(restored)
        backend = NativeBackend.__new__(NativeBackend)
        backend.config = self.config
        backend.validate_ready(ready)
        for key, value in (("physical_widths", [64, 32, 32]), ("top_k", [1]),
                           ("router_dtypes", {}), ("quantized_modules", ["invented.kernel"])):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                backend.validate_ready({**ready, key: value})
        restored.language_model.model.layers[1].mlp.experts[1].up_proj.out_features += 8
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            observed_topology(restored)
