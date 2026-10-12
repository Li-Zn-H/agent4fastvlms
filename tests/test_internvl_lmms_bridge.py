"""Native LMMS isolation: actual CPU tensor IPC, generation and error cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search.lmms import build_command, cache_environment
from search.internvl_native_worker import NativeBackend

MODEL = Path("models/InternVL3_5-14B")
SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}


class DescriptorTests(unittest.TestCase):
    def test_native_isolation_requires_explicit_backend_and_preserves_descriptor(self):
        config = {"adapter": "internvl35", "path": str(MODEL), "device": "cuda:0", "dtype": "bfloat16",
                  "attn_implementation": "sdpa", "stack_adapter": True, "token_keep_ratio": .5,
                  "token_selection": SELECTION, "native_worker_python": sys.executable}
        self.assertIn("internvl3_5", build_command(config, "mmstar", Path("unused")))
        self.assertEqual(json.loads(cache_environment(config)["A4FVLM_LMMS_INTERNVL_MODEL"]), config)
        for change in ({"native_worker_python": "missing/python"}, {"token_selection": {}},
                       {"token_keep_ratio": 1}, {"execution_backend": "vllm"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_command({**config, **change}, "mmstar", Path("unused"))


@unittest.skipUnless((MODEL / "config.json").exists(), "native InternVL code fixture required")
class GenerationBoundaryTests(unittest.TestCase):
    def test_proxy_preserves_original_native_chat_prompt_and_decoding(self):
        from types import SimpleNamespace
        from adapters.internvl35_dense import load_processor, native_model_class
        from search.internvl_native_worker import create_native_chat_proxy
        import torch

        config = {"adapter": "internvl35", "path": str(MODEL), "device": "cpu", "dtype": "float32",
                  "attn_implementation": "sdpa", "token_keep_ratio": .5,
                  "token_selection": SELECTION, "native_worker_python": sys.executable}
        processor = load_processor(config)
        captured = []
        def generate(**kwargs):
            captured.append(kwargs)
            return torch.tensor([[processor.tokenizer.eos_token_id]])
        reference = SimpleNamespace(template=processor.config.template,
                                    system_message=processor.conversation_factory(processor.config.template).system_message,
                                    num_image_token=256, device=torch.device("cpu"), generate=generate)
        pixels = torch.randn(3, 3, 448, 448)
        with patch("search.internvl_native_worker.NativeBackend", return_value=SimpleNamespace(generate=generate)):
            proxy = create_native_chat_proxy(config)
            from accelerate import Accelerator
            self.assertIs(Accelerator(cpu=True).unwrap_model(proxy), proxy)
            self.assertIsInstance(proxy, torch.nn.Module)
            actual = proxy.chat(processor.tokenizer, pixels, "Describe this image.", {"max_new_tokens": 4},
                                return_history=True)
        expected = native_model_class(MODEL.resolve()).chat(
            reference, processor.tokenizer, pixels, "Describe this image.", {"max_new_tokens": 4},
            return_history=True)
        self.assertEqual(actual, expected)
        self.assertEqual(captured[0].keys(), captured[1].keys())
        for key, value in captured[0].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, captured[1][key], rtol=0, atol=0)
            else:
                self.assertEqual(value, captured[1][key])

    def test_serving_bridge_supports_native_lmms_accelerate_unwrap(self):
        from types import SimpleNamespace
        from accelerate import Accelerator
        from lmms_eval.models.simple import internvl3
        from search.lmms_internvl_vllm_entry import install_bridge
        import torch

        processor = SimpleNamespace(config=SimpleNamespace(template="native"), tokens_per_tile=256,
                                    conversation_factory=lambda _: SimpleNamespace(system_message="system"))
        descriptor = {"path": str(MODEL), "dtype": "bfloat16"}
        backend = SimpleNamespace()
        with patch.object(internvl3, "AutoModel"), patch.object(internvl3, "load_image"), \
             patch("adapters.internvl35_dense.load_processor", return_value=processor), \
             patch("adapters.internvl35_dense.native_model_class", return_value=SimpleNamespace(chat=None)), \
             patch("search.lmms_internvl_vllm_entry.InternVLVllmBackend", return_value=backend), \
             patch.dict(os.environ, {"A4FVLMSP_TOKEN_RECEIPT": "unused"}):
            install_bridge(descriptor)
            proxy = internvl3.AutoModel.from_pretrained(MODEL, device_map="cuda:0").eval()
            self.assertIs(Accelerator(cpu=True).unwrap_model(proxy), proxy)
            self.assertIsInstance(proxy, torch.nn.Module)
            self.assertEqual(list(proxy.parameters()), [])
            self.assertIs(proxy.to("cuda:0").backend, backend)
            with self.assertRaises(ValueError):
                proxy.to("cuda:1")

    def test_real_subprocess_matches_direct_suffix_and_cache_then_fails_closed(self):
        from test_internvl35 import NativeCPUTests
        from adapters.internvl35_dense import InternVLTokenController
        from search.lmms_gate import _mechanism_file
        import torch

        NativeCPUTests.setUpClass()
        self.addCleanup(NativeCPUTests.tearDownClass)
        native = NativeCPUTests.template_model
        config = {"adapter": "internvl35", "path": str(MODEL), "device": "cpu", "dtype": "float32",
                  "token_keep_ratio": .5, "token_selection": SELECTION,
                  "native_worker_python": os.environ.get("INTERNVL_TEST_NATIVE_WORKER", sys.executable),
                  "native_worker_timeout_seconds": 60}
        child_code = """
import sys, json
sys.path.insert(0, 'tests')
from test_internvl35 import NativeCPUTests
from multiprocessing.connection import Connection
from search.internvl_native_worker import serve
NativeCPUTests.setUpClass()
serve(Connection(int(sys.argv[1])), json.loads(sys.argv[2]),
      loader=lambda config: (NativeCPUTests.template_model, None))
"""
        popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "fixture-token-mechanism.jsonl"
            with (root / "child.log").open("w") as log:
                def child(command, **kwargs):
                    return popen(command[:2] + ["-c", child_code] + command[-2:],
                                 **kwargs, stdout=log, stderr=log)
                with patch("search.internvl_native_worker.subprocess.Popen", side_effect=child), \
                     patch.dict(os.environ, {"A4FVLMSP_TOKEN_RECEIPT": str(receipt)}):
                    backend = NativeBackend(config)
                    self.addCleanup(backend.close)
                    process = backend.process
                    ids = torch.tensor([[10] + [98] * (native.num_image_token * 3) + [11, 12]])
                    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                              "pixel_values": torch.randn(3, 3, 112, 112)}
                    options = NativeCPUTests.options
                    controller = InternVLTokenController(native, SELECTION, .5)
                    with torch.inference_mode():
                        expected = controller.generate(dict(inputs), options)
                    result = backend.generate(**inputs, **options)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    self.assertEqual(result.shape[1], 3)
                    row = json.loads(receipt.read_text())
                    for key, value in controller.last_metrics.items():
                        self.assertEqual(row[key], value)
                    self.assertEqual(_mechanism_file(root, "fixture", 1, .5), receipt)
                    with self.assertRaisesRegex(RuntimeError, "greedy single-beam"):
                        backend.generate(**inputs, **{**options, "do_sample": True})
                    self.assertIsNotNone(process.poll())
                    self.assertIsNone(backend.connection)
                    self.assertEqual(len(receipt.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
