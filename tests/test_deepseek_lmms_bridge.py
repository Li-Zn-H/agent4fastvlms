"""Native optimized LMMS isolation, physical topology and real stop criteria."""

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search.deepseek_native_worker import NativeBackend
from search.lmms import build_command

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}}


class DescriptorTests(unittest.TestCase):
    def test_optimized_descriptor_requires_explicit_isolation_and_physical_quantization(self):
        config = json.loads((ROOT / "configs/model-deepseek-vl2.json").read_text())
        config.update(stack_adapter=True, token_keep_ratio=.5, token_selection=SELECTION,
                      native_worker_python=sys.executable)
        for change, message in (({}, "freshly quantized physical Parent"),
                                ({"native_worker_python": "missing/python"}, "native_worker_python"),
                                ({"token_selection": {}}, "selection policy"),
                                ({"token_keep_ratio": 1}, "keep ratio"),
                                ({"execution_backend": "vllm"}, "Runtime")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                build_command({**config, **change}, "mmstar", Path("unused"))

    def test_import_does_not_load_model_or_quantization_dependencies(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import search.deepseek_native_worker, search.lmms_deepseek_entry, sys; "
            "assert not any(x in sys.modules for x in ('torch','transformers','gptqmodel','lmms_eval','vllm'))"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


def native_environment():
    try:
        return (importlib.metadata.version("transformers") == "4.46.3" and
                (ROOT / ".cache/deepseek-vl2/upstream/deepseek_vl2").is_dir())
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(native_environment(), "original native DeepSeek environment required")
class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        import torch
        from deepseek_lmms_fixture import cpu_attention, tokenizer, write_parent
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = write_parent(self.root / "parent")
        self.tokenizer = tokenizer()
        self.config = json.loads((ROOT / "configs/model-deepseek-vl2.json").read_text())
        self.config.update(path=str(self.root / "parent"), device="cpu", dtype="float32",
                           token_keep_ratio=.5, token_selection=SELECTION,
                           native_worker_python=os.environ.get("DEEPSEEK_TEST_NATIVE_WORKER", sys.executable),
                           native_worker_timeout_seconds=60)
        attention = patch("xformers.ops.memory_efficient_attention", cpu_attention)
        attention.start()
        self.addCleanup(attention.stop)
        self.options = dict(max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=2,
                            eos_token_id=None, return_dict_in_generate=False)

    def backend(self):
        child_code = "import sys; sys.path.insert(0, 'tests'); from deepseek_lmms_fixture import serve_fixture; serve_fixture()"
        popen = subprocess.Popen
        log = (self.root / "child.log").open("w")
        self.addCleanup(log.close)
        def launch(command, **kwargs):
            return popen(command[:2] + ["-c", child_code] + command[-2:], **kwargs, stdout=log, stderr=log)
        with patch("search.internvl_native_worker.subprocess.Popen", side_effect=launch):
            result = NativeBackend(self.config)
        self.addCleanup(result.close)
        return result

    def test_real_physical_subprocess_matches_native_generation_stops_and_cache(self):
        import torch
        from deepseek_lmms_fixture import inputs
        from skills.deepseek_vl2_token import DeepSeekTokenController
        from search.lmms_gate import _mechanism_file
        backend = self.backend()
        self.assertEqual(backend.observed_model["physical_widths"], [48, 16, 32])
        self.assertEqual(len(backend.observed_model["router_dtypes"]), 1)
        receipt = self.root / "fixture-token-mechanism.jsonl"
        controller = DeepSeekTokenController(self.model, SELECTION, .5)
        # Natural native output tests the actual decoder, independent of a forced stop fixture.
        with torch.inference_mode(), patch.dict(os.environ, {"A4FVLMSP_TOKEN_RECEIPT": str(receipt)}):
            for nx in (1, 2):
                expected = controller.generate(inputs(nx), self.options)
                actual = backend.generate(**inputs(nx), **self.options)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                row = json.loads(receipt.read_text().splitlines()[-1])
                for key, value in controller.last_metrics.items():
                    self.assertEqual(row[key], value)
            # Force an ordinary byte token to verify generation stops at two tokens,
            # rather than merely trimming a five-token result after decoding.
            a = self.tokenizer.convert_tokens_to_ids("a")
            options = {**self.options, "suppress_tokens": [i for i in range(len(self.tokenizer)) if i != a]}
            self.assertEqual(backend.generate(**inputs(), **options).shape[1], 5)
            stopped = {**options, "stop_strings": ["aa"]}
            expected = controller.generate(inputs(), {**stopped, "tokenizer": self.tokenizer})
            actual = backend.generate(**inputs(), **stopped)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.tolist(), [[a, a]])
            row = json.loads(receipt.read_text().splitlines()[-1])
            self.assertEqual(row["output_tokens"], 2)
            self.assertEqual(row["decode_steps"], 1)
            self.assertEqual(_mechanism_file(self.root, "fixture", 4, .5), receipt)
            process = backend.process
            with self.assertRaisesRegex(RuntimeError, "greedy cached single-beam"):
                backend.generate(**inputs(), **{**self.options, "do_sample": True})
            self.assertIsNotNone(process.poll())
            self.assertIsNone(backend.connection)
            self.assertEqual(len(receipt.read_text().splitlines()), 4)

    def test_readiness_rejects_wrong_expert_width_router_precision_and_missing_kernels(self):
        from search.deepseek_native_worker import observed_topology
        ready = observed_topology(self.model)
        backend = NativeBackend.__new__(NativeBackend)
        backend.config = self.config
        backend.validate_ready(ready)
        for key, value in (("physical_widths", [48, 32, 32]), ("router_dtypes", {}),
                           ("quantized_modules", ["invented.kernel"])):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                backend.validate_ready({**ready, key: value})
        self.model.language.model.layers[1].mlp.experts[1].up_proj.out_features += 8
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            observed_topology(self.model)

    def test_weightless_lmms_proxy_preserves_native_context_config(self):
        from types import SimpleNamespace
        from search.deepseek_native_worker import create_native_proxy
        backend = SimpleNamespace(generate=lambda **kwargs: kwargs)
        with patch("search.deepseek_native_worker.NativeBackend", return_value=backend):
            model = create_native_proxy(self.config)
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(model.config.language_config.max_position_embeddings, 128)
        self.assertEqual(model.config.a4fvlm_stack_widths, [48, 16, 32])
        self.assertEqual(model.generate(max_new_tokens=5), {"max_new_tokens": 5})

    def test_lmms_keeps_native_prompt_pixels_budget_stops_and_decoding_in_parent(self):
        import torch
        from types import SimpleNamespace
        from PIL import Image
        from adapters.deepseek_vl2_dense import load_processor, prepare_image
        from search.deepseek_lmms import DeepSeekVL2LMMS
        descriptor = json.loads((ROOT / "configs/model-deepseek-vl2.json").read_text())
        processor = load_processor(descriptor)
        image = Image.new("RGB", (768, 384), (70, 90, 100))
        expected = prepare_image(processor, image, "What is shown?")
        calls, cache = [], []
        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([processor.tokenizer.encode("answer STOP trailing", add_special_tokens=False)])
        # The ordinary LMMS generation loop is tested with a weightless backend;
        # real cross-interpreter native generation is exercised separately above.
        model = DeepSeekVL2LMMS.__new__(DeepSeekVL2LMMS)
        model.processor, model.isolated, model.serving, model._device = processor, True, False, torch.device("cpu")
        model._model = SimpleNamespace(generate=generate,
            config=SimpleNamespace(language_config=SimpleNamespace(max_position_embeddings=4096)))
        model.task_dict = {"fixture": {"validation": [image]}}
        model.cache_hook = SimpleNamespace(add_partial=lambda *args: cache.append(args))
        options = {"max_new_tokens": 32, "temperature": 0, "until": ["STOP"]}
        original = copy.deepcopy(options)
        request = SimpleNamespace(args=("What is shown?", options, lambda doc: [doc], 0, "fixture", "validation"))
        self.assertEqual(model.generate_until([request]), ["answer "])
        self.assertEqual(options, original)
        self.assertEqual(len(cache), 1)
        for key, tensor in expected.items():
            torch.testing.assert_close(calls[0][key], tensor, rtol=0, atol=0)
        self.assertNotIn("tokenizer", calls[0])
        self.assertEqual(calls[0]["stop_strings"], ["STOP"])
        self.assertEqual(calls[0]["max_new_tokens"], 32)
        serving_calls = []
        def serving_generate(image, context, inputs, processor_arg, sampling):
            serving_calls.append(sampling)
            self.assertIs(processor_arg, processor)
            self.assertEqual(context, "What is shown?")
            for key, tensor in expected.items():
                torch.testing.assert_close(inputs[key], tensor, rtol=0, atol=0)
            return processor.tokenizer.encode("answer STOP trailing", add_special_tokens=False), {}
        model.serving = True
        model.backend = SimpleNamespace(generate=serving_generate)
        self.assertEqual(model.generate_until([request]), ["answer "])
        self.assertEqual(serving_calls[0]["max_tokens"], 32)
        self.assertEqual(serving_calls[0]["stop"], ["STOP"])
        self.assertEqual(serving_calls[0]["stop_token_ids"], [processor.tokenizer.eos_token_id])
        model._model.config.language_config.max_position_embeddings = 32
        with self.assertRaisesRegex(ValueError, "no truncation"):
            model.generate_until([request])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
