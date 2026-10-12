"""Native serving suffix, prompt identity and the shared timing boundary."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODEL = Path("models/InternVL3_5-14B")


class StreamingTests(unittest.TestCase):
    def test_warmup_and_measured_batches_keep_distinct_image_cache_identities(self):
        from search.worker import _run_streaming_vllm_benchmark
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            prompts = []
            for index in range(7):
                path = Path(folder) / f"image-{index}.png"
                Image.new("RGB", (3, 2), (index, 0, 0)).save(path)
                prompts.append({"native_prompt": "native <image>", "image_path": str(path),
                                "image_token_id": 98})
            for batch_size in (1, 2, 4, 8):
                cache, identities, generated_pixels = {}, [], []
                def render(requests):
                    request, = requests
                    identity = request["multi_modal_uuids"]["image"]
                    identities.append(identity)
                    pixel = cache.setdefault(identity, request["multi_modal_data"]["image"].getpixel((0, 0)))
                    return [{"prompt_token_ids": [1, 98, 2], "fixture_pixel": pixel[0]}]
                async def generate(inputs, *args):
                    generated_pixels.append(inputs["fixture_pixel"])
                    yield SimpleNamespace(outputs=[SimpleNamespace(text="answer", token_ids=[4])])
                engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()),
                    renderer=SimpleNamespace(render_cmpl=render), generate=generate)
                with self.subTest(batch_size=batch_size), patch.dict(
                        "sys.modules", {"vllm.renderers": SimpleNamespace(ChatParams=None)}):
                    result = asyncio.run(_run_streaming_vllm_benchmark(
                        engine=engine, prompts=prompts, sampling=None, warmup=5,
                        repeats=2, batch_size=batch_size))
                    self.assertEqual(generated_pixels, list(range(5)) + list(range(7)) * 2)
                    self.assertEqual(len(set(identities)), 19)
                    self.assertEqual(result[3], 14, "Warmup tokens must stay outside measured totals")

    def test_native_performance_worker_renders_completion_with_fresh_image_identity(self):
        from search.worker import _stream_vllm_request
        from PIL import Image

        async def generate(*args):
            yield SimpleNamespace(outputs=[SimpleNamespace(text="answer", token_ids=[4, 5])])

        observed = []
        def render(prompts):
            observed.extend(prompts)
            return [{"prompt_token_ids": [1, 98, 98, 2]}]
        engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()),
                                 renderer=SimpleNamespace(render_cmpl=render), generate=generate)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            Image.new("RGB", (40, 30)).save(image)
            with patch.dict("sys.modules", {"vllm.renderers": SimpleNamespace(ChatParams=None)}):
                result = asyncio.run(_stream_vllm_request(engine,
                    {"native_prompt": "native <image>", "image_path": str(image), "image_token_id": 98},
                    None, "request-unique"))
        self.assertEqual(result["image_tokens"], 2)
        self.assertEqual(result["tokens"], 2)
        self.assertEqual(observed[0]["prompt"], "native <image>")
        self.assertEqual(observed[0]["multi_modal_uuids"], {"image": "request-unique"})


@unittest.skipUnless((MODEL / "config.json").exists(), "native InternVL code/tokenizer fixture required")
class NativeServingTests(unittest.TestCase):
    model_path = MODEL
    model_config = "configs/model-internvl35.json"
    serving_config = "configs/serving-internvl35.json"

    def test_snapshot_uses_native_chat_stop_token(self):
        from adapters.internvl35_dense import load_processor
        from search.serving import snapshot
        dense = json.loads(Path(self.model_config).read_text())
        freeze = {"model": dense, "id": "test-stop-token"}
        with patch("search.serving.load_freeze", return_value=(freeze, {
                "worker_python": ".venv-runtime-internvl/bin/python"})):
            observed = snapshot(self.serving_config)
        processor = load_processor(dense)
        separator = processor.conversation_factory(processor.config.template).sep.strip()
        eos = processor.tokenizer.convert_tokens_to_ids(separator)
        self.assertEqual(observed["generation"]["stop_token_ids"], [eos])

    def test_both_backends_keep_native_prompt_and_suffix_and_audit_after_timing(self):
        import torch
        from PIL import Image
        from transformers import BatchFeature
        from adapters.internvl35_dense import load_processor, prepare_image
        from search.internvl_serving import InternVLSession
        from search.lmms_internvl_vllm_entry import InternVLVllmBackend
        from search.lmms_vllm_entry import compact_prompt
        from skills.quantization import digest

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        processor = load_processor({"path": str(self.model_path), "dtype": "bfloat16"})
        separator = processor.conversation_factory(processor.config.template).sep.strip()
        eos = processor.tokenizer.convert_tokens_to_ids(separator)
        self.assertEqual(eos, json.loads((self.model_path / "added_tokens.json").read_text())[separator])
        ids = processor.tokenizer("A red object.", add_special_tokens=False)["input_ids"] + [eos]
        options = {"max_tokens": 64, "temperature": 0., "ignore_eos": False,
                   "stop_token_ids": [eos], "repetition_penalty": 1.}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path = root / "image.png"
            Image.new("RGB", (896, 448), (220, 20, 10)).save(image_path)
            with Image.open(image_path) as image:
                native = prepare_image(processor, image, "Describe.")
            native_ids = native["input_ids"][0].tolist()
            before = native_ids.count(processor.image_token_id)
            after = before // 2
            compacted = compact_prompt(native_ids, processor.image_token_id, [(before, after)])
            output = SimpleNamespace(prompt_token_ids=compacted, outputs=[SimpleNamespace(token_ids=ids)])
            requests = {"sample_id": "fixture", "image": str(image_path), "prompt": "Describe."}
            captured, rows = [], []
            for backend in ("dense", "serving"):
                session = object.__new__(InternVLSession)
                session.snap, session.backend = {"generation": options}, backend
                session.processor, session.separator = processor, separator
                def generate(**kwargs):
                    captured.append(kwargs)
                    return torch.tensor([ids])
                session.model = SimpleNamespace(generate=generate)
                actual = object.__new__(InternVLVllmBackend)
                actual.descriptor = {"token_keep_ratio": .5}
                actual.index, actual.trace_offset = 0, 0
                actual.receipt = root / "token-mechanism.jsonl"
                actual.trace = root / "selection.jsonl"
                actual.trace.write_text(json.dumps({
                    "position_type": "1d_compacted", "positions_after": after,
                    "image_tokens_before": before, "image_tokens_after": after,
                    "tiles_selected": before // 256, "token_keep_ratio": .5,
                    "selected_indices_sha256": "fixture-selection",
                }) + "\n")
                def infer(image, prompt, sampling):
                    self.assertEqual(prompt, processor.unexpanded_prompt("Describe."))
                    self.assertEqual(sampling, {**options, "detokenize": False})
                    return output
                actual.infer = infer
                actual.engine = SimpleNamespace(reset_prefix_cache=lambda: True)
                session.native = actual
                original_audit = actual.audit
                with patch("torch.cuda.synchronize"), \
                     patch.object(BatchFeature, "to", lambda self, *args, **kwargs: self), \
                     patch("search.internvl_serving.perf_counter", side_effect=[10., 12.]) as timer, \
                     patch.object(actual, "audit") as audit:
                    def audit_after_timer(*args):
                        self.assertEqual(timer.call_count, 2)
                        return original_audit(*args)
                    audit.side_effect = audit_after_timer
                    rows.append(session.generate(requests))
                    if backend == "serving":
                        audit.assert_called_once()
                        self.assertEqual(timer.call_count, 2)
            for row in rows:
                self.assertEqual(row["output_text"], "A red object.")
                self.assertEqual(row["output_tokens"], len(ids))
                self.assertEqual(row["native_prompt_sha256"], digest(native_ids))
                self.assertEqual(row["end_to_end_seconds"], 2.)
            self.assertEqual(captured[0]["input_ids"][0].tolist(), native_ids)
            self.assertEqual(captured[0]["eos_token_id"], [eos])
            self.assertIsNone(rows[0]["token_mechanism"])
            self.assertTrue(rows[1]["token_mechanism"]["prompt_verified"])


if __name__ == "__main__":
    unittest.main()
