"""Frozen inference and comparable end-to-end measurements, without GPU work."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search import serving
from search.worker import save_json
from skills.quantization import digest


class ServingTests(unittest.TestCase):
    def test_generate_preserves_native_pixels_through_backend_exif_normalization(self):
        from PIL import Image, ImageOps
        import torch

        # Exercise the actual Session call site. vLLM normalizes EXIF even for
        # already decoded PIL inputs, while the native processor uses the raster.
        for orientation in (2, 6):
            with self.subTest(orientation=orientation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / "image.png"
                image = Image.new("RGB", (4, 3))
                image.putpixel((0, 0), (255, 0, 0))
                exif = image.getexif()
                exif[274] = orientation
                image.save(path, exif=exif)
                pixels = image.tobytes()
                trace = root / "selection.jsonl"
                trace.write_text("")

                class Processor:
                    image_token_id = 7
                    tokenizer = SimpleNamespace(decode=lambda *a, **k: "answer")

                    def apply_chat_template(self, *a, **k):
                        return "native prompt"

                    def __call__(self, *, text, images, return_tensors):
                        return {"input_ids": torch.tensor([[1] + [7] * images[0].width + [2]])}

                observed = []

                class Engine:
                    def reset_prefix_cache(self):
                        return True

                    def generate(self, requests, *a, **k):
                        normalized = ImageOps.exif_transpose(requests[0]["multi_modal_data"]["image"])
                        observed.append((normalized.size, normalized.tobytes()))
                        before = normalized.width
                        trace.write_text(json.dumps({"image_tokens_before": before,
                                                     "image_tokens_after": before - 1}) + "\n")
                        return [SimpleNamespace(prompt_token_ids=[1] + [7] * (before - 1) + [2],
                                                outputs=[SimpleNamespace(token_ids=[3])])]

                session = serving.Session.__new__(serving.Session)
                session.backend, session.snap = "serving", {"generation": {}}
                session.processor, session.engine = Processor(), Engine()
                session.trace, session.trace_offset, session.index = trace, 0, 0
                with patch("torch.cuda.synchronize"), patch.dict("sys.modules", {
                    "vllm": SimpleNamespace(SamplingParams=lambda **k: None),
                }):
                    result = session.generate({"image": str(path), "prompt": "question", "sample_id": "a"})
                self.assertEqual(observed, [((4, 3), pixels)])
                self.assertTrue(result["token_mechanism"]["prompt_verified"])
                self.assertEqual(result["native_prompt_tokens"], 6)

    def test_plan_uses_qualified_freeze_and_full_search_without_gpu_or_result_writes(self):
        with patch("search.serving.hardware", side_effect=AssertionError("GPU query")), \
                patch("search.serving.launch", side_effect=AssertionError("Worker")), \
                patch("search.serving.save_json", side_effect=AssertionError("write")):
            snap = serving.snapshot("configs/serving.json", benchmark=True)
        self.assertEqual(snap["candidate_id"], "serving-e6934a2234f9a6c82279")
        self.assertEqual(len(snap["requests"]), 128)
        self.assertEqual(snap["model"]["runtime"]["vllm_max_model_len"], 32768)
        self.assertEqual(snap["model"]["token_keep_ratio"], .8)
        self.assertTrue(all(len(row["image_sha256"]) == 64 for row in snap["requests"]))

    def test_changed_or_unqualified_freeze_is_rejected_before_loading(self):
        original_read = serving.read
        config = original_read("configs/serving.json")
        for change in ({"status": "frozen_for_lmms"}, {"stack_digest": "wrong"},
                       {"model": {}}, {"source_digest": "wrong"}):
            def changed(path):
                value = original_read(path)
                return value | change if Path(path).name == "serving-freeze.json" else value
            with self.subTest(change=change), patch("search.serving.read", side_effect=changed):
                with self.assertRaises(ValueError):
                    serving.load_freeze(config["freeze"])

    def test_generation_uses_identical_greedy_limits_and_dense_eos(self):
        gen = {"max_new_tokens": 64, "do_sample": False, "ignore_eos": False}
        options = serving.generation_options(gen, [1, 2])
        self.assertEqual(options["stop_token_ids"], [1, 2])
        self.assertEqual(options["max_tokens"], 64)
        self.assertEqual(options["temperature"], 0)
        self.assertEqual(serving.generation_options(gen | {"ignore_eos": True}, [1, 2])["stop_token_ids"], [])
        for bad in ({"do_sample": True}, {"max_new_tokens": 0}, {"max_new_tokens": True}):
            with self.assertRaises(ValueError):
                serving.generation_options(gen | bad, [1, 2])

    def pair(self):
        rows = [{"sample_id": "a", "round": 1, "native_prompt_sha256": "same",
                 "output_tokens": 10, "end_to_end_seconds": 2.}]
        report = {"hardware": "gpu", "snapshot_digest": "snapshot",
                  "timing_scope": serving.TIMING_SCOPE, "summary": serving.summarize(rows)}
        second = copy.deepcopy(rows)
        second[0].update(end_to_end_seconds=1., output_tokens=5)
        return (report, rows), (report | {"summary": serving.summarize(second)}, second)

    def test_latency_gain_and_token_throughput_are_separate_for_shorter_answers(self):
        result = serving.compare(*self.pair())
        self.assertEqual(result["end_to_end_speedup"], 2.)
        self.assertEqual(result["output_token_throughput_ratio"], 1.)

    def test_comparison_rejects_different_hardware_scope_snapshot_or_native_prompt(self):
        for field in ("hardware", "timing_scope", "snapshot_digest"):
            dense, selected = self.pair()
            selected[0][field] = "different"
            with self.assertRaisesRegex(ValueError, "differ"):
                serving.compare(dense, selected)
        dense, selected = self.pair()
        selected[1][0]["native_prompt_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differ"):
            serving.compare(dense, selected)

    def fixture(self, folder):
        root = Path(folder)
        freeze_path = root / "serving-freeze.json"
        save_json(freeze_path, {"lmms": {"retention": .96, "minimum_retention": .95}})
        snap = {"mode": "benchmark", "requests": [{"sample_id": "a"}], "repeats": 1,
                "config": {"output": str(root / "job"), "device": 0, "freeze": str(freeze_path)},
                "candidate_id": "selected", "freeze_digest": "hash",
                "dense_worker_python": "/dense/python", "worker_python": "/serving/python",
                "model": {"token_keep_ratio": .8, "runtime": {"kv_cache_dtype": "auto"}}}
        return snap

    def finish(self, request, directory):
        directory.mkdir(parents=True)
        snap, backend = request["snapshot"], request["backend"]
        save_json(directory / "request.json", request)
        save_json(directory / "result.json", {"status": "completed"})
        save_json(directory / "runtime.json", {"model": snap["model"], "observed_model": {
            "token_pruning_enabled": True, "token_runtime": {"keep_ratio": .8}, "kv_cache_dtypes": ["auto"]}})
        row = {"sample_id": "a", "round": 1, "native_prompt_sha256": "same", "output_text": "answer",
               "output_tokens": 10, "end_to_end_seconds": 2. if backend == "dense" else 1.,
               "token_mechanism": {"prompt_verified": True, "images": [{"token_keep_ratio": .8,
                   "selected_indices_sha256": "indices", "mrope_positions": 8, "image_tokens_after": 8}]}}
        (directory / "outputs.jsonl").write_text(json.dumps(row) + "\n")
        save_json(directory / "summary.json", {"status": "completed", "backend": backend,
            "snapshot_digest": digest(snap), "hardware": request["hardware"],
            "timing_scope": serving.TIMING_SCOPE, "summary": serving.summarize([row])})

    def test_llava_receipt_requires_compacted_1d_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            snap = self.fixture(folder)
            snap["model"]["adapter"] = "llava_onevision"
            directory = Path(folder) / "attempt"
            self.finish({"snapshot": snap, "backend": "serving", "hardware": "gpu"}, directory)
            with self.assertRaisesRegex(ValueError, "Token/prompt"):
                serving.complete_output(directory, snap, "serving")
            path = directory / "outputs.jsonl"
            row = json.loads(path.read_text())
            image = row["token_mechanism"]["images"][0]
            image.pop("mrope_positions")
            image.update(position_type="1d_compacted", positions_after=8)
            path.write_text(json.dumps(row) + "\n")
            serving.complete_output(directory, snap, "serving")
            image["positions_after"] = 9
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Token/prompt"):
                serving.complete_output(directory, snap, "serving")

    def test_coordinator_reuses_complete_workers_and_rejects_snapshot_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            snap = self.fixture(folder)
            calls = []
            def launch(request, directory):
                calls.append(request["backend"])
                self.finish(request, directory)
            with patch("search.serving.snapshot", return_value=snap), \
                    patch("search.serving.hardware", return_value="same-gpu"), \
                    patch("search.serving.launch", side_effect=launch):
                result = serving.execute("unused.json", benchmark=True)
                self.assertEqual(result["comparison"]["end_to_end_speedup"], 2.)
                serving.execute("unused.json", benchmark=True)
                self.assertEqual(calls, ["dense", "serving"])
                snap["repeats"] = 2
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    serving.execute("unused.json", benchmark=True)

    def test_execution_failure_stops_and_resume_only_runs_missing_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            snap = self.fixture(folder)
            calls = []
            def launch(request, directory):
                calls.append(request["backend"])
                if calls == ["dense", "serving"]:
                    raise RuntimeError("engine failed")
                self.finish(request, directory)
            with patch("search.serving.snapshot", return_value=snap), \
                    patch("search.serving.hardware", return_value="same-gpu"), \
                    patch("search.serving.launch", side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, "engine failed"):
                    serving.execute("unused.json", benchmark=True)
                state = serving.read(Path(folder) / "job/benchmark/state.json")
                self.assertEqual(state["status"], "blocked")
                serving.execute("unused.json", benchmark=True)
                self.assertEqual(calls, ["dense", "serving", "serving"])

    def test_partial_outputs_or_missing_runtime_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            snap = self.fixture(folder)
            directory = Path(folder) / "attempt"
            self.finish({"snapshot": snap, "backend": "serving", "hardware": "gpu"}, directory)
            serving.complete_output(directory, snap, "serving")
            original = (directory / "outputs.jsonl").read_text()
            (directory / "outputs.jsonl").write_text(original * 2)
            with self.assertRaises(ValueError):
                serving.complete_output(directory, snap, "serving")
            (directory / "outputs.jsonl").write_text(original)
            (directory / "runtime.json").unlink()
            with self.assertRaises(FileNotFoundError):
                serving.complete_output(directory, snap, "serving")


if __name__ == "__main__":
    unittest.main()
