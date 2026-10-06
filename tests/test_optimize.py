"""Exercise artifact reuse and checkpoint replay without a model or GPU."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search import optimize as opt
from search.benchmark import summarize
from search.run import evaluate_candidate


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.baseline = self.results / "dense/quality"
        self.baseline.mkdir(parents=True)
        (self.root / "image.jpg").touch()
        self.rows = [
            dict(
                stable_row_id=str(i),
                primary_category="task",
                logical_question="Answer?",
                answers=["yes"],
                scoring_type="deterministic_rule_or_structure",
                media={"logical_path": "image.jpg"},
            )
            for i in range(40)
        ]
        for stage in ("search", "dev"):
            self.jsonl(self.root / f"{stage}.jsonl", self.rows)
            self.jsonl(self.baseline / f"{stage}.jsonl", self.outputs(40))
        self.snap = {
            "device": 0,
            "model": {},
            "identity": {"model": "fixed"},
            "benchmark_code": "timing-v1",
            "evaluation": {
                "data": {
                    "search": "search.jsonl",
                    "dev": "dev.jsonl",
                    "media_root": ".",
                },
                "generation": {},
                "scoring": {
                    "baseline": "dense",
                    "maximum_response_tokens": 64,
                    "negation_window_tokens": 3,
                },
                "quality": {
                    "minimum_retention": 0.95,
                    "search_direct_pass_minimum_retention": 0.975,
                },
            },
            "vision": {
                "selection": {
                    "score_dtype": "float32",
                    "coverage_grid": {"rows": 6, "columns": 6},
                },
                "search": {
                    "start_keep_ratio": 0.95,
                    "minimum_keep_ratio": 0.85,
                    "keep_ratio_step": 0.05,
                    "consecutive_dev_failures": 2,
                },
            },
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(opt, "PROJECT_ROOT", self.root))
        self.stack.enter_context(patch.object(opt, "RESULTS", self.results))
        self.stack.enter_context(
            patch.object(opt, "hardware", return_value="GPU-test, driver")
        )
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        prep = self.results / "prep"
        prep.mkdir()
        opt.save_json(
            prep / "request.json",
            {"identity": self.snap["identity"], "operation": "prepare"},
        )
        opt.save_json(
            prep / "geometry.json",
            [{"sample_id": str(i), "rows": 20, "columns": 20} for i in range(40)],
        )

    @staticmethod
    def jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def outputs(correct):
        return [
            {"sample_id": str(i), "output_text": "yes" if i < correct else "no"}
            for i in range(40)
        ]

    def artifact(self, ratio=0.95, correct=40, dev=None):
        directory = self.results / f"keep-{ratio}"
        (directory / "quality").mkdir(parents=True)
        request = {
            "operation": "candidate",
            "identity": self.snap["identity"],
            "selection": self.snap["vision"]["selection"],
            "keep_ratio": ratio,
        }
        opt.save_json(directory / "request.json", request)
        self.jsonl(directory / "quality/search.jsonl", self.outputs(correct))
        if dev is not None:
            self.jsonl(directory / "quality/dev.jsonl", self.outputs(dev))
        # An obsolete summary must not dictate the current decision.
        opt.save_json(directory / "quality/summary.json", {"action": "reject"})
        return directory

    def performance(self, ratio):
        directory = self.results / (f"keep-{ratio}" if ratio is not None else "control")
        directory.mkdir(exist_ok=True)
        request_path = directory / "request.json"
        request = (
            opt.read(request_path)
            if request_path.exists()
            else {"identity": self.snap["identity"], "keep_ratio": ratio}
        )
        request.update(
            hardware="GPU-test, driver",
            benchmark_code=self.snap["benchmark_code"],
            benchmark_protocol={"warmup": 5, "repeats": 1},
        )
        opt.save_json(request_path, request)
        (directory / "benchmark").mkdir()
        records = [
            {
                "sample_id": str(i),
                "round": 1,
                "performance": {
                    "end_to_end_seconds": 1.0,
                    "generate_seconds": 0.9,
                    "output_tokens": 2,
                    "peak_allocated_bytes": 100,
                    "vision_encode_seconds": 0.2,
                    "pruning_seconds": 0.01,
                    "prefill_seconds": 0.3,
                    "decode_seconds": 0.3,
                    "decode_tokens": 1,
                },
            }
            for i in range(40)
        ]
        self.jsonl(directory / "benchmark/metrics.jsonl", records)
        opt.save_json(
            directory / "benchmark/summary.json",
            {
                "warmup_samples": 5,
                "rounds": [summarize(records)],
                "mean_round_seconds": 40.0,
            },
        )
        return directory

    def test_existing_performance_does_not_require_new_metadata_or_phases(self):
        self.artifact()
        directory = self.performance(0.95)
        paths = opt.evidence_paths([self.results])
        self.assertIsNotNone(
            opt.inspect_benchmark(self.snap, 0.95, paths, "GPU-test, driver")
        )
        self.assertIsNotNone(
            opt.inspect_benchmark(self.snap, 0.95, paths, "GPU-other, driver")
        )
        self.assertIsNotNone(opt.inspect_benchmark(self.snap, 0.95, paths, None))
        records = opt.lines(directory / "benchmark/metrics.jsonl")
        del records[0]["performance"]["prefill_seconds"]
        self.jsonl(directory / "benchmark/metrics.jsonl", records)
        self.assertIsNotNone(
            opt.inspect_benchmark(self.snap, 0.95, paths, "GPU-test, driver")
        )

    def test_all_reused_and_invocation_limit_do_not_launch(self):
        self.artifact()
        self.performance(0.95)
        self.performance(None)
        destination = self.root / "run"
        destination.mkdir()
        with (
            patch.object(opt, "launch", side_effect=AssertionError("Must reuse")),
            patch.object(opt, "hardware", side_effect=AssertionError("No GPU query")),
        ):
            first = opt.execute(self.snap, destination, 0)
            second = opt.execute(self.snap, destination, 0)
        self.assertEqual(first["stop_reason"], "invocation_limit")
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertTrue(first["candidates"][0]["reused_all"])
        self.assertEqual(first["attempts"], [])
        self.snap["vision"]["search"]["minimum_keep_ratio"] = 0.95
        with (
            patch.object(opt, "launch", side_effect=AssertionError("Must reuse")),
            patch.object(opt, "hardware", side_effect=AssertionError("No GPU query")),
        ):
            completed = opt.execute(self.snap, destination)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["speedup"], 1.0)

    def test_legacy_records_reused_without_request_metadata(self):
        directory = self.artifact()
        self.performance(0.95)
        (directory / "request.json").unlink()
        legacy = self.results / "keep-95"
        directory.rename(legacy)
        paths = opt.evidence_paths([self.results])
        sources, quality = opt.inspect_quality(self.snap, 0.95, paths)
        self.assertEqual(quality["action"], "pass")
        self.assertEqual(sources["search"], str(legacy / "quality/search.jsonl"))
        self.assertIsNotNone(opt.inspect_benchmark(self.snap, 0.95, paths))

    def test_no_speedup_does_not_stop_and_summary_does_not_copy_checkpoint(self):
        for ratio in (0.95, 0.9, 0.85):
            self.artifact(ratio)
            self.performance(ratio)
        self.performance(None)
        destination = self.root / "run"
        destination.mkdir()
        with patch.object(opt, "launch", side_effect=AssertionError("Must reuse")):
            state = opt.execute(self.snap, destination)
        self.assertEqual(state["stop_reason"], "declared_range_exhausted")
        summary = opt.read(destination / "summary.json")
        self.assertEqual(summary["evaluated_candidates"], 3)
        self.assertEqual(summary["reused_candidates"], 3)
        self.assertEqual(summary["speedup"], 1.0)
        self.assertNotIn("candidates", summary)
        self.assertNotIn("control", summary)
        self.assertEqual(
            opt.read(destination / summary["checkpoint"]), json.loads(json.dumps(state))
        )

    def test_physical_duplicates_do_not_increment_failure_count(self):
        self.snap["vision"]["search"].update(
            start_keep_ratio=0.9, minimum_keep_ratio=0.8, keep_ratio_step=0.01
        )
        opt.save_json(
            self.results / "prep/geometry.json",
            [{"sample_id": str(i), "rows": 1, "columns": 10} for i in range(40)],
        )
        self.artifact(0.9, 37, 37)
        self.artifact(0.8, 37, 37)
        destination = self.root / "run"
        destination.mkdir()
        with patch.object(opt, "launch", side_effect=AssertionError("Must reuse")):
            state = opt.execute(self.snap, destination)
        self.assertEqual(state["stop_reason"], "consecutive_dev_failures")
        evaluated = [row for row in state["candidates"] if "quality" in row]
        self.assertEqual([row["confirmed_failure_streak"] for row in evaluated], [1, 2])
        self.assertEqual(
            opt.read(destination / "summary.json")["duplicate_configurations"], 9
        )

    def test_candidate_request_preserves_precision_without_preliminary_trial(self):
        self.snap["vision"]["selection"]["score_dtype"] = "float16"
        destination = self.root / "run"
        destination.mkdir()
        requests = []

        def capture(request, directory):
            requests.append(request)
            raise RuntimeError("Stop before inference")

        with patch.object(opt, "launch", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "Stop before inference"):
                opt.execute(self.snap, destination)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["operation"], "candidate")
        self.assertEqual(requests[0]["selection"]["score_dtype"], "float16")
        self.assertEqual(requests[0]["keep_ratio"], 0.95)

    def test_complete_raw_outputs_rescored_without_receipt(self):
        directory = self.artifact()
        sources, quality = opt.inspect_quality(
            self.snap, 0.95, opt.evidence_paths([self.results])
        )
        self.assertEqual(quality["action"], "pass")
        self.assertEqual(sources, {"search": str(directory / "quality/search.jsonl")})

    def test_incompatible_and_partial_outputs_not_reused(self):
        directory = self.artifact()
        request = opt.read(directory / "request.json")
        request["selection"] = {"score_dtype": "float16"}
        opt.save_json(directory / "request.json", request)
        paths = opt.evidence_paths([self.results])
        self.assertEqual(opt.inspect_quality(self.snap, 0.95, paths), ({}, None))
        request["selection"] = self.snap["vision"]["selection"]
        opt.save_json(directory / "request.json", request)
        self.jsonl(directory / "quality/search.jsonl", self.outputs(40)[:-1])
        self.assertEqual(opt.inspect_quality(self.snap, 0.95, paths), ({}, None))

    def test_search_reuse_generates_only_dev(self):
        directory = self.artifact(correct=38)
        sources, quality = opt.inspect_quality(
            self.snap, 0.95, opt.evidence_paths([self.results])
        )
        self.assertIsNone(quality)
        evaluation = self.snap["evaluation"]
        evaluation["scoring"]["baseline"] = str(self.baseline.parent)
        calls = []

        def generate(model, processor, samples, **kwargs):
            calls.extend(row["stable_row_id"] for row in samples)
            return [
                {"sample_id": row["stable_row_id"], "output_text": "yes"}
                for row in samples
            ]

        with (
            patch("search.run.PROJECT_ROOT", self.root),
            patch("search.run.generate_outputs", generate),
        ):
            result = evaluate_candidate(
                None,
                None,
                evaluation,
                output_dir=self.root / "new",
                reuse_outputs=sources,
            )
        self.assertEqual(calls, [str(i) for i in range(40)])
        self.assertEqual(result["action"], "pass")
        self.assertFalse((self.root / "new/search.jsonl").exists())
        self.assertTrue((self.root / "new/dev.jsonl").exists())
        self.assertTrue((directory / "quality/search.jsonl").exists())

    def test_reused_dev_failures_reconstruct_stop_without_worker(self):
        self.artifact(0.95, 37, 37)
        self.artifact(0.90, 37, 37)
        destination = self.root / "run"
        destination.mkdir()
        with (
            patch.object(opt, "launch", side_effect=AssertionError("Must reuse")),
            patch.object(opt, "hardware", side_effect=AssertionError("No GPU query")),
        ):
            state = opt.execute(self.snap, destination)
            again = opt.execute(self.snap, destination)
        self.assertEqual(state["stop_reason"], "consecutive_dev_failures")
        self.assertEqual(len(state["candidates"]), 2)
        self.assertEqual(again["candidates"], state["candidates"])
        self.assertEqual(state["attempts"], [])

    def test_reused_pass_resets_failure_streak(self):
        self.artifact(0.95, 37, 37)
        self.artifact(0.90)
        self.performance(0.90)
        self.performance(None)
        self.artifact(0.85, 37, 37)
        destination = self.root / "run"
        destination.mkdir()
        with (
            patch.object(opt, "launch", side_effect=AssertionError("Must reuse")),
            patch.object(opt, "hardware", side_effect=AssertionError("No GPU query")),
        ):
            state = opt.execute(self.snap, destination)
        self.assertEqual(state["stop_reason"], "declared_range_exhausted")
        self.assertEqual(
            [row["confirmed_failure_streak"] for row in state["candidates"]], [1, 0, 1]
        )

    def test_failure_checkpoint_preserves_quality_and_retries_only_benchmark(self):
        self.artifact()
        destination = self.root / "run"
        destination.mkdir()
        calls = []

        def fail(request, directory):
            calls.append(request["operation"])
            raise RuntimeError("OOM")

        with patch.object(opt, "launch", fail):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                opt.execute(self.snap, destination, 1)
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                opt.execute(self.snap, destination, 1)
        self.assertEqual(calls, ["benchmark", "benchmark"])
        state = opt.read(destination / "state.json")
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["candidates"], [])
        self.assertFalse(state["resource_accounting_complete"])

    def test_plan_is_deterministic_and_never_launches(self):
        self.artifact()
        paths = opt.evidence_paths([self.results])
        with (
            patch.object(opt, "hardware", side_effect=AssertionError("No GPU")),
            patch.object(opt, "launch", side_effect=AssertionError("No worker")),
        ):
            self.assertEqual(opt.plan(self.snap, paths), opt.plan(self.snap, paths))
        self.assertEqual(
            opt.plan(self.snap, paths)["candidates"][0]["quality_work"], "reuse"
        )

    def test_lock_excludes_other_coordinator(self):
        path = self.root / "lock"
        with opt.lock(path):
            with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                with opt.lock(path):
                    self.fail("Lock was acquired twice")

    def test_failed_worker_keeps_gpu_lock_until_exit(self):
        # The child must inherit the lock even if a coordinator is killed.
        lock_path = self.root / "gpu.lock"
        with opt.lock(lock_path):
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                pass_fds=tuple(opt.LOCK_FDS),
            )
            try:
                self.assertTrue(opt.LOCK_FDS)
                with self.assertRaises(RuntimeError):
                    with opt.lock(lock_path):
                        self.fail("Worker lease was not held")
            finally:
                child.terminate()
                child.wait(timeout=5)
        with opt.lock(lock_path):
            pass

    def test_worker_failure_has_receipt_and_releases_process(self):
        destination = self.root / "bad-worker"
        with self.assertRaisesRegex(RuntimeError, "Worker failed"):
            # Use the real project as subprocess cwd; the unknown operation never loads a model.
            with patch.object(opt, "PROJECT_ROOT", Path(__file__).resolve().parents[1]):
                opt.launch({"operation": "unknown", "device": None}, destination)
        result = opt.read(destination / "result.json")
        self.assertEqual(result["failure"], "execution_error")
        self.assertEqual(result["gpu_reserved_seconds"], 0)


class ImportTests(unittest.TestCase):
    def test_planner_does_not_import_inference_packages(self):
        script = "import sys; import search.optimize; assert not {'torch', 'transformers', 'PIL'} & sys.modules.keys()"
        subprocess.run(
            [sys.executable, "-B", "-c", script],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )


if __name__ == "__main__":
    unittest.main()
