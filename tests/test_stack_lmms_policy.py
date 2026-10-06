"""Stack and serving quality decisions must come from full native LMMS."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from search.stack import _completed_handoff, _lmms_blocked, _lmms_quality_rejected, execute, snapshot
from search.stack_acceleration import execute as accelerate, _reuse
from search.serving_lmms import qualified
from search.worker import run_stack_acceleration_candidate, run_stack_candidate, save_json
from skills.stack import compare_candidates, evidence_config, make_candidate, validate_config
from skills.stack_runtime import performance_ready


class FullLmmsPolicyTests(unittest.TestCase):
    def candidate(self):
        return make_candidate(model_digest="m", sp=None, quantization=None,
            token={"candidate_id": "token", "config_digest": "t"}, token_keep_ratio=.5)

    def snap(self, candidate):
        return {"schema": "a4fvlm.stack.v1", "config_path": "/tmp/job.json", "status": "ready",
                "model_digest": "m", "model": {}, "evaluation": {}, "token_config": {"selection": {}},
                "blockers": [], "sources": {}, "candidates": [candidate], "initial_candidates": [candidate],
                "config": {"device": 0, "limits": {"initial_candidates": 1, "total_candidates": 1},
                           "lmms": {"enabled": True}}}

    def test_stack_worker_never_scores_search_or_dev(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("search.context.model_adapter", return_value=SimpleNamespace(resolve=lambda _: lambda *a: object())), \
                patch("search.model.load_model", return_value=(SimpleNamespace(), object())), \
                patch("search.run.evaluate_candidate", side_effect=AssertionError("local quality")) as quality, \
                patch("search.benchmark.benchmark", return_value={"mean_round_seconds": 1, "rounds": []}) as benchmark:
            result = run_stack_candidate({"candidate": self.candidate(), "probe_output": folder,
                                          "model": {}, "selection": {}, "evaluation": {}})
            quality.assert_not_called()
            benchmark.assert_called_once()
            self.assertEqual(result["probe_status"], "mechanism_ready")
            self.assertEqual(json.loads((Path(folder) / "probe.json").read_text())["quality"]["status"], "pending_full_lmms")
            self.assertFalse((Path(folder) / "quality").exists())

    def test_legacy_dev_rejection_reopens_same_candidate_without_quality_recovery(self):
        candidate = self.candidate()
        snap = self.snap(candidate)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "candidates" / candidate["candidate_id"]
            directory.mkdir(parents=True)
            identity = {k: candidate[k] for k in ("candidate_id", "candidate_key")}
            old_probe = identity | {"status": "quality_rejected", "quality": {"dev_retention": .1}}
            save_json(directory / "probe.json", old_probe)
            save_json(directory / "result.json", identity | {"probe_status": "quality_rejected"})
            save_json(root / "state.json", {"status": "completed", "stop_reason": "recovery_ladder_exhausted"})

            def worker(request, attempt):
                attempt.mkdir(parents=True, exist_ok=True)
                self.assertEqual(request["candidate"]["candidate_key"], candidate["candidate_key"])
                probe = identity | {"schema": "a4fvlm.stack-result.v1", "status": "ready_for_lmms",
                    "fresh_evidence": True, "mechanism": {"status": "passed"}, "reload": {"status": "passed"},
                    "benchmark": {"status": "passed", "mean_seconds": 1, "peak_allocated_bytes": 1}}
                save_json(attempt / "probe.json", probe)
                return {"probe_status": "ready_for_lmms"}

            with patch("search.stack.snapshot", return_value=snap), \
                    patch("search.stack.hardware", return_value={}), \
                    patch("search.stack.launch", side_effect=worker) as launch, \
                    patch("search.stack._run_lmms_gate", return_value={"status": "blocked", "reason": "API mismatch"}) as lmms, \
                    patch("search.stack._recovery_neighbors", side_effect=AssertionError("no quality decision")):
                result = execute("ignored", output=root)
            launch.assert_called_once()
            lmms.assert_called_once()
            self.assertEqual(result["stop_reason"], "blocked_preflight")
            self.assertEqual(json.loads((directory / "legacy-local-quality/probe.json").read_text()), old_probe)
            self.assertEqual(json.loads((root / "state.json").read_text())["quality_policy"], "full_lmms")
            self.assertEqual(json.loads((root / "legacy-local-quality/state.json").read_text())["stop_reason"], "recovery_ladder_exhausted")

    def test_local_pass_cannot_select_stack_without_full_lmms(self):
        row = {"status": "qualified", "quality": {"dev_retention": 1.2},
               "benchmark": {"mean_seconds": 1, "peak_allocated_bytes": 1}}
        self.assertEqual(compare_candidates([row]), [])
        receipt = {"status": "blocked", "quality_gate": "quality_rejected", "retention": .9}
        self.assertFalse(_lmms_quality_rejected(receipt))
        self.assertTrue(_lmms_blocked(receipt))

    def test_new_stack_config_cannot_weaken_full_gate_or_drop_tasks(self):
        cfg = json.loads(Path("configs/stack.json").read_text())
        self.assertNotIn("quality", cfg)
        validate_config(cfg)
        cfg["lmms"]["minimum_retention"] = .94
        with self.assertRaises(ValueError):
            validate_config(cfg)
        cfg["lmms"]["minimum_retention"] = .95
        cfg["lmms"]["tasks"] = cfg["lmms"]["tasks"][:1]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_runtime_and_kv_worker_only_measure_requested_profiles(self):
        for operation in ("stack_runtime_candidate", "stack_kv_cache_candidate"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                artifact = root / "artifact"
                artifact.mkdir()
                benchmark = {"status": "passed", "summary": {"elapsed_seconds": 1,
                    "request_count": 128, "output_token_throughput": 10}, "runtime": {}}

                def measure(request):
                    self.assertNotIn("quality_validation", request)
                    self.assertEqual(request["benchmark_split"], "search")
                    path = Path(request["probe_output"]) / "benchmark.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_json(path, benchmark)
                    return {"probe_status": "vllm_benchmark_passed", "report": str(path)}

                with patch("skills.stack_vllm.validate_stack_artifact"), \
                        patch("search.worker.run_quantization_vllm_benchmark", side_effect=measure) as run:
                    result = run_stack_acceleration_candidate({"operation": operation, "probe_output": str(root),
                        "artifact": str(artifact), "evaluation": {}, "profiles": [{"name": "search",
                            "limit": 128, "warmup": 5, "repeats": 3, "batch_size": 1}],
                        # Even a stale historical request must not launch Dev.
                        "quality_validation": {"minimum_retention": .95, "limit": 256}})
                run.assert_called_once()
                self.assertEqual(result["probe_status"], "performance_ready")
                probe = json.loads((root / "probe.json").read_text())
                self.assertTrue(performance_ready(probe))
                self.assertTrue(qualified(probe))
                self.assertEqual(probe["quality"]["status"], "pending_full_lmms")
                self.assertFalse((root / "vllm-quality-dev256").exists())

    def test_runtime_coordinator_does_not_request_quality_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = {"id": "runtime", "candidate_key": "k", "runtime": {}}
            snap = {"kind": "runtime", "config": {"output": str(root), "device": 0,
                        "quality": {"minimum_dev_retention": .95}},
                    "parent": {"path": "parent", "digest": "p", "artifact": "a", "freeze": {}},
                    "benchmark": {"search_limit": 128, "dev_limit": 256}, "evaluation": {},
                    "token_config": {"selection": {}}, "candidates": [candidate]}

            def worker(request, attempt):
                attempt.mkdir(parents=True, exist_ok=True)
                self.assertNotIn("quality_validation", request)
                self.assertEqual([p["split"] for p in request["profiles"]], ["search"])
                save_json(attempt / "probe.json", {"status": "runtime_blocked"})

            with patch("search.stack_acceleration.snapshot", return_value=snap), \
                    patch("search.stack_acceleration.hardware", return_value={}), \
                    patch("search.stack_acceleration.launch", side_effect=worker) as run:
                accelerate("ignored", kind="runtime")
            run.assert_called_once()

    def test_runtime_dev_rejection_can_reuse_complete_timing_but_missing_timing_cannot(self):
        probe = {"status": "quality_rejected", "candidate_key": "k", "quality": {"status": "rejected", "retention": .1},
                 "benchmarks": {"search": {"status": "passed", "summary": {
                     "elapsed_seconds": 1, "request_count": 128, "output_token_throughput": 10}}}}
        self.assertTrue(qualified(probe))
        self.assertFalse(qualified({**probe, "benchmarks": {}}))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_json(root / "probe.json", probe)
            save_json(root / "result.json", {"candidate_key": "k"})
            reused = _reuse(root, "k")
            self.assertEqual(reused["status"], "performance_ready")
            self.assertEqual(reused["legacy_quality"], probe["quality"])
            self.assertTrue((root / "legacy-local-quality/probe.json").is_file())

    def test_completed_qwen_stack_is_reused_without_gpu_or_receipt_changes(self):
        snap = snapshot("configs/stack.json")
        root = Path(snap["config"]["output"])
        paths = [p for p in root.rglob("*.json") if p.name in
                 {"config.json", "state.json", "summary.json", "probe.json", "result.json", "stack-freeze.json", "lmms-model.json"}]
        before = {p: p.read_bytes() for p in paths}
        with patch("search.stack.subprocess.run", side_effect=AssertionError("no Worker")), \
                patch("search.stack.hardware", side_effect=AssertionError("no GPU")):
            summary = _completed_handoff(root.resolve(), snap)
        self.assertEqual(summary["stop_reason"], "lmms_quality_passed")
        self.assertEqual({p: p.read_bytes() for p in paths}, before)

    def test_pipeline_reopens_legacy_exhaustion_but_honors_full_lmms_exhaustion(self):
        from search.pipeline import Pipeline
        pipeline = Pipeline()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pipeline.outputs["stack"] = root
            for policy, expected in ((None, "pending"), ("full_lmms", "quality_unmet")):
                save_json(root / "state.json", {"status": "completed", "quality_policy": policy,
                                               "stop_reason": "recovery_ladder_exhausted"})
                with patch("search.optimize.launch", side_effect=AssertionError("no Worker")):
                    self.assertEqual(pipeline.inspect("stack")["status"], expected)

    def test_config_compatibility_ignores_only_removed_quality_fields(self):
        current = {"runtime": {"vllm_max_model_len": 32768}, "benchmark": {"search_limit": 128}}
        old = {**current, "quality": {"minimum_dev_retention": .95},
               "benchmark": {"search_limit": 128, "dev_limit": 256}}
        self.assertEqual(evidence_config(old), current)
        self.assertNotEqual(evidence_config({**old, "runtime": {"vllm_max_model_len": 8192}}), current)


if __name__ == "__main__":
    unittest.main()
