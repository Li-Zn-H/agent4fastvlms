"""Stage dispatch, failure semantics and reuse without launching GPU work."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from search.baseline import complete_outputs, run_baseline
from search.pipeline import Pipeline, STAGES, execute, plan, result, run_command
from search.stack import discover_tokens
from search.worker import save_json


class SimulatedPipeline:
    """Fake stage receipts; exercise the real coordinator and checkpoint files."""
    def __init__(self, directory):
        self.root = Path(directory) / "pipeline"
        self.receipts = Path(directory) / "receipts"
        self.receipts.mkdir()
        self.job = {"model_id": "test-pipeline"}
        self.revision = 1

    def snapshot(self):
        return {"revision": self.revision}

    def inspect(self, stage):
        path = self.receipts / (stage + ".json")
        return json.loads(path.read_text()) if path.exists() else result("pending", "missing")

    def command(self, stage):
        return ["simulated-worker", stage]

    def finish(self, stage, status="complete", reason="passed"):
        save_json(self.receipts / (stage + ".json"), result(status, reason, evidence=str(self.receipts / stage)))


class PipelineTests(unittest.TestCase):
    def test_execute_initializes_summary_only_read_only_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            pipeline.root.mkdir(parents=True)
            save_json(pipeline.root / "summary.json", {"schema": "a4fvlm.pipeline-plan.v1"})
            state = execute(pipeline, runner=lambda job, stage, *_: job.finish(stage))
            self.assertEqual(state["status"], "completed")
            self.assertTrue((pipeline.root / "plan-summary.json").exists())
            self.assertTrue((pipeline.root / "config.json").exists())

    def test_dispatch_order_and_reuse_then_resume_without_any_new_child(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            pipeline.finish("dense")
            pipeline.finish("dense_lmms")
            calls = []

            def run(job, stage, directory, started):
                calls.append(stage)
                job.finish(stage)

            state = execute(pipeline, runner=run)
            self.assertEqual(calls, list(STAGES[2:]))
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["stages"]["dense"]["action"], "reused")
            with patch("search.pipeline.run_stage", side_effect=AssertionError("must not run")):
                again = execute(pipeline, resume=True, runner=lambda *args: self.fail("unexpected child"))
            self.assertEqual(again["stage_invocations"], len(calls))
            self.assertTrue(all(row["action"] == "reused" for row in again["stages"].values()))

    def test_error_stops_and_resume_rechecks_receipts_even_after_crash_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            calls = []

            def fail(job, stage, directory, started):
                calls.append(stage)
                job.finish(stage)
                if stage == "token":
                    raise RuntimeError("coordinator crashed after Worker saved evidence")

            with self.assertRaisesRegex(RuntimeError, "crashed"):
                execute(pipeline, runner=fail)
            state = json.loads((pipeline.root / "state.json").read_text())
            self.assertEqual(state["stop_reason"], "execution_error")
            self.assertEqual(state["next_stage"], "token")
            self.assertIsNone(state["active_stage"])
            self.assertNotIn("sp", calls)
            resumed = []
            execute(pipeline, resume=True, runner=lambda job, stage, *_: (resumed.append(stage), job.finish(stage)))
            self.assertEqual(resumed, list(STAGES[3:]))

    def test_quality_exhaustion_and_preflight_blocks_do_not_launch_downstream(self):
        for status, reason in [("quality_unmet", "recovery_ladder_exhausted"), ("blocked", "blocked_preflight")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                pipeline = SimulatedPipeline(folder)
                calls = []

                def run(job, stage, *_):
                    calls.append(stage)
                    job.finish(stage, status if stage == "stack" else "complete", reason)

                state = execute(pipeline, runner=run)
                self.assertEqual(state["status"], status)
                self.assertEqual(state["stop_reason"], reason)
                self.assertNotIn("serving", calls)
                self.assertNotIn("final_freeze", state)

    def test_zero_exit_without_valid_receipt_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            state = execute(pipeline, runner=lambda *args: None)
            self.assertEqual(state["status"], "blocked")
            self.assertEqual(state["stop_reason"], "stage_incomplete")
            self.assertEqual(state["stage_invocations"], 1)

    def test_explicit_resume_can_retry_backend_block_once(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            for stage in STAGES[:5]:
                pipeline.finish(stage)
            save_json(pipeline.receipts / "quantization.json",
                      result("blocked", "blocked_preflight", retryable=True))
            state = execute(pipeline, runner=lambda *args: self.fail("blocked job must stop"))
            self.assertEqual(state["status"], "blocked")
            calls = []
            state = execute(pipeline, resume=True,
                            runner=lambda job, stage, *_: (calls.append(stage), job.finish(stage)))
            self.assertEqual(calls, ["quantization", "stack", "serving"])
            self.assertEqual(state["status"], "completed")

    def test_snapshot_drift_is_rejected_before_resume_and_after_running_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            execute(pipeline, runner=lambda job, stage, *_: job.finish(stage))
            pipeline.revision += 1
            with self.assertRaisesRegex(ValueError, "changed"):
                execute(pipeline, resume=True, runner=lambda *args: self.fail("unexpected child"))
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)

            def change(job, stage, *_):
                job.revision += 1
                job.finish(stage)

            with self.assertRaisesRegex(ValueError, "changed while"):
                execute(pipeline, runner=change)

    def test_model_lock_prevents_a_second_coordinator(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)

            def run(job, stage, *_):
                if stage == "dense":
                    with self.assertRaisesRegex(RuntimeError, "Another coordinator"):
                        execute(job, resume=True)
                job.finish(stage)

            execute(pipeline, runner=run)

    def test_current_completed_job_reuses_all_evidence_without_subprocess_or_gpu_query(self):
        pipeline = Pipeline()
        with tempfile.TemporaryDirectory() as folder:
            pipeline.root = Path(folder) / "handoff"
            with patch("subprocess.Popen", side_effect=AssertionError("no child process")), \
                    patch("search.optimize.hardware", side_effect=AssertionError("no GPU query")), \
                    patch("search.optimize.launch", side_effect=AssertionError("no Worker")):
                planned = plan(pipeline)
                self.assertEqual(planned["status"], "completed")
                self.assertFalse(pipeline.root.exists())
                state = execute(pipeline)
            self.assertEqual(state["stage_invocations"], 0)
            self.assertAlmostEqual(state["stages"]["serving"]["retention"], .9540109865149781)

    def test_stale_configuration_and_changed_parent_benchmark_block_before_gpu_work(self):
        pipeline = Pipeline()
        pipeline.configs["runtime"]["runtime"]["vllm_max_model_len"] = 65536
        self.assertEqual(pipeline.inspect("serving")["status"], "blocked")
        pipeline = Pipeline()
        pipeline.model["path"] = "models/another-qwen3-vl"
        self.assertEqual(pipeline.inspect("token")["status"], "blocked")

    def test_backend_failure_child_exit_and_interrupt_are_not_quality_results(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = SimulatedPipeline(folder)
            with self.assertRaises(KeyboardInterrupt):
                execute(pipeline, runner=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
            state = json.loads((pipeline.root / "state.json").read_text())
            self.assertEqual(state["stop_reason"], "interrupted")
            self.assertNotIn("final_freeze", state)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(subprocess.CalledProcessError):
                run_command([str(Path('.venv-qwen3vl/bin/python').absolute()), "-c", "raise SystemExit(3)"],
                            Path(folder) / "log")

    def test_dense_resume_repairs_search_from_dev_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"stable_row_id": "a", "primary_category": "task", "logical_question": "Q?",
                     "answers": ["yes"], "scoring_type": "deterministic_rule_or_structure", "media": {"logical_path": "image.jpg"}}]
            output = {"sample_id": "a", "output_text": "yes"}
            (root / "dev.jsonl").write_text(json.dumps(output) + "\n")
            config = {"data": {"dev": "dev.jsonl", "search": "search.jsonl", "media_root": "."},
                      "scoring": {"maximum_response_tokens": 64, "negation_window_tokens": 3}}
            with patch("search.baseline.load_samples", return_value=rows), \
                    patch("search.baseline.load_model", side_effect=AssertionError("no model load")), redirect_stdout(io.StringIO()):
                run_baseline({}, config, root, resume=True)
            self.assertIsNotNone(complete_outputs(root / "search.jsonl", rows))
            (root / "search.jsonl").write_text((json.dumps(output) + "\n") * 2)
            self.assertIsNone(complete_outputs(root / "search.jsonl", rows))

    def test_stack_reads_unified_token_receipts_and_rejects_wrong_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"selection": {"score_dtype": "float32"}, "search": {}}
            snap = {"vision": config, "model": {"path": "correct"}, "evaluation": {}}
            save_json(root / "config.json", snap)
            save_json(root / "state.json", {"status": "completed", "candidates": [{"keep_ratio": .8}]})
            with patch("search.optimize.inspect_quality", return_value=({}, {"action": "pass"})), \
                    patch("search.optimize.inspect_benchmark", return_value={"source": "real-benchmark.json"}):
                rows, blockers = discover_tokens(root, config, [.8], model=snap["model"], evaluation={})
                self.assertEqual([row["candidate_id"] for row in rows], ["token-80"])
                self.assertEqual(rows[0]["source"], str(root / "state.json"))
                rows, _ = discover_tokens(root, config, [.8], model={"path": "wrong"})
                self.assertEqual(rows, [])

    def test_token_handoff_uses_referenced_receipts_when_runner_has_no_local_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"selection": {}, "search": {}}
            save_json(root / "config.json", {"vision": config, "model": {}, "evaluation": {}})
            source = root / "elsewhere/keep-80"
            save_json(root / "state.json", {"candidates": [{
                "keep_ratio": .8, "quality_sources": {"search": str(source / "quality/search.jsonl")},
                "benchmark": {"source": str(source / "benchmark/summary.json")},
            }, {"keep_ratio": .75, "benchmark": None}]})
            with patch("search.optimize.inspect_quality", return_value=({}, {"action": "pass"})) as quality, \
                    patch("search.optimize.inspect_benchmark", return_value={"source": "benchmark.json"}):
                rows, _ = discover_tokens(root, config, [.8])
            self.assertIn(source / "request.json", quality.call_args.args[2])
            self.assertEqual(rows[0]["ratio"], .8)

    def test_qualified_runtime_kv_dispatches_only_final_gate(self):
        pipeline = Pipeline()
        command = pipeline.command("serving")
        self.assertIn("search.serving_lmms", command)
        self.assertNotIn("search.stack_serving", command)


if __name__ == "__main__":
    unittest.main()
