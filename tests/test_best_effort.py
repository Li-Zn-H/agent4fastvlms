"""Below-target results remain useful without becoming qualified results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search.lmms_gate import (DEFAULT_TASKS, complete_lmms, quality_speed_frontier,
                              select_measured, selection_policy, stack_handoff_eligible)
from search.serving import acceptance_outcome, load_freeze
from search.stack import execute as execute_stack
from search.serving_lmms import execute as execute_serving
from search.serving_lmms import plan as serving_plan
from search.stack_acceleration import _parent
from search.worker import save_json
from skills.quantization import digest
from skills.stack import make_candidate
import test_serving_lmms


def receipt(retention, model_config="model.json"):
    return {"status": "completed", "task_count": 5, "all_task_hashes_match": True,
            "token_mechanism_verified": True, "aggregation": "mean_per_task_retention",
            "retention": retention, "minimum_retention": .95, "model_config": model_config,
            "quality_gate": "passed" if retention >= .95 else "quality_rejected",
            "tasks": [dict(spec, samples=1, task_hash_match=True, token_mechanism="trace.jsonl",
                           retention=retention) for spec in DEFAULT_TASKS]}


class BestEffortTests(unittest.TestCase):
    def test_complete_means_five_finite_tasks_with_mechanisms(self):
        valid = receipt(.94)
        self.assertTrue(complete_lmms(valid))
        for change in ({"status": "blocked"}, {"tasks": valid["tasks"][:-1]},
                       {"retention": float("nan")}, {"quality_gate": "passed"},
                       {"token_mechanism_verified": False}, {"retention": .93}):
            self.assertFalse(complete_lmms(valid | change))
        self.assertEqual(selection_policy({}), "strict")
        with self.assertRaises(ValueError):
            selection_policy({"selection_policy": "ignore_errors"})

    def test_frontier_prefers_fastest_target_met_else_highest_quality(self):
        rows = [{"candidate_id": "quality", "retention": .99, "mean_seconds": 2},
                {"candidate_id": "fast", "retention": .96, "mean_seconds": 1},
                {"candidate_id": "lossy", "retention": .94, "mean_seconds": .5},
                {"candidate_id": "dominated", "retention": .93, "mean_seconds": 3}]
        self.assertEqual(select_measured(rows)["candidate_id"], "fast")
        self.assertEqual(select_measured(rows[2:])["candidate_id"], "lossy")
        self.assertEqual({r["candidate_id"] for r in quality_speed_frontier(rows)}, {"quality", "fast", "lossy"})
        self.assertIsNone(select_measured([dict(rows[0], mean_seconds=float("nan"))]))

    def stack_run(self, root, outcomes, *, policy="best_effort", budget=2):
        candidates = [make_candidate(model_digest="m", sp=None, quantization=None,
                       token={"candidate_id": f"token-{i}", "config_digest": str(i)}, token_keep_ratio=ratio)
                      for i, ratio in enumerate((.5, .6))]
        snap = {"schema": "a4fvlm.stack.v1", "config_path": "job.json", "status": "ready",
                "model_digest": "m", "model": {}, "evaluation": {}, "token_config": {"selection": {}},
                "blockers": [], "sources": {}, "candidates": candidates, "initial_candidates": candidates[:1],
                "config": {"device": 0, "selection_policy": policy,
                           "limits": {"initial_candidates": 1, "total_candidates": budget},
                           "lmms": {"enabled": True, "minimum_retention": .95}}}
        self.stack_snapshot = snap

        def worker(request, directory):
            candidate = request["candidate"]
            artifact = directory / "artifact"
            artifact.mkdir(parents=True)
            probe = {"schema": "a4fvlm.stack-result.v1", "candidate_id": candidate["candidate_id"],
                     "candidate_key": candidate["candidate_key"], "status": "ready_for_lmms",
                     "artifact": str(artifact), "fresh_evidence": True, "skills": candidate["skills"],
                     "mechanism": {"status": "passed"}, "reload": {"status": "passed"},
                     "benchmark": {"status": "passed", "mean_seconds": 1, "peak_allocated_bytes": 1}}
            save_json(directory / "probe.json", probe)
            return {"probe_status": "ready_for_lmms", "artifact": str(artifact)}

        with patch("search.stack.snapshot", return_value=snap), \
                patch("search.stack.hardware", return_value={}), \
                patch("search.stack.launch", side_effect=worker), \
                patch("search.stack._recovery_neighbors", side_effect=lambda c, s: candidates[1:] if c == candidates[0] else []), \
                patch("search.stack._run_lmms_gate", side_effect=outcomes):
            return execute_stack("unused", output=root), candidates

    def test_stack_exhaustion_selects_best_not_last_and_keeps_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, candidates = self.stack_run(root, [receipt(.94), receipt(.91)])
            self.assertEqual(result["selected_candidate"], candidates[0]["candidate_id"])
            self.assertEqual(result["stop_reason"], "best_available_selected")
            self.assertEqual(result["search_stop_reason"], "recovery_ladder_exhausted")
            self.assertFalse(result["quality_target_met"])
            freeze = json.loads((root / "final/stack-freeze.json").read_text())
            self.assertEqual(freeze["lmms"]["quality_gate"], "quality_rejected")
            self.assertTrue(stack_handoff_eligible(freeze))
            self.assertFalse(stack_handoff_eligible(freeze | {"selection_policy": "strict"}))
            self.assertFalse(stack_handoff_eligible(freeze | {"search_stop_reason": "execution_error"}))
            with patch("search.stack.snapshot", return_value=self.stack_snapshot), \
                    patch("skills.stack_vllm.validate_stack_artifact"), \
                    patch("search.stack._run_lmms_gate", return_value=freeze["lmms"]), \
                    patch("search.stack.hardware", side_effect=AssertionError("GPU query during reuse")), \
                    patch("search.stack.launch", side_effect=AssertionError("duplicate Worker")):
                self.assertEqual(execute_stack("unused", output=root), result)

    def test_budget_exhaustion_can_fallback_but_pass_stops_early(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.stack_run(Path(folder), [receipt(.94)], budget=1)
            self.assertEqual(result["search_stop_reason"], "candidate_budget_exhausted")
            self.assertEqual(result["stop_reason"], "best_available_selected")
        with tempfile.TemporaryDirectory() as folder:
            result, _ = self.stack_run(Path(folder), [receipt(.96)])
            self.assertEqual(result["candidate_count"], 1)
            self.assertTrue(result["quality_target_met"])

    def test_strict_partial_and_execution_error_never_fallback(self):
        cases = [("strict", [receipt(.94), receipt(.93)]),
                 ("best_effort", [receipt(.94), {"status": "blocked", "reason": "API failure"}]),
                 ("best_effort", [dict(receipt(.94), tasks=[]), dict(receipt(.93), tasks=[])])]
        for policy, outcomes in cases:
            with self.subTest(policy=policy, outcomes=outcomes), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result, _ = self.stack_run(root, outcomes, policy=policy)
                self.assertIsNone(result["selected_candidate"])
                self.assertFalse((root / "final/stack-freeze.json").exists())
                if policy == "best_effort" and not outcomes[0]["tasks"]:
                    self.assertEqual(result["candidate_count"], 1)
                    self.assertEqual(result["stop_reason"], "blocked_preflight")

    def serving_run(self, root, outcomes, policy="best_effort"):
        path, config, _ = test_serving_lmms.ServingLmmsTests().fixture(root)
        config["selection_policy"] = policy
        save_json(path, config)

        def aggregate(output, *args, **kw):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return receipt(outcome, kw["model_config"])

        with patch("search.serving_lmms._run_tasks", side_effect=lambda c, m, o, **kw: o.mkdir(parents=True, exist_ok=True)), \
                patch("search.serving_lmms._task_complete", return_value=True), \
                patch("search.serving_lmms.aggregate_lmms", side_effect=aggregate):
            return execute_serving(path)

    def test_final_exhaustion_freezes_evaluated_candidate_for_real_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = self.serving_run(root, [.94, .92])
            self.assertEqual(result["stop_reason"], "best_available_selected")
            self.assertFalse(result["quality_target_met"])
            path = root / "gate/final/serving-freeze.json"
            freeze = json.loads(path.read_text())
            self.assertEqual(freeze["status"], "lmms_evaluated")
            self.assertEqual(freeze["lmms"]["retention"], .94)
            with patch("skills.stack_vllm.validate_stack_artifact"), \
                    patch("search.lmms_gate.aggregate_lmms", return_value=freeze["lmms"]):
                self.assertEqual(load_freeze(path)[0], freeze)
                changed = copy.deepcopy(freeze["lmms"])
                changed["tasks"].pop()
                with patch("search.lmms_gate.aggregate_lmms", return_value=changed):
                    with self.assertRaises(ValueError):
                        load_freeze(path)

    def test_final_error_stops_without_promotion_and_strict_keeps_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                self.serving_run(root, [.94, RuntimeError("execution failed")])
            self.assertFalse((root / "gate/final/serving-freeze.json").exists())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = self.serving_run(root, [.94, .93], policy="strict")
            self.assertEqual(result["stop_reason"], "serving_quality_unmet")
            self.assertFalse((root / "gate/final/serving-freeze.json").exists())

    def test_concurrent_only_timing_is_rejected_before_final_gpu_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, config, kv_path = test_serving_lmms.ServingLmmsTests().fixture(root)
            config["selection_policy"] = "best_effort"
            save_json(path, config)
            kv = json.loads(kv_path.read_text())
            kv["benchmarks"] = {"batch8": kv["benchmarks"]["search"]}
            save_json(kv_path, kv)
            with patch("search.serving_lmms._run_tasks", side_effect=AssertionError("GPU work")):
                with self.assertRaisesRegex(ValueError, "batch-1"):
                    execute_serving(path)

    def test_final_quality_and_matched_speed_are_separate(self):
        self.assertEqual(acceptance_outcome(.95, 1.01)["outcome"], "goal_achieved")
        result = acceptance_outcome(.944661, 1.5)
        self.assertEqual(result["outcome"], "accelerated_quality_unmet")
        self.assertFalse(result["goal_achieved"])
        for quality in (.94, .97):
            self.assertEqual(acceptance_outcome(quality, 1)["outcome"], "no_acceleration")
        with self.assertRaises(ValueError):
            acceptance_outcome(float("nan"), 2)

    def test_below_target_parent_crosses_runtime_and_serving_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, kv_path = test_serving_lmms.ServingLmmsTests().fixture(root)
            stack_path = root / "stack/final/stack-freeze.json"
            stack = json.loads(stack_path.read_text())
            Path(stack["artifact"]).mkdir()
            stack.update(status="frozen_best_available", selection_policy="best_effort",
                         quality_target_met=False, search_stop_reason="recovery_ladder_exhausted",
                         lmms=receipt(.94, stack["lmms"]["model_config"]))
            save_json(stack_path, stack)
            runtime_path = root / "runtime/final/runtime-freeze.json"
            runtime = json.loads(runtime_path.read_text()) | {"parent_digest": digest(stack)}
            save_json(runtime_path, runtime)
            fallback_path = root / "runtime/candidates/baseline-auto-o2/probe.json"
            save_json(fallback_path, json.loads(fallback_path.read_text()) | {"parent_digest": digest(stack)})
            save_json(kv_path, json.loads(kv_path.read_text()) | {"parent_digest": digest(runtime)})
            chain_path = Path(config["serving_state"])
            save_json(chain_path, json.loads(chain_path.read_text()) | {"stack_digest": digest(stack)})
            with patch("search.stack_acceleration.validate_stack_artifact"):
                self.assertEqual(_parent({"parent_freeze": str(stack_path)}, kind="runtime")["freeze"], stack)
            with self.assertRaises(ValueError):
                serving_plan(config)
            self.assertEqual(len(serving_plan(config | {"selection_policy": "best_effort"})["candidates"]), 2)
            save_json(stack_path, stack | {"lmms": dict(stack["lmms"], tasks=[])})
            with patch("search.stack_acceleration.validate_stack_artifact"):
                with self.assertRaises(ValueError):
                    _parent({"parent_freeze": str(stack_path)}, kind="runtime")


if __name__ == "__main__":
    unittest.main()
