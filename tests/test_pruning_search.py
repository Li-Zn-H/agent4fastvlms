"""Nonuniform allocation, finite refinement, reuse and SP invocation budgets."""

from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from search import pruning_search as sp
from skills.structured_pruning import (
    allocation_path,
    make_recipe,
    next_evaluation,
    validate_search,
)


POLICY = {
    "initial_depth_step": 0.05,
    "maximum_depth_step": 0.05,
    "failure_step_multiplier": 0.5,
    "success_step_multiplier": 2.0,
    "confirmed_failures_to_stop": 2,
}
DIMENSIONS = {"num_hidden_layers": 2, "hidden_size": 4, "intermediate_size": 100}
SELECTION = {"channel_group_size": 2, "protected_layers": []}
SCORES = [list(range(100)), [1.0] * 100]


def verdict(passed=True, confirmed=False):
    stages = {"search": {"action": "pass" if passed else "dev"}}
    if confirmed or not passed:
        stages["dev"] = {"action": "pass" if passed else "reject"}
    return {"action": "pass" if passed else "reject", "stages": stages}


def uniform_depth_path(count=50):
    return [{"actual_depth": i / 100} for i in range(count + 1)]


class PolicyTests(unittest.TestCase):
    def test_first_dev_failure_recovers_one_candidate_level_at_a_time(self):
        policy = {
            "initial_depth_step": 0.05,
            "failure_policy": "stop_on_first_dev_failure",
        }
        path = uniform_depth_path()
        quality = {
            5: verdict(confirmed=True),
            10: verdict(),
            15: verdict(False),
            20: verdict(confirmed=True),
        }
        before = json.dumps(quality, sort_keys=True)
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["reason"]), (10, "candidate_recovery"))
        self.assertTrue(job["force_dev"])
        self.assertEqual(
            (job["ladder_level"], job["parent_candidate"]),
            (2, 5),
        )
        self.assertEqual(job["failed_index"], 15)
        self.assertEqual(json.dumps(quality, sort_keys=True), before)
        quality[10] = verdict(False)
        terminal = next_evaluation(path, quality, policy)
        self.assertEqual(terminal["stop_reason"], "recovered_boundary")
        self.assertEqual(terminal["parent"], 5)
        self.assertEqual(terminal["failed_index"], 10)
        pending = {5: {"action": "pending", "stages": {"search": {"action": "pass"}}}}
        self.assertTrue(next_evaluation(path, pending, policy)["force_dev"])
        with self.assertRaises(ValueError):
            next_evaluation(path, {5: {"action": "reject", "stages": {}}}, policy)
        validate_search(
            {
                "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                "search": policy,
            }
        )

    def test_first_failure_policy_keeps_coarse_steps_and_extension_anchor(self):
        policy = {
            "initial_depth_step": 0.05,
            "failure_policy": "stop_on_first_dev_failure",
        }
        path, quality = uniform_depth_path(), {}
        for i in range(5, 51, 5):
            self.assertEqual(next_evaluation(path, quality, policy)["index"], i)
            quality[i] = verdict()
        self.assertEqual(
            next_evaluation(path, quality, policy)["stop_reason"],
            "declared_range_exhausted",
        )
        quality[25] = verdict(confirmed=True)
        quality[30] = verdict(False)
        job = next_evaluation(path, quality, policy, start_parent=25)
        self.assertEqual(job["stop_reason"], "recovered_boundary")
        self.assertEqual(job["parent"], 25)

    def test_failed_parent_confirmation_does_not_chain_to_older_parents(self):
        policy = {
            "initial_depth_step": 0.05,
            "failure_policy": "stop_on_first_dev_failure",
        }
        path = uniform_depth_path()
        quality = {
            5: verdict(),
            10: {
                "action": "reject",
                "stages": {
                    "search": {"action": "pass"},
                    "dev": {"action": "reject"},
                },
                "forced_dev": True,
            },
        }
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["reason"]), (5, "candidate_recovery"))
        self.assertTrue(job["force_dev"])

    def test_recovery_step_can_restore_two_candidate_levels(self):
        policy = {
            "initial_depth_step": 0.05,
            "failure_policy": "step_back_on_dev_failure",
            "failure_recovery_steps": 2,
        }
        path = uniform_depth_path()
        quality = {15: verdict(False)}
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["recovery_steps"]), (5, 2))

    def test_dev_failure_returns_to_previous_five_percent_candidate(self):
        policy = {
            "initial_depth_step": 0.05,
            "failure_policy": "step_back_on_dev_failure",
        }
        path = [{"actual_depth": i * 0.05 / 136} for i in range(2050)]
        quality = {
            1360: verdict(),
            1496: verdict(),
            1632: verdict(False),
        }
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["parent"]), (1496, 1632))
        self.assertTrue(job["force_dev"])
        self.assertEqual(job["reason"], "candidate_recovery")

    def test_bounded_spacing_confirms_parent_before_refinement(self):
        policy = {
            **POLICY,
            "minimum_depth_step": 0.01,
            "confirm_parent_on_failure": True,
        }
        path = [{"actual_depth": i / 10000} for i in range(1001)]
        quality = {500: verdict(), 1000: verdict(False)}
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["reason"]), (500, "confirm_parent"))
        quality[500] = verdict(confirmed=True)
        evaluated = []
        for _ in range(10):
            job = next_evaluation(path, quality, policy)
            if "stop_reason" in job:
                break
            self.assertNotIn(job["index"], evaluated)
            self.assertGreaterEqual(
                path[job["index"]]["actual_depth"] - 0.05 + 1e-12, 0.01
            )
            evaluated.append(job["index"])
            quality[job["index"]] = verdict(False)
        self.assertEqual(evaluated, [750, 625, 600])
        self.assertEqual(job["stop_reason"], "resolution_limit_reached")
        self.assertEqual(job["parent"], 500)
        # A resume must produce the identical terminal decision without more work.
        self.assertEqual(next_evaluation(path, quality, policy), job)

    def test_spacing_confirms_refined_parent_and_preserves_failed_history(self):
        policy = {
            **POLICY,
            "minimum_depth_step": 0.01,
            "confirm_parent_on_failure": True,
        }
        path = [{"actual_depth": i / 10000} for i in range(1001)]
        quality = {500: verdict(False), 250: verdict()}
        job = next_evaluation(path, quality, policy)
        self.assertEqual((job["index"], job["reason"]), (250, "confirm_parent"))
        quality[250] = verdict(False)
        before = json.dumps(quality, sort_keys=True)
        self.assertEqual(next_evaluation(path, quality, policy)["index"], 125)
        self.assertEqual(json.dumps(quality, sort_keys=True), before)

    def test_spacing_rounds_up_across_admission_holes_and_visits_endpoint(self):
        policy = {
            **POLICY,
            "minimum_depth_step": 0.01,
            "confirm_parent_on_failure": True,
        }
        path = [{"actual_depth": x} for x in (0, 0.02, 0.07, 0.075)]
        quality = {1: verdict(confirmed=True), 2: verdict(False)}
        self.assertEqual(
            next_evaluation(path, quality, policy)["stop_reason"],
            "resolution_limit_reached",
        )
        quality[2] = verdict(confirmed=True)
        self.assertEqual(next_evaluation(path, quality, policy)["index"], 3)
        with self.assertRaises(ValueError):
            validate_search(
                {
                    "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                    "search": {**policy, "minimum_depth_step": -0.01},
                }
            )

    def test_nonuniform_allocation_scale_invariance_and_holes(self):
        widths = [100, 98, 94, 90]
        path = allocation_path(SCORES, DIMENSIONS, SELECTION, widths)
        self.assertEqual(path[1]["widths"], [98, 100])
        self.assertEqual(path[2]["widths"], [94, 100])
        self.assertEqual(path[-1]["widths"], [90, 90])
        self.assertEqual(
            path,
            allocation_path(
                [[x * 8 for x in SCORES[0]], SCORES[1]], DIMENSIONS, SELECTION, widths
            ),
        )
        self.assertTrue(
            all(a["actual_depth"] < b["actual_depth"] for a, b in zip(path, path[1:]))
        )
        protected = allocation_path(
            SCORES, DIMENSIONS, {**SELECTION, "protected_layers": [0]}, widths
        )
        self.assertTrue(all(row["widths"][0] == 100 for row in protected))

    def test_extension_keeps_prefix_and_starts_at_confirmed_endpoint(self):
        old = allocation_path(SCORES, DIMENSIONS, SELECTION, [100, 98, 96])
        suffix = allocation_path(
            SCORES,
            DIMENSIONS,
            SELECTION,
            [100, 98, 96, 94, 92],
            starting_widths=old[-1]["widths"],
        )
        path = old + suffix[1:]
        self.assertEqual(path[: len(old)], old)
        self.assertEqual(suffix[0]["widths"], [96, 96])
        self.assertEqual(path[-1]["widths"], [92, 92])
        anchor = len(old) - 1
        snap = {
            "path": path,
            "config": {"search": POLICY},
            "extension": {"parent": anchor},
        }
        with self.assertRaises(ValueError):
            sp.decision_for(snap, {anchor: verdict()})
        job = sp.decision_for(snap, {anchor: verdict(confirmed=True)})
        self.assertEqual(job["parent"], anchor)
        self.assertGreater(job["index"], anchor)
        with self.assertRaises(ValueError):
            allocation_path(
                SCORES, DIMENSIONS, SELECTION, [100, 96], starting_widths=[98, 96]
            )

    def test_high_retention_uses_ten_milestones_not_fifty(self):
        path, quality, visited = uniform_depth_path(), {}, []
        while True:
            job = next_evaluation(path, quality, POLICY)
            if "stop_reason" in job:
                break
            self.assertNotIn(job["index"], visited)
            visited.append(job["index"])
            quality[job["index"]] = verdict()
        self.assertEqual(visited, list(range(5, 51, 5)))
        self.assertEqual(job["stop_reason"], "declared_range_exhausted")

    def test_failure_halves_then_passing_refines_known_interval(self):
        path = uniform_depth_path()
        quality = {5: verdict(False)}
        job = next_evaluation(path, quality, POLICY)
        self.assertEqual(job["index"], 2)
        self.assertEqual(job["step"], 0.025)
        quality[2] = verdict()
        job = next_evaluation(path, quality, POLICY)
        self.assertEqual(job["index"], 3)
        self.assertEqual(job["step"], 0.05)
        quality[3] = verdict()
        self.assertEqual(next_evaluation(path, quality, POLICY)["index"], 4)

    def test_boundary_requires_parent_dev_and_two_distinct_depths(self):
        path = uniform_depth_path()
        quality = {
            5: verdict(),
            10: verdict(False),
            7: verdict(False),
            6: verdict(False),
        }
        job = next_evaluation(path, quality, POLICY)
        self.assertEqual((job["index"], job["reason"]), (5, "confirm_parent"))
        quality[5] = verdict(confirmed=True)
        job = next_evaluation(path, quality, POLICY)
        self.assertEqual(job["stop_reason"], "confirmed_path_boundary")
        self.assertEqual(job["failed_indices"], [6, 7])

    def test_deeper_pass_resets_failure_and_forces_confirmation(self):
        path = uniform_depth_path()
        quality = {5: verdict(False), 2: verdict(False), 1: verdict(False)}
        self.assertEqual(
            next_evaluation(path, quality, POLICY)["failed_indices"], [1, 2]
        )
        quality[2] = verdict()
        # A minimum-step failure followed by an independently passing deeper node.
        small_policy = {**POLICY, "initial_depth_step": 0.01}
        job = next_evaluation(path, quality, small_policy)
        self.assertEqual((job["index"], job["force_dev"]), (2, True))
        quality[2] = verdict(confirmed=True)
        job = next_evaluation(path, quality, small_policy)
        self.assertEqual(job["parent"], 2)
        self.assertNotIn("stop_reason", job)

    def test_late_parent_failure_replays_without_erasing_descendants(self):
        quality = {5: verdict(False), 10: verdict(confirmed=True)}
        before = json.dumps(quality, sort_keys=True)
        job = next_evaluation(uniform_depth_path(), quality, POLICY)
        self.assertEqual((job["parent"], job["index"]), (0, 2))
        self.assertEqual(json.dumps(quality, sort_keys=True), before)

    def test_minimum_overshoot_and_incomplete_decision(self):
        path = [{"actual_depth": 0.0}, {"actual_depth": 0.1}]
        self.assertTrue(next_evaluation(path, {}, POLICY)["overshoot"])
        pending = next_evaluation(
            path,
            {1: {"action": "pending", "stages": {"search": {"action": "pass"}}}},
            POLICY,
        )
        self.assertTrue(pending["force_dev"])
        self.assertEqual(pending["reason"], "complete_confirmation")
        with self.assertRaises(ValueError):
            next_evaluation(path, {1: {"action": "reject", "stages": {}}}, POLICY)
        self.assertEqual(
            next_evaluation(path, {1: verdict(False)}, POLICY)["stop_reason"],
            "declared_range_exhausted",
        )
        with self.assertRaises(ValueError):
            validate_search(
                {
                    "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                    "search": {**POLICY, "failure_step_multiplier": 1.0},
                }
            )

    def test_quality_candidate_budget_is_validated(self):
        validate_search(
            {
                "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                "search": {
                    "initial_depth_step": 0.05,
                    "failure_policy": "stop_on_first_dev_failure",
                    "maximum_quality_candidates": 20,
                },
            }
        )
        with self.assertRaises(ValueError):
            validate_search(
                {
                    "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                    "search": {
                        "initial_depth_step": 0.05,
                        "failure_policy": "stop_on_first_dev_failure",
                        "maximum_quality_candidates": 0,
                    },
                }
            )

    def test_legacy_failure_policy_is_normalized(self):
        config = {
            "allocation": {"method": "layer_normalized_marginal_cost_v1"},
            "search": {
                "initial_depth_step": 0.05,
                "failure_policy": "stop_on_first_dev_failure",
            },
        }
        validate_search(config)
        self.assertEqual(
            config["search"]["failure_policy"], "step_back_on_dev_failure"
        )
        with self.assertRaises(ValueError):
            validate_search(
                {
                    "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                    "search": {
                        "initial_depth_step": 0.05,
                        "failure_policy": "step_back_on_dev_failure",
                        "failure_recovery_steps": 0,
                    },
                }
            )


class ExecutionTests(unittest.TestCase):
    def test_revision_preserves_path_and_inherits_cost_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {
                "allocation": {"method": "layer_normalized_marginal_cost_v1"},
                "search": POLICY,
            }
            old = {
                "config": config,
                "path": uniform_depth_path(),
                "extension": {"parent": 5, "previous_gpu_reserved_seconds": 12},
            }
            state = {
                "status": "incomplete",
                "stop_reason": "interrupted",
                "resource_accounting_complete": True,
                "gpu_reserved_seconds": 23,
            }
            sp.save_json(root / "config.json", old)
            sp.save_json(root / "state.json", state)
            source_bytes = (root / "config.json").read_bytes()
            settings = root / "settings.json"
            sp.save_json(
                settings,
                {
                    **config,
                    "search": {
                        **POLICY,
                        "minimum_depth_step": 0.01,
                        "confirm_parent_on_failure": True,
                    },
                },
            )
            revised = sp.revise_snapshot(root, settings)
            self.assertEqual(revised["path"], old["path"])
            self.assertEqual(revised["extension"], old["extension"])
            self.assertEqual(revised["revision"]["previous_gpu_reserved_seconds"], 35)
            self.assertEqual((root / "config.json").read_bytes(), source_bytes)
            state["status"] = "running"
            sp.save_json(root / "state.json", state)
            with self.assertRaises(ValueError):
                sp.revise_snapshot(root, settings)

    def test_discovery_requires_exact_channels_and_dev_overrides_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = allocation_path(SCORES, DIMENSIONS, SELECTION, [100, 98, 96])
            snap = {
                "model": {},
                "evaluation": {"generation": {}, "data": {}},
                "path": path,
                "dimensions": DIMENSIONS,
                "config": {"selection": SELECTION},
            }
            recipe = make_recipe(SCORES, path[1]["widths"], DIMENSIONS, SELECTION)
            sp.save_json(root / "recipe.json", recipe)
            request = {
                "operation": "structured_pruning_candidate",
                "model": {},
                "evaluation": snap["evaluation"],
                "recipe": str(root / "recipe.json"),
            }
            sp.save_json(root / "request.json", request)
            (root / "quality").mkdir()
            sp.save_json(root / "quality/search.jsonl", {"action": "pass"})
            with patch.object(
                sp, "stage_score", side_effect=lambda snap, stage, name: sp.read(name)
            ):
                _, quality = sp.discover(snap, root, SCORES)
                self.assertEqual(quality[1]["action"], "pass")
                request["force_dev"] = True
                sp.save_json(root / "request.json", request)
                _, quality = sp.discover(snap, root, SCORES)
                self.assertEqual(quality[1]["action"], "pending")
                self.assertTrue(quality[1]["forced_dev"])
                sp.save_json(root / "quality/dev.jsonl", {"action": "reject"})
                _, quality = sp.discover(snap, root, SCORES)
                self.assertEqual(quality[1]["action"], "reject")
                self.assertTrue(quality[1]["forced_dev"])
                recipe["retained_channels"][0][0] = 0
                sp.save_json(root / "recipe.json", recipe)
                self.assertEqual(sp.discover(snap, root, SCORES), ({}, {}))

    def test_discovery_ignores_failed_worker_receipts_with_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = allocation_path(SCORES, DIMENSIONS, SELECTION, [100, 98, 96])
            snap = {
                "model": {},
                "evaluation": {"generation": {}, "data": {}},
                "path": path,
                "dimensions": DIMENSIONS,
                "config": {"selection": SELECTION},
            }
            recipe = make_recipe(SCORES, path[1]["widths"], DIMENSIONS, SELECTION)
            sp.save_json(root / "recipe.json", recipe)
            request = {
                "operation": "structured_pruning_candidate",
                "model": {},
                "evaluation": snap["evaluation"],
                "recipe": str(root / "recipe.json"),
                "force_dev": True,
            }
            sp.save_json(root / "request.json", request)
            sp.save_json(root / "result.json", {"status": "failed"})
            (root / "quality").mkdir()
            sp.save_json(root / "quality/search.jsonl", {"action": "pass"})
            sp.save_json(root / "quality/dev.jsonl", {"action": "reject"})
            with patch.object(
                sp, "stage_score", side_effect=lambda snap, stage, name: sp.read(name)
            ):
                self.assertEqual(sp.discover(snap, root, SCORES), ({}, {}))

    def test_discovery_rejects_mismatched_ladder_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = allocation_path(SCORES, DIMENSIONS, SELECTION, [100, 98, 96])
            snap = {
                "model": {},
                "evaluation": {"generation": {}, "data": {}},
                "path": path,
                "dimensions": DIMENSIONS,
                "config": {
                    "selection": SELECTION,
                    "search": {
                        "initial_depth_step": 0.05,
                        "failure_policy": "step_back_on_dev_failure",
                    },
                },
            }
            recipe = make_recipe(SCORES, path[1]["widths"], DIMENSIONS, SELECTION)
            sp.save_json(root / "recipe.json", recipe)
            request = {
                "operation": "structured_pruning_candidate",
                "model": {},
                "evaluation": snap["evaluation"],
                "recipe": str(root / "recipe.json"),
                "search_run_id": "sp-wrong-protocol",
                "ladder_level": 99,
                "parent_candidate": 0,
            }
            sp.save_json(root / "request.json", request)
            (root / "quality").mkdir()
            sp.save_json(root / "quality/search.jsonl", {"action": "pass"})
            with patch.object(
                sp, "stage_score", side_effect=lambda snap, stage, name: sp.read(name)
            ):
                self.assertEqual(sp.discover(snap, root, SCORES), ({}, {}))

    def test_budget_resume_and_full_reuse_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            snap = {
                "dimensions": DIMENSIONS,
                "device": 0,
                "model": {},
                "evaluation": {},
                "config": {
                    "calibration": {"source": "unused"},
                    "selection": SELECTION,
                    "search": POLICY,
                },
                "path": allocation_path(
                    SCORES, DIMENSIONS, SELECTION, list(range(100, 89, -2))
                ),
            }
            quality = {}
            requests = []

            def launch(request, directory):
                requests.append(request)
                directory.mkdir(parents=True)
                sp.save_json(
                    directory / "result.json",
                    {"worker_seconds": 1.0, "gpu_reserved_seconds": 1.0},
                )
                quality[request["decision"]["index"]] = verdict()

            patches.enter_context(patch.object(sp, "RESULTS", root))
            patches.enter_context(
                patch.object(sp, "lock", side_effect=lambda path: nullcontext())
            )
            patches.enter_context(
                patch.object(sp, "load_calibration", return_value=SCORES)
            )
            patches.enter_context(
                patch.object(
                    sp,
                    "discover",
                    side_effect=lambda *args: ({i: {} for i in quality}, dict(quality)),
                )
            )
            patches.enter_context(patch.object(sp, "hardware", return_value="GPU-test"))
            patches.enter_context(patch.object(sp, "launch", side_effect=launch))
            first = sp.execute(snap, root, 1)
            self.assertEqual(first["stop_reason"], "invocation_limit")
            second = sp.execute(snap, root, 1)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(len(requests), 2)
            with (
                patch.object(sp, "hardware", side_effect=AssertionError("GPU queried")),
                patch.object(
                    sp, "launch", side_effect=AssertionError("Worker started")
                ),
            ):
                again = sp.execute(snap, root, 1)
            self.assertEqual(again["evaluated_candidates"], 2)
            self.assertEqual(again["gpu_reserved_seconds"], 2.0)

    def test_quality_budget_persists_across_resume(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            snap = {
                "dimensions": DIMENSIONS,
                "device": 0,
                "model": {},
                "evaluation": {},
                "config": {
                    "calibration": {"source": "unused"},
                    "selection": SELECTION,
                    "search": {
                        "initial_depth_step": 0.05,
                        "failure_policy": "stop_on_first_dev_failure",
                        "maximum_quality_candidates": 1,
                    },
                },
                "path": allocation_path(
                    SCORES, DIMENSIONS, SELECTION, list(range(100, 89, -2))
                ),
            }
            quality = {}
            requests = []

            def launch(request, directory):
                requests.append(request)
                directory.mkdir(parents=True)
                sp.save_json(
                    directory / "result.json",
                    {"worker_seconds": 1.0, "gpu_reserved_seconds": 1.0},
                )
                quality[request["decision"]["index"]] = verdict()

            patches.enter_context(patch.object(sp, "RESULTS", root))
            patches.enter_context(
                patch.object(sp, "lock", side_effect=lambda path: nullcontext())
            )
            patches.enter_context(
                patch.object(sp, "load_calibration", return_value=SCORES)
            )
            patches.enter_context(
                patch.object(
                    sp,
                    "discover",
                    side_effect=lambda *args: (
                        {i: {} for i in quality},
                        dict(quality),
                    ),
                )
            )
            patches.enter_context(patch.object(sp, "hardware", return_value="GPU-test"))
            patches.enter_context(patch.object(sp, "launch", side_effect=launch))
            first = sp.execute(snap, root)
            self.assertEqual(first["stop_reason"], "evaluation_budget_exhausted")
            self.assertEqual(first["scheduled_candidate_count"], 1)
            self.assertEqual(len(requests), 1)
            self.assertTrue(requests[0]["search_run_id"].startswith("sp-"))
            self.assertEqual(
                (requests[0]["ladder_level"], requests[0]["parent_candidate"]),
                (1, 0),
            )
            second = sp.execute(snap, root)
            self.assertEqual(second["stop_reason"], "evaluation_budget_exhausted")
            self.assertEqual(second["scheduled_candidate_count"], 1)
            self.assertEqual(len(requests), 1)

    def test_imports_remain_lightweight(self):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; import search.pruning_search; assert not {'torch','transformers'} & sys.modules.keys()",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
