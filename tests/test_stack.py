"""CPU contracts for the bounded Stack stage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from search.stack import (
    discover_structured_pruning,
    execute,
    _recovery_neighbors,
    _retain_attempt_diagnostics,
    _retain_evaluation_outputs,
    _run_lmms_gate,
    _score_token_benchmark_ready,
    plan,
    snapshot,
)
from search.worker import run_stack_candidate
from skills.stack import (
    compare_candidates,
    make_candidate,
    recovery_candidates,
    select_initial_candidates,
    validate_config,
    validate_result,
)
from skills.stack_model import STACK_ARCHITECTURE, create_pruned_model_class


class StackContractTests(unittest.TestCase):
    def test_sp_handoff_uses_latest_numbered_attempt_and_dev_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates/step-1233"
            candidate.mkdir(parents=True)
            (candidate / "recipe.json").write_text(json.dumps({
                "widths": [128], "actual_depth": .4,
            }))
            older = candidate / "attempt-9"
            newer = candidate / "attempt-11"
            older.mkdir()
            newer.mkdir()
            (older / "result.json").write_text(json.dumps({
                "status": "completed", "quality": {"action": "pass", "stages": {
                    "search": {"action": "pass", "retention": .98}}},
            }))
            for action, retention in (("pass", .97), ("reject", .90)):
                with self.subTest(latest_dev_action=action):
                    (newer / "result.json").write_text(json.dumps({
                        "status": "completed", "quality": {"action": action, "stages": {
                            "dev": {"action": action, "retention": retention}}},
                    }))
                    rows, _ = discover_structured_pruning(root)
                    if action == "pass":
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(Path(rows[0]["result_path"]), newer / "result.json")
                        self.assertEqual(rows[0]["quality"]["stages"]["dev"]["retention"], .97)
                    else:
                        self.assertEqual(rows, [])

    def test_export_resume_authenticates_parent_and_request(self):
        import copy
        from skills.quantization import digest, make_manifest
        from skills.stack_model import LLAVA_STACK_ARCHITECTURE, validate_resumable_export

        request = {"model": {"adapter": "llava_onevision", "path": "dense"},
                   "candidate": {"structured_pruning": {"recipe_digest": "recipe"}},
                   "calibration": {"data": "heldout.jsonl", "limit": 128},
                   "evaluation": {"root": "model-specific"}, "selection": {"keep": .45},
                   "quantization_runtime": {"quantize": True}}
        quant = {"route": "gptqmodel_w4a16", "bits": 4, "group_size": 128, "mapper_precision": "bf16",
                 "calibration": request["calibration"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = {"schema": "a4fvlm.stack-parent.v1", "architecture": LLAVA_STACK_ARCHITECTURE,
                      "widths": [128] * 28, "physical_compaction": True, "recipe_digest": "recipe"}
            config = {"model_type": "llava_onevision", "architectures": [LLAVA_STACK_ARCHITECTURE],
                      "a4fvlm_stack_widths": parent["widths"], "a4fvlm_stack_physical_compaction": True,
                      "text_config": {"hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28},
                      "quantization_config": {"bits": 4, "group_size": 128}}
            manifest = make_manifest(model=request["model"], config=quant, calibration=request["calibration"],
                module_names=["model.language_model.layers.0.mlp.gate_proj"], status="exported")
            manifest["stack"] = {"parent_recipe_digest": "recipe", "parent_manifest_digest": digest(parent),
                                 "fresh_calibration": True}
            visual = {"sample_ids": ["held-out"]}
            manifest["visual_quantization"] = {"artifact_config_digest": digest(config),
                                               "calibration_receipt_digest": digest(visual)}
            manifest["artifact_digest"] = digest({k: v for k, v in manifest.items() if k != "artifact_digest"})
            for name, value in (("config.json", config), ("stack-parent.json", parent), ("manifest.json", manifest),
                                ("vision-quantization.json", visual)):
                (root / name).write_text(json.dumps(value))
            result = validate_resumable_export(root, request, request)
            self.assertFalse(result["quality_reused"])
            for key in ("model", "evaluation", "candidate", "calibration", "selection", "quantization_runtime"):
                changed = copy.deepcopy(request)
                changed[key]["changed"] = True
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "request mismatch"):
                    validate_resumable_export(root, request, changed)
            (root / "vision-quantization.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "visual export"):
                validate_resumable_export(root, request, request)
            (root / "vision-quantization.json").write_text(json.dumps(visual))
            parent["recipe_digest"] = "unrelated"
            (root / "stack-parent.json").write_text(json.dumps(parent))
            with self.assertRaisesRegex(ValueError, "physical parent/calibration"):
                validate_resumable_export(root, request, request)

    def test_export_retry_skips_quantization_and_local_quality_preserves_export_on_error(self):
        from types import SimpleNamespace
        from search.worker import run_stack_sp_quantized_candidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "retained"
            artifact.mkdir()
            (artifact / "manifest.json").write_text('{}')
            recipe = root / "recipe.json"
            recipe.write_text('{}')
            calibration = root / "calibration.jsonl"
            calibration.write_text('{}\n')
            original = root / "export-request.json"
            original.write_text('{}')
            candidate = make_candidate(model_digest="m", sp={"recipe_path": str(recipe), "recipe_digest": "r"},
                quantization={"route": "gptqmodel_w4a16"}, token={"config_digest": "t"}, token_keep_ratio=.45)
            request = {"candidate": candidate, "worker_python": "python", "model": {
                           "adapter": "llava_onevision", "path": "models/LLaVA-OneVision-7B"},
                       "calibration": {"data": str(calibration)}, "quantization_runtime": {"quantize": True},
                       "probe_output": str(root / "work"), "resume_artifact": str(artifact),
                       "export_request": str(original), "prior_probe": {"mechanism": {"status": "passed"}},
                       "selection": {"score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}},
                       "evaluation": {}}
            with patch("skills.stack_model.validate_resumable_export", return_value={"quality_reused": False}), \
                    patch("skills.stack_model.load_stack_model", return_value=(SimpleNamespace(), object())), \
                    patch("search.model.load_model") as dense, \
                    patch("skills.stack_adapters.export_pruned_gptqmodel") as export, \
                    patch("search.run.evaluate_candidate", side_effect=AssertionError("local quality must not run")) as quality, \
                    patch("search.benchmark.benchmark", side_effect=RuntimeError("benchmark retry reached")) as timing:
                result = run_stack_sp_quantized_candidate(request)
            dense.assert_not_called()
            export.assert_not_called()
            quality.assert_not_called()
            timing.assert_called_once()
            self.assertEqual(result["artifact"], str(artifact))
            self.assertEqual(result["probe_status"], "candidate_error")
            probe = json.loads((root / "work/probe.json").read_text())
            self.assertIn("benchmark retry reached", probe["traceback"])

    def test_native_smoke_failure_stops_before_full_lmms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            snap = {"model": {}, "token_config": {"selection": {}}, "config": {
                "device": 0, "lmms": {"enabled": True, "smoke_samples_per_task": 2}}}
            with patch("search.stack.subprocess.run") as run, \
                    patch("search.lmms._task_complete", return_value=False), \
                    patch("search.stack.aggregate_lmms") as aggregate:
                result = _run_lmms_gate(candidate={"candidate_id": "llava", "token_keep_ratio": .45},
                    candidate_dir=root, probe={"artifact": str(artifact)}, snap=snap)
            self.assertEqual(result["status"], "blocked")
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--limit") + 1], "2")
            self.assertEqual(command[command.index("--output") + 1], str(root / "lmms-smoke"))
            aggregate.assert_not_called()

    def test_lmms_gate_resumes_partial_directory_without_losing_candidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_dir = Path(directory)
            artifact = candidate_dir / "artifact"
            artifact.mkdir()
            lmms = candidate_dir / "lmms"
            lmms.mkdir()
            marker = lmms / "completed-task.txt"
            marker.write_text("preserve")
            candidate = {"candidate_id": "stack-a", "token_keep_ratio": .7}
            snap = {"model": {}, "token_config": {"selection": {}}, "config": {
                "device": 0, "lmms": {"enabled": True},
            }}
            with patch("search.stack.subprocess.run") as run, \
                    patch("search.stack.aggregate_lmms", return_value={"quality_gate": "passed"}) as aggregate:
                _run_lmms_gate(candidate=candidate, candidate_dir=candidate_dir,
                               probe={"artifact": str(artifact)}, snap=snap)
            self.assertIn("--resume", run.call_args.args[0])
            self.assertEqual(aggregate.call_args.kwargs["candidate_id"], "stack-a")
            self.assertEqual(marker.read_text(), "preserve")

    def test_stack_reuses_only_score_based_vllm_token_receipts(self):
        old = {
            "vllm_benchmark": {
                "vision_token": {
                    "status": "enabled",
                    "method": "vllm_image_processor_budget",
                }
            }
        }
        current = {
            "vllm_benchmark": {
                "vision_token": {
                    "status": "enabled",
                    "method": "vllm_score_based_post_encoder",
                }
            }
        }
        self.assertFalse(_score_token_benchmark_ready(old))
        self.assertTrue(_score_token_benchmark_ready(current))

    def test_config_and_current_receipts_plan_without_gpu_query(self):
        result = plan("configs/stack.json")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["source_counts"]["quantization"], 1)
        self.assertEqual(len(result["initial_candidates"]), 1)
        self.assertFalse(result["metrics_transferable"])

    def test_explicit_initial_sources_select_exact_pair_or_block(self):
        config = json.loads(Path("configs/stack.json").read_text())
        config["initial_sources"] = {
            "structured_pruning": "sp-1224", "vision_token": "token-55",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            result = snapshot(path)
            self.assertEqual(result["status"], "ready")
            candidate = result["initial_candidates"][0]
            self.assertEqual(candidate["structured_pruning"]["candidate_id"], "sp-1224")
            self.assertEqual(candidate["vision_token"]["candidate_id"], "token-55")
            self.assertEqual(candidate["token_keep_ratio"], .55)
            config["initial_sources"]["structured_pruning"] = "missing"
            path.write_text(json.dumps(config))
            result = snapshot(path)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("initial_sources_not_available", result["blockers"])

    def test_quality_outputs_survive_worker_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder) / "candidate"
            with tempfile.TemporaryDirectory() as work:
                attempt = Path(work)
                (attempt / "quality").mkdir()
                (attempt / "quality/dev.jsonl").write_text('{"sample_id":"one"}\n')
                (attempt / "quality/summary.json").write_text('{}')
                probe = {"quality": {"status": "passed"}}
                _retain_evaluation_outputs(attempt, candidate, probe)
            retained = Path(probe["quality"]["outputs"]["dev"])
            self.assertEqual(json.loads(retained.read_text())["sample_id"], "one")

    def test_stack_retry_keeps_each_attempt_log_timing_and_parent_metadata(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, attempt = root / "candidate", root / ".work/candidate"
            for status in ("stack_runtime_blocked", "ready_for_lmms"):
                attempt.mkdir(parents=True)
                files = {
                    "request.json": '{"operation":"stack_candidate"}',
                    "probe.json": json.dumps({"status": status}),
                    "worker.log": status,
                    "benchmark/metrics.jsonl": status + " timing",
                    "evidence/runtime.log": status + " startup",
                    "artifact/manifest.json": '{"artifact_digest":"quantized-parent"}',
                    "artifact/stack-parent.json": '{"recipe_digest":"physical"}',
                    "artifact/model.safetensors": "weights are retained separately",
                }
                for name, value in files.items():
                    path = attempt / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value)
                _retain_attempt_diagnostics(attempt, candidate)
                shutil.rmtree(attempt)
            for number, status in enumerate(("stack_runtime_blocked", "ready_for_lmms"), 1):
                saved = candidate / "evidence" / f"attempt-{number}"
                self.assertEqual((saved / "worker.log").read_text(), status)
                self.assertEqual((saved / "benchmark/metrics.jsonl").read_text(), status + " timing")
                self.assertEqual((saved / "evidence/runtime.log").read_text(), status + " startup")
                self.assertEqual(json.loads((saved / "probe.json").read_text())["status"], status)
                self.assertEqual(json.loads((saved / "artifact-metadata/stack-parent.json").read_text())["recipe_digest"], "physical")
                self.assertFalse(list(saved.rglob("*.safetensors")))

    def test_candidate_identity_changes_for_each_physical_axis(self):
        common = dict(model_digest="m", sp={"recipe_digest": "sp"}, quantization={"manifest_digest": "q"}, token={"config_digest": "t"}, source_ids=())
        first = make_candidate(**common, token_keep_ratio=0.50)
        second = make_candidate(**common, token_keep_ratio=0.55)
        self.assertNotEqual(first["candidate_key"], second["candidate_key"])
        self.assertFalse(first["metrics_transferable"])
        completed_vision = make_candidate(**(common | {"quantization": {
            "manifest_digest": "q", "export_identity_digest": "visual-gptq"}}), token_keep_ratio=0.50)
        self.assertNotEqual(first["candidate_key"], completed_vision["candidate_key"],
                            "A legacy declared-precision hash cannot identify a new visual export")

    def test_initial_selection_is_bounded_and_prefers_wider_then_deeper_order(self):
        rows = []
        for depth in (0.1, 0.3, 0.2):
            rows.append(
                make_candidate(
                    model_digest="m",
                    sp={"recipe_digest": str(depth), "actual_depth": depth},
                    quantization={"manifest_digest": "q"},
                    token={"config_digest": "t"},
                    token_keep_ratio=0.5,
                )
            )
        selected = select_initial_candidates(rows, 1)
        self.assertEqual(selected[0]["structured_pruning"]["actual_depth"], 0.3)

    def test_recovery_changes_one_axis(self):
        original = make_candidate(
            model_digest="m",
            sp={"recipe_digest": "narrow", "actual_depth": 0.4},
            quantization={"manifest_digest": "q"},
            token={"config_digest": "t"},
            token_keep_ratio=0.5,
        )
        neighbors = recovery_candidates(
            original,
            wider_sp={"recipe_digest": "wide", "actual_depth": 0.3},
            higher_token_keep_ratio=0.55,
        )
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0]["token_keep_ratio"], 0.5)
        self.assertEqual(neighbors[1]["token_keep_ratio"], 0.55)
        self.assertNotEqual(neighbors[0]["candidate_key"], original["candidate_key"])

    def test_recovery_changes_both_axes_in_one_candidate(self):
        original = make_candidate(
            model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": 0.5},
            quantization={"candidate_id": "q", "manifest_digest": "q"},
            token={"candidate_id": "token-50", "config_digest": "t50", "ratio": 0.5},
            token_keep_ratio=0.5,
        )
        rows = recovery_candidates(
            original,
            paired_sp={"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": 0.4},
            paired_token_keep_ratio=0.55,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["structured_pruning"]["candidate_id"], "sp-wide")
        self.assertEqual(rows[0]["token_keep_ratio"], 0.55)

    def test_recovery_neighbors_are_paired_and_adjacent(self):
        original = make_candidate(
            model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": 0.5},
            quantization={"candidate_id": "q", "manifest_digest": "q", "route": "gptqmodel_w4a16"},
            token={"candidate_id": "token-50", "config_digest": "t50", "ratio": 0.5},
            token_keep_ratio=0.5,
        )
        snap = {
            "config": {"recovery_axes": ["paired_structured_token_recovery"]},
            "sources": {
                "structured_pruning": [
                    {"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": 0.5},
                    {"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": 0.4},
                ],
                "vision_token": [
                    {"candidate_id": "token-50", "config_digest": "t50", "ratio": 0.5},
                    {"candidate_id": "token-55", "config_digest": "t55", "ratio": 0.55},
                ],
            },
        }
        rows = _recovery_neighbors(original, snap)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["structured_pruning"]["candidate_id"], "sp-wide")
        self.assertEqual(rows[0]["token_keep_ratio"], 0.55)
        self.assertEqual(rows[0]["vision_token"]["candidate_id"], "token-55")
        # Candidate identity canonicalises source IDs so receipts remain stable
        # regardless of transformation order.
        self.assertEqual(rows[0]["source_ids"], ["q", "sp-wide", "token-55"])

    def test_token_exhaustion_policy_requires_explicit_paired_lmms_config(self):
        config = json.loads(Path("configs/stack-internvl35.json").read_text())
        config["token_recovery_exhaustion"] = "reuse_last_qualified"
        validate_config(config)
        for value in (True, None, [], "reuse_failed"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config({**config, "token_recovery_exhaustion": value})
        with self.assertRaises(ValueError):
            validate_config({**config, "recovery_axes": ["wider_structured_pruning"]})
        with self.assertRaises(ValueError):
            validate_config({**config, "selection_policy": "strict", "lmms": {"enabled": False}})

    def test_token_exhaustion_reuses_only_the_last_qualified_source(self):
        import copy

        token = {"candidate_id": "token-95", "config_digest": "t", "ratio": .95,
                 "status": "completed", "quality": {"action": "pass"}, "fresh_evidence": True}
        original = make_candidate(model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": .5},
            quantization={"candidate_id": "q", "manifest_digest": "q"},
            token=token, token_keep_ratio=.95)
        snap = {"config": {"recovery_axes": ["paired_structured_token_recovery"]},
                "sources": {"structured_pruning": [original["structured_pruning"],
                    {"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": .4},
                    {"candidate_id": "sp-widest", "recipe_digest": "widest", "actual_depth": .3}],
                    "vision_token": [token]}}
        self.assertEqual(_recovery_neighbors(original, snap), [])
        snap["config"]["token_recovery_exhaustion"] = "reuse_last_qualified"
        first, = _recovery_neighbors(original, snap)
        self.assertEqual(first["token_keep_ratio"], .95)
        self.assertEqual(first["vision_token"], token)
        self.assertEqual(first["quantization"], original["quantization"])
        self.assertEqual(first["source_ids"], ["q", "sp-wide", "token-95"])
        self.assertNotEqual(first["candidate_key"], original["candidate_key"])
        self.assertEqual(first["recovery"]["strategy"], "wider_sp_reuse_last_token")
        second, = _recovery_neighbors(first, snap)
        self.assertEqual(second["structured_pruning"]["candidate_id"], "sp-widest")
        self.assertEqual(_recovery_neighbors(second, snap), [])
        for change in ({"status": "running"}, {"quality": {"action": "reject"}},
                       {"fresh_evidence": False}, {"config_digest": "changed"}, {"ratio": .9}):
            changed = copy.deepcopy(snap)
            changed["sources"]["vision_token"] = [{**token, **change}]
            with self.subTest(change=change):
                self.assertEqual(_recovery_neighbors(original, changed), [])
        # An available higher ratio still takes the normal paired step.
        snap["sources"]["vision_token"].append({**token, "candidate_id": "token-98", "ratio": .98})
        paired, = _recovery_neighbors(original, snap)
        self.assertEqual(paired["token_keep_ratio"], .98)
        self.assertEqual(paired["recovery"]["strategy"], "paired_structured_token_recovery")

    def test_pareto_frontier_does_not_use_scalar_speedup(self):
        rows = [
            {"candidate_id": "a", "status": "qualified", "lmms": {"quality_gate": "passed", "retention": 0.96}, "benchmark": {"mean_seconds": 1.0, "peak_allocated_bytes": 10}},
            {"candidate_id": "b", "status": "qualified", "lmms": {"quality_gate": "passed", "retention": 0.96}, "benchmark": {"mean_seconds": 2.0, "peak_allocated_bytes": 20}},
            {"candidate_id": "c", "status": "qualified", "lmms": {"quality_gate": "passed", "retention": 0.99}, "benchmark": {"mean_seconds": 1.2, "peak_allocated_bytes": 12}},
        ]
        self.assertEqual([row["candidate_id"] for row in compare_candidates(rows)], ["a", "c"])

    def test_sp_quantization_worker_is_fail_closed(self):
        candidate = make_candidate(
            model_digest="m",
            sp={"recipe_digest": "sp"},
            quantization={"manifest_digest": "q", "artifact": "/missing"},
            token={"config_digest": "t"},
            token_keep_ratio=0.5,
        )
        with tempfile.TemporaryDirectory() as folder:
            result = run_stack_candidate(
                {
                    "candidate": candidate,
                    "probe_output": folder,
                    "model": {"path": "models/Qwen3-VL-4B-Instruct", "dtype": "bfloat16"},
                    "evaluation": {},
                }
            )
            self.assertEqual(result["probe_status"], "stack_bridge_blocked")
            report = json.loads((Path(folder) / "probe.json").read_text())
            self.assertEqual(report["mechanism"]["reason"], "sp_quantization_bridge_required")

    def test_stack_topology_class_is_lazy_and_named(self):
        self.assertEqual(create_pruned_model_class().__name__, STACK_ARCHITECTURE)

    def test_sp_quantization_requires_isolated_worker_and_calibration(self):
        candidate = make_candidate(
            model_digest="m",
            sp={"recipe_digest": "sp"},
            quantization={"manifest_digest": "q", "route": "gptqmodel_w4a16"},
            token={"config_digest": "t"},
            token_keep_ratio=0.5,
        )
        with tempfile.TemporaryDirectory() as folder:
            result = run_stack_candidate(
                {
                    "candidate": candidate,
                    "probe_output": folder,
                    "model": {"path": "models/Qwen3-VL-4B-Instruct", "dtype": "bfloat16"},
                    "evaluation": {},
                    "quantization_runtime": {"quantize": True},
                }
            )
            self.assertEqual(result["probe_status"], "stack_bridge_blocked")
            report = json.loads((Path(folder) / "probe.json").read_text())
            self.assertEqual(report["mechanism"]["reason"], "quantization_worker_python_missing")

        with tempfile.TemporaryDirectory() as folder:
            result = run_stack_candidate(
                {
                    "candidate": candidate,
                    "probe_output": folder,
                    "model": {"path": "models/Qwen3-VL-4B-Instruct", "dtype": "bfloat16"},
                    "evaluation": {},
                    "worker_python": ".venv-quant-gptqmodel/bin/python",
                    "quantization_runtime": {"quantize": True},
                }
            )
            self.assertEqual(result["probe_status"], "stack_bridge_blocked")
            report = json.loads((Path(folder) / "probe.json").read_text())
            self.assertEqual(report["mechanism"]["reason"], "stack_calibration_missing")

    def test_result_requires_fresh_mechanisms_without_local_quality(self):
        row = {
            "schema": "a4fvlm.stack-result.v1",
            "candidate_id": "stack-a",
            "fresh_evidence": True,
            "quality": {"action": "reject", "dev_retention": 0.1},
            "mechanism": {"status": "passed"},
            "reload": {"status": "passed"},
            "benchmark": {"status": "passed"},
        }
        validate_result(row)
        row["fresh_evidence"] = False
        with self.assertRaises(ValueError):
            validate_result(row)

    def test_execute_automatically_recovers_after_lmms_rejection(self):
        """One execute call must own the whole local → LMMS → recovery loop."""

        initial = make_candidate(
            model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": 0.5},
            quantization={"candidate_id": "q", "manifest_digest": "q"},
            token={"candidate_id": "token-50", "config_digest": "t50", "ratio": 0.5},
            token_keep_ratio=0.5,
        )
        snap = {
            "schema": "a4fvlm.stack.v1",
            "config_path": "/tmp/stack.json",
            "blockers": [],
            "config": {
                "device": 0,
                "limits": {"initial_candidates": 1, "total_candidates": 2},
                "quality": {"minimum_dev_retention": 0.95},
                "runtime": {"vllm_reload": False, "vllm_benchmark": False},
                "quantization": {"worker_python": "python"},
                "recovery_axes": ["paired_structured_token_recovery"],
                "lmms": {
                    "enabled": True,
                    "worker_python": "python",
                    "baseline_root": "dense",
                    "minimum_retention": 0.95,
                    "output_subdir": "lmms",
                },
            },
            "model": {"path": "model"},
            "model_digest": "m",
            "evaluation": {},
            "token_config": {"selection": {}},
            "sources": {
                "structured_pruning": [
                    initial["structured_pruning"],
                    {"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": 0.4},
                ],
                "vision_token": [
                    initial["vision_token"],
                    {"candidate_id": "token-55", "config_digest": "t55", "ratio": 0.55},
                ],
            },
            "initial_candidates": [initial],
            "candidates": [initial],
            "status": "ready",
        }

        def launch_candidate(request, attempt):
            candidate = request["candidate"]
            attempt = Path(attempt)
            artifact = attempt / "artifact"
            artifact.mkdir(parents=True)
            probe = {
                "schema": "a4fvlm.stack-result.v1",
                "candidate_id": candidate["candidate_id"],
                "candidate_key": candidate["candidate_key"],
                "skills": candidate["skills"],
                "fresh_evidence": True,
                "status": "qualified",
                "quality": {
                    "action": "pass",
                    "dev_retention": 0.97,
                    "stages": {"dev": {"action": "pass", "retention": 0.97}},
                },
                "mechanism": {"status": "passed", "fresh_shape_reparse": True, "fresh_calibration": True},
                "reload": {"status": "passed"},
                "benchmark": {
                    "status": "passed",
                    "mean_seconds": 1.0,
                    "peak_allocated_bytes": 1,
                },
                "artifact": str(artifact),
            }
            (attempt / "probe.json").write_text(json.dumps(probe))
            result = {
                "schema": "a4fvlm.stack-result.v1",
                "candidate_id": candidate["candidate_id"],
                "candidate_key": candidate["candidate_key"],
                "probe_status": "qualified",
                "artifact": str(artifact),
            }
            (attempt / "result.json").write_text(json.dumps(result))
            return result

        def lmms_gate(*, candidate, **_kwargs):
            return {
                "schema": "a4fvlm.lmms-summary.v1",
                "status": "completed",
                "candidate_id": candidate["candidate_id"],
                "quality_gate": "passed" if candidate["token_keep_ratio"] > 0.5 else "quality_rejected",
                "retention": 0.96 if candidate["token_keep_ratio"] > 0.5 else 0.80,
            }

        with tempfile.TemporaryDirectory() as folder, \
                patch("search.stack.snapshot", return_value=snap), \
                patch("search.stack.hardware", return_value={"name": "cpu"}), \
                patch("search.stack.launch", side_effect=launch_candidate) as worker, \
                patch("search.stack._run_lmms_gate", side_effect=KeyboardInterrupt) as gate:
            # Simulate interruption after local qualification, before LMMS completes.
            with self.assertRaises(KeyboardInterrupt):
                execute("ignored.json", output=Path(folder) / "stack")
            candidate_dir = Path(folder) / "stack/candidates" / initial["candidate_id"]
            self.assertTrue((candidate_dir / "probe.json").is_file())
            self.assertTrue((candidate_dir / "result.json").is_file())
            gate.side_effect = lmms_gate
            summary = execute("ignored.json", output=Path(folder) / "stack")
            self.assertEqual(worker.call_count, 2, "Resume must reuse the first candidate's local work")
            freeze = json.loads(
                (Path(folder) / "stack" / "final" / "stack-freeze.json").read_text()
            )
            self.assertEqual(freeze["status"], "frozen_for_lmms")
            self.assertEqual(freeze["token_keep_ratio"], 0.55)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["selected_candidate"], summary["results"][1]["candidate_id"], summary)
        self.assertEqual(summary["stop_reason"], "lmms_quality_passed")
        self.assertEqual(summary["results"][0]["lmms"]["quality_gate"], "quality_rejected")
        self.assertEqual(summary["results"][1]["lmms"]["quality_gate"], "passed")
        self.assertEqual(
            summary["results"][1]["recovery"]["strategy"],
            "paired_structured_token_recovery",
        )

        # An execution/API failure on the recovered candidate is not ladder
        # exhaustion just because an earlier candidate failed quality.
        for failure, expected_reason in (("stack_runtime_blocked", "blocked_preflight"),
                                         ("candidate_error", "execution_error")):
            def interrupted_recovery(request, attempt):
                result = launch_candidate(request, attempt)
                if request["candidate"]["token_keep_ratio"] > .5:
                    result["probe_status"] = failure
                    probe = json.loads((attempt / "probe.json").read_text())
                    probe["status"] = failure
                    (attempt / "probe.json").write_text(json.dumps(probe))
                    (attempt / "result.json").write_text(json.dumps(result))
                return result

            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder, \
                    patch("search.stack.snapshot", return_value=snap), \
                    patch("search.stack.hardware", return_value={"name": "cpu"}), \
                    patch("search.stack.launch", side_effect=interrupted_recovery), \
                    patch("search.stack._run_lmms_gate", side_effect=lmms_gate):
                failed = execute("ignored.json", output=Path(folder) / "stack")
            self.assertEqual(failed["stop_reason"], expected_reason)
            self.assertEqual(failed["candidate_count"], 2)

    def test_execute_does_not_quality_recover_after_lmms_block(self):
        initial = make_candidate(
            model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": 0.5},
            quantization={"candidate_id": "q", "manifest_digest": "q"},
            token={"candidate_id": "token-50", "config_digest": "t50", "ratio": 0.5,
                   "status": "completed", "quality": {"action": "pass"}},
            token_keep_ratio=0.5,
        )
        snap = {
            "schema": "a4fvlm.stack.v1", "config_path": "/tmp/stack.json", "blockers": [],
            "config": {
                "device": 0, "limits": {"initial_candidates": 1, "total_candidates": 2},
                "quality": {"minimum_dev_retention": 0.95},
                "runtime": {"vllm_reload": False, "vllm_benchmark": False},
                "quantization": {"worker_python": "python"},
                "recovery_axes": ["paired_structured_token_recovery"],
                "token_recovery_exhaustion": "reuse_last_qualified",
                "lmms": {"enabled": True, "worker_python": "python", "baseline_root": "dense", "output_subdir": "lmms"},
            },
            "model": {"path": "model"}, "model_digest": "m", "evaluation": {}, "token_config": {"selection": {}},
            "sources": {"structured_pruning": [initial["structured_pruning"],
                {"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": .4}],
                "vision_token": [initial["vision_token"]]},
            "initial_candidates": [initial], "candidates": [initial], "status": "ready",
        }
        self.assertEqual(len(_recovery_neighbors(initial, snap)), 1)

        def launch_candidate(request, attempt):
            attempt = Path(attempt)
            artifact = attempt / "artifact"
            artifact.mkdir(parents=True)
            probe = {
                "schema": "a4fvlm.stack-result.v1", "candidate_id": request["candidate"]["candidate_id"],
                "candidate_key": request["candidate"]["candidate_key"], "skills": request["candidate"]["skills"],
                "fresh_evidence": True, "status": "qualified",
                "quality": {"action": "pass", "dev_retention": 0.97, "stages": {"dev": {"action": "pass", "retention": 0.97}}},
                "mechanism": {"status": "passed", "fresh_shape_reparse": True, "fresh_calibration": True}, "reload": {"status": "passed"},
                "benchmark": {"status": "passed", "mean_seconds": 1.0, "peak_allocated_bytes": 1},
                "artifact": str(artifact),
            }
            (attempt / "probe.json").write_text(json.dumps(probe))
            result = {"schema": "a4fvlm.stack-result.v1", "candidate_id": request["candidate"]["candidate_id"], "candidate_key": request["candidate"]["candidate_key"], "probe_status": "qualified", "artifact": str(artifact)}
            (attempt / "result.json").write_text(json.dumps(result))
            return result

        with tempfile.TemporaryDirectory() as folder, \
                patch("search.stack.snapshot", return_value=snap), \
                patch("search.stack.hardware", return_value={"name": "cpu"}), \
                patch("search.stack.launch", side_effect=launch_candidate), \
                patch("search.stack._run_lmms_gate", return_value={"status": "blocked", "reason": "blocked_preflight"}):
            summary = execute("ignored.json", output=Path(folder) / "stack")

        self.assertEqual(summary["candidate_count"], 1)
        self.assertIsNone(summary["selected_candidate"])
        self.assertEqual(summary["stop_reason"], "blocked_preflight")

    def test_expanded_budget_resumes_neighbor_then_reports_ladder_exhaustion(self):
        self._assert_expanded_budget_recovery(reuse_token=False)

    def test_token_exhaustion_recovery_obeys_budget_and_reuses_completed_attempts(self):
        self._assert_expanded_budget_recovery(reuse_token=True)

    def _assert_expanded_budget_recovery(self, *, reuse_token):
        initial = make_candidate(
            model_digest="m",
            sp={"candidate_id": "sp-deep", "recipe_digest": "deep", "actual_depth": .5},
            quantization={"candidate_id": "q", "manifest_digest": "q"},
            token={"candidate_id": "token-50", "config_digest": "t", "ratio": .5,
                   "status": "completed", "quality": {"action": "pass"}},
            token_keep_ratio=.5,
        )
        snap = {
            "schema": "a4fvlm.stack.v1", "config_path": "/tmp/stack.json",
            "blockers": [], "status": "ready", "model_digest": "m",
            "model": {}, "evaluation": {}, "token_config": {"selection": {}},
            "config": {
                "device": 0, "limits": {"initial_candidates": 1, "total_candidates": 1},
                "recovery_axes": ["paired_structured_token_recovery"],
                "token_recovery_exhaustion": "reuse_last_qualified" if reuse_token else "stop",
                "lmms": {"enabled": True},
            },
            "sources": {
                "structured_pruning": [
                    initial["structured_pruning"],
                    {"candidate_id": "sp-wide", "recipe_digest": "wide", "actual_depth": .4},
                ],
                "vision_token": [
                    initial["vision_token"],
                    *([] if reuse_token else [
                        {"candidate_id": "token-55", "config_digest": "t", "ratio": .55}]),
                ],
            },
            "initial_candidates": [initial], "candidates": [initial],
        }

        def rejected_receipts(directory, candidate):
            directory.mkdir(parents=True, exist_ok=True)
            identity = {k: candidate[k] for k in ("candidate_id", "candidate_key")}
            result = {**identity, "status": "completed", "probe_status": "ready_for_lmms"}
            (directory / "probe.json").write_text(json.dumps({**identity,
                "schema": "a4fvlm.stack-result.v1", "status": "ready_for_lmms", "fresh_evidence": True,
                "mechanism": {"status": "passed"}, "reload": {"status": "passed"},
                "benchmark": {"status": "passed"}}))
            (directory / "result.json").write_text(json.dumps(result))
            return result

        with tempfile.TemporaryDirectory() as folder, \
                patch("search.stack.snapshot", return_value=snap), \
                patch("search.stack.hardware", return_value={}), \
                patch("search.stack._run_lmms_gate", return_value={"status": "completed", "quality_gate": "quality_rejected", "retention": .8}), \
                patch("search.stack.launch", side_effect=lambda request, attempt: rejected_receipts(
                    Path(attempt), request["candidate"],
                )) as worker:
            output = Path(folder)
            rejected_receipts(output / "candidates" / initial["candidate_id"], initial)
            limited = execute("ignored.json", output=output)
            self.assertEqual(limited["stop_reason"], "candidate_budget_exhausted")
            worker.assert_not_called()

            snap["config"]["limits"]["total_candidates"] = 2
            expanded = execute("ignored.json", output=output)
            worker.assert_called_once()
            recovered = worker.call_args.args[0]["candidate"]
            self.assertEqual(recovered["token_keep_ratio"], .5 if reuse_token else .55)
            self.assertEqual(recovered["structured_pruning"]["candidate_id"], "sp-wide")
            if reuse_token:
                self.assertEqual(recovered["vision_token"], initial["vision_token"])
                self.assertEqual(recovered["recovery"]["strategy"], "wider_sp_reuse_last_token")
            self.assertTrue(expanded["results"][0]["reused"])
            self.assertEqual(expanded["candidate_count"], 2)
            self.assertEqual(expanded["stop_reason"], "recovery_ladder_exhausted")
            self.assertIsNone(expanded["selected_candidate"])

            repeated = execute("ignored.json", output=output)
            worker.assert_called_once()
            self.assertTrue(all(row["reused"] for row in repeated["results"]))
            self.assertEqual(repeated["stop_reason"], "recovery_ladder_exhausted")


if __name__ == "__main__":
    unittest.main()
