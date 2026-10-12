"""Independent JSON-driven quantization candidate contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from search.quantization import _candidate_digest, execute, plan, snapshot
from search.worker import (
    run_quantization_candidate,
    run_quantization_framework_probe,
    save_json,
)
from skills.quantization import (
    MODULE_PRECISION_BF16,
    MODULE_PRECISION_INT4,
    ROUTE_GPTQMODEL_W4A16,
    ROUTE_LLM_COMPRESSOR_FP8,
    adapter_contract,
    classify_module,
    make_manifest,
    stage_template,
    validate_config,
    validate_manifest,
    validate_module_inventory,
    validate_stage_trace,
)


class PolicyTests(unittest.TestCase):
    def test_runtime_changes_invalidate_runtime_receipt(self):
        candidate = {
            "id": "gptq",
            "route": ROUTE_GPTQMODEL_W4A16,
            "bits": 4,
            "group_size": 128,
            "scheme": "W4A16",
            "runtime_config": {
                "quantize": True,
                "vllm_reload": True,
                "vllm_gpu_memory_utilization": 0.92,
                "vllm_max_model_len": 8192,
            },
        }
        changed = {
            **candidate,
            "runtime_config": {
                **candidate["runtime_config"],
                "vllm_gpu_memory_utilization": 0.8,
                "vllm_max_model_len": 32768,
            },
        }
        self.assertNotEqual(_candidate_digest(candidate), _candidate_digest(changed))
        disabled = {
            **candidate,
            "runtime_config": {**candidate["runtime_config"], "quantize": False},
        }
        self.assertNotEqual(_candidate_digest(candidate), _candidate_digest(disabled))

    def test_json_plan_is_candidate_based_and_recipe_free(self):
        result = plan("configs/quantization.json")
        self.assertTrue(result["independent_from_structured_pruning"])
        self.assertNotIn("recipe_input", result)
        self.assertGreaterEqual(len(result["candidates"]), 4)
        self.assertIn("gptqmodel_w4a16", {row["candidate"] for row in result["candidates"]})

    def test_dense_module_policy_preserves_visual_mapper(self):
        self.assertEqual(
            classify_module("model.language_model.layers.0.mlp.gate_proj"),
            MODULE_PRECISION_INT4,
        )
        self.assertEqual(
            classify_module("model.visual.blocks.0.mlp.linear_fc1"),
            MODULE_PRECISION_INT4,
        )
        self.assertEqual(
            classify_module("model.visual.merger.linear_fc2"),
            MODULE_PRECISION_BF16,
        )
        self.assertEqual(
            classify_module("model.visual.patch_embed.proj"),
            MODULE_PRECISION_BF16,
        )

    def test_inventory_requires_both_modalities(self):
        names = {
            "model.language_model.layers.0.self_attn.q_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.merger.linear_fc2",
        }
        report = validate_module_inventory(names)
        self.assertTrue(report["valid"])
        self.assertEqual(report["int4_count"], 2)

    def test_candidate_validation_and_manifest_digest(self):
        config = {
            "route": ROUTE_GPTQMODEL_W4A16,
            "bits": 4,
            "group_size": 128,
            "scheme": "W4A16",
            "mapper_precision": "bf16",
            "calibration": {"data": "calibration.jsonl", "limit": 2},
            "runtime_config": {"clean_reload": True, "vllm_reload": True},
        }
        validate_config(config)
        manifest = make_manifest(
            model={"path": "models/dense", "dtype": "bfloat16"},
            config=config,
            calibration=config["calibration"],
            module_names=[
                "model.language_model.layers.0.mlp.gate_proj",
                "model.visual.merger.linear_fc2",
            ],
        )
        validate_manifest(manifest)
        manifest["module_assignment_digest"] = "stale"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_fp8_candidate_does_not_require_calibration(self):
        validate_config(
            {
                "route": ROUTE_LLM_COMPRESSOR_FP8,
                "bits": 8,
                "group_size": 128,
                "scheme": "FP8_DYNAMIC",
                "calibration": {},
                "runtime_config": {},
            }
        )

    def test_routes_declare_isolated_requirements_and_lifecycle(self):
        result = plan("configs/quantization.json")
        self.assertTrue(
            all(not row["missing_requirements"] for row in result["candidates"])
        )
        contract = adapter_contract(ROUTE_GPTQMODEL_W4A16)
        self.assertEqual(
            contract["stages"],
            ["prepare", "calibrate", "export", "manifest", "reload"],
        )
        stages = stage_template()
        for stage in stages.values():
            stage["status"] = "passed"
        validate_stage_trace(stages)


class WorkerTests(unittest.TestCase):
    def test_raw_evidence_survives_failed_attempt_cleanup_and_resume(self):
        from search.quantization import _candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            candidate = _candidate({"calibration": {"data": "heldout.jsonl", "limit": 2}}, {
                "id": "gptq", "route": ROUTE_GPTQMODEL_W4A16, "bits": 4, "group_size": 128,
            })
            snap = {"config_path": "job.json", "model_path": "original-model",
                    "model": {"path": "original-model"}, "evaluation": {}, "device": 0,
                    "output": str(output), "candidates": [candidate]}

            def worker(request, directory):
                directory.mkdir(parents=True)
                save_json(directory / "request.json", request)
                (directory / "worker.log").write_text(directory.name)
                (directory / "quality").mkdir()
                (directory / "quality/dev.jsonl").write_text('{"sample_id":"a","answer":"yes"}\n')
                (directory / "benchmark").mkdir()
                save_json(directory / "benchmark/summary.json", {"elapsed_seconds": 2})
                if directory.name == "attempt-1":
                    raise RuntimeError("failed after saving raw outputs")
                save_json(directory / "probe.json", {"status": "framework_ready"})
                return {"status": "completed", "probe_status": "framework_ready"}

            with (patch("search.quantization.snapshot", return_value=snap),
                  patch("search.quantization._preflight", return_value={"ready_for_worker": True}),
                  patch("search.quantization.hardware", return_value="CPU"),
                  patch("search.quantization.launch", side_effect=worker)):
                execute("job.json", output=output)
                execute("job.json", output=output)
            self.assertFalse((output / ".work").exists())
            for attempt in ("attempt-1", "attempt-2"):
                evidence = output / "candidates/gptq/evidence" / attempt
                self.assertEqual((evidence / "worker.log").read_text(), attempt)
                self.assertEqual(json.loads((evidence / "request.json").read_text())["model"],
                                 {"path": "original-model", "device": "cuda:0"})
                self.assertEqual(json.loads((evidence / "quality/dev.jsonl").read_text())["answer"], "yes")
                self.assertEqual(json.loads((evidence / "benchmark/summary.json").read_text()),
                                 {"elapsed_seconds": 2})

    def test_new_export_contract_block_overrides_old_qualified_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            snap = snapshot("configs/quantization-llava15.json")
            candidate = snap["candidates"][0]
            directory = output / "candidates" / candidate["id"]
            directory.mkdir(parents=True)
            save_json(output / "config.json", snap)
            save_json(directory / "probe.json", {"status": "qualified", "candidate_digest": _candidate_digest(candidate)})
            save_json(directory / "result.json", {"status": "completed", "probe_status": "qualified",
                                                  "candidate_digest": _candidate_digest(candidate)})
            with patch("search.quantization._preflight", return_value={
                "ready_for_worker": False, "implementation_blocked": True
            }), patch("search.quantization.launch") as launch:
                result = execute("configs/quantization-llava15.json", output=output)
            launch.assert_not_called()
            self.assertEqual(result["results"][candidate["id"]]["status"], "blocked")
            self.assertIsNone(result["selected_candidate"])
            self.assertTrue(list((output / "history").glob("probe-*/probe.json")))

    def test_resume_authenticates_export_before_replacing_snapshot(self):
        from skills.quantization import digest

        for change in ("runtime", "calibration", "forged_digest", "manifest", "model", "evaluation"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                old = snapshot("configs/quantization-llava15.json")
                old["output"] = str(output)
                candidate = old["candidates"][0]
                candidate["runtime_config"].pop("vllm_gpu_memory_utilization", None)
                candidate["runtime_config"].pop("vllm_max_model_len", None)
                current = copy.deepcopy(old)
                current["candidates"][0]["runtime_config"]["vllm_gpu_memory_utilization"] = 0.8
                directory = output / "candidates" / candidate["id"]
                artifact = directory / "retry-artifact"
                artifact.mkdir(parents=True)
                manifest = make_manifest(
                    model=old["model"], config=candidate,
                    calibration=candidate["calibration"],
                    module_names=["model.language_model.layers.0.mlp.gate_proj"],
                    status="exported",
                )
                stages = {name: {"status": "passed"} for name in stage_template()}
                stages["prepare"]["assignment_digest"] = manifest["module_assignment_digest"]
                probe = {
                    "candidate": candidate["id"], "route": candidate["route"],
                    "status": "candidate_error", "candidate_digest": digest(candidate),
                    "quality": {"status": "passed", "dev_retention": 0.97},
                    "performance": {"status": "passed", "mean_round_seconds": 20},
                    "stages": stages,
                }
                if change == "calibration":
                    current["candidates"][0]["calibration"]["limit"] = 64
                elif change == "forged_digest":
                    probe["candidate_digest"] = "unrelated"
                elif change == "manifest":
                    manifest["model_digest"] = "unrelated"
                elif change in ("model", "evaluation"):
                    current[change]["changed"] = True
                save_json(output / "config.json", old)
                save_json(directory / "probe.json", probe)
                save_json(artifact / "manifest.json", manifest)

                def fake_launch(request, destination):
                    destination.mkdir(parents=True)
                    expected = "quantization_runtime_validation" if change == "runtime" else "quantization_candidate"
                    self.assertEqual(request["operation"], expected)
                    if change == "runtime":
                        self.assertEqual(request["artifact"], str(artifact))
                        self.assertEqual(request["prior_probe"]["candidate_digest"], digest(candidate))
                    save_json(destination / "probe.json", {"status": "candidate_error"})
                    return {"status": "completed", "probe_status": "candidate_error"}

                with (
                    patch("search.quantization.snapshot", return_value=current),
                    patch("search.quantization._preflight", return_value={"ready_for_worker": True}),
                    patch("search.quantization.hardware", return_value="CPU"),
                    patch("search.quantization.launch", side_effect=fake_launch) as launch,
                ):
                    if change in ("model", "evaluation"):
                        with self.assertRaisesRegex(ValueError, "Saved quantization context mismatch"):
                            execute("unused", output=output)
                        launch.assert_not_called()
                    else:
                        execute("unused", output=output)
                        launch.assert_called_once()
                        self.assertEqual(launch.call_args.args[0]["operation"],
                                         "quantization_runtime_validation" if change == "runtime" else "quantization_candidate")

    def test_missing_builder_dependency_is_blocked_not_worker_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            result = run_quantization_framework_probe(
                {
                    "probe_output": str(destination),
                    "candidate": {
                        "id": "fake",
                        "route": ROUTE_GPTQMODEL_W4A16,
                        "framework": "fake",
                        "imports": ["module_that_does_not_exist_a4fvlm"],
                        "runtime_imports": [],
                    },
                    "runtime": {},
                }
            )
            self.assertEqual(result["probe_status"], "blocked_missing_dependency")
            report = json.loads((destination / "probe.json").read_text())
            self.assertEqual(report["status"], "blocked_missing_dependency")

    def test_candidate_lifecycle_stops_before_export_when_dependency_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            result = run_quantization_candidate(
                {
                    "probe_output": str(destination),
                    "candidate": {
                        "id": "fake",
                        "route": ROUTE_GPTQMODEL_W4A16,
                        "framework": "fake",
                        "imports": ["module_that_does_not_exist_a4fvlm"],
                        "calibration_required": True,
                    },
                    "runtime": {"quantize": True},
                }
            )
            self.assertEqual(result["probe_status"], "blocked_missing_dependency")
            report = json.loads((destination / "probe.json").read_text())
            self.assertEqual(report["stages"]["prepare"]["status"], "blocked")
            self.assertEqual(report["stages"]["export"]["status"], "not_run")

    def test_execute_keeps_only_candidate_receipts_for_blocked_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "quantization"
            with patch(
                "search.quantization._preflight",
                return_value={"ready_for_worker": False},
            ):
                result = execute("configs/quantization.json", output=output)
            self.assertEqual(result["status"], "completed")
            self.assertFalse((output / ".work").exists())
            self.assertTrue(
                (output / "candidates/gptqmodel_w4a16/probe.json").exists()
            )
            self.assertTrue(
                (output / "candidates/gptqmodel_w4a16/result.json").exists()
            )
            self.assertEqual(
                {
                    path.name
                    for path in (output / "candidates/gptqmodel_w4a16").iterdir()
                },
                {"probe.json", "result.json", "evidence"},
            )
            self.assertFalse(list(output.glob("candidate-*/")))

    def test_successful_worker_attempt_is_compacted_and_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "quantization.json"
            config.write_text(
                json.dumps(
                    {
                        "model_config": "configs/model.json",
                        "device": 0,
                        "output": str(Path(temporary) / "out"),
                        "calibration": {
                            "data": "data/evaluation/search-128.jsonl",
                            "limit": 2,
                        },
                        "runtime": {},
                        "candidates": [
                            {
                                "id": "gptq",
                                "enabled": True,
                                "route": ROUTE_GPTQMODEL_W4A16,
                                "bits": 4,
                                "group_size": 128,
                                "scheme": "W4A16",
                                "mapper_precision": "bf16",
                            }
                        ],
                    }
                )
            )

            def fake_launch(request, directory):
                state = json.loads((Path(temporary) / "out/state.json").read_text())
                self.assertEqual(state["status"], "running")
                self.assertEqual(state["active_candidate"], "gptq")
                self.assertEqual(state["active_attempt"], str(directory))
                directory.mkdir(parents=True)
                save_json(
                    directory / "probe.json",
                    {
                        "schema": "a4fvlm.quantization-probe.v1",
                        "candidate": "gptq",
                        "status": "framework_ready",
                    },
                )
                return {
                    "status": "completed",
                    "probe_status": "framework_ready",
                    "report": str(directory / "probe.json"),
                }

            class Present:
                pass

            with (
                patch(
                    "search.quantization.importlib.util.find_spec",
                    return_value=Present(),
                ),
                patch("search.quantization.hardware", return_value="CPU"),
                patch("search.quantization.launch", side_effect=fake_launch),
            ):
                result = execute(config, output=Path(temporary) / "out")
            candidate = Path(temporary) / "out/candidates/gptq"
            self.assertEqual(result["results"]["gptq"]["status"], "framework_ready")
            self.assertTrue((candidate / "probe.json").exists())
            self.assertTrue((candidate / "result.json").exists())
            self.assertFalse((Path(temporary) / "out/.work").exists())
            self.assertEqual(list((Path(temporary) / "out").rglob("request.json")), [])
            self.assertEqual(list((Path(temporary) / "out").rglob("worker.log")), [])
            receipt = json.loads((candidate / "result.json").read_text())
            self.assertEqual(receipt["report"], str(candidate / "probe.json"))
            self.assertEqual(
                {path.name for path in candidate.iterdir()},
                {"probe.json", "result.json"},
            )

    def test_resume_retries_failed_receipt_and_removes_stale_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "quantization.json"
            output = Path(temporary) / "out"
            config.write_text(
                json.dumps(
                    {
                        "model_config": "configs/model.json",
                        "device": 0,
                        "output": str(output),
                        "calibration": {
                            "data": "data/evaluation/search-128.jsonl",
                            "limit": 2,
                        },
                        "runtime": {},
                        "candidates": [
                            {
                                "id": "gptq",
                                "enabled": True,
                                "route": ROUTE_GPTQMODEL_W4A16,
                                "bits": 4,
                                "group_size": 128,
                                "scheme": "W4A16",
                                "mapper_precision": "bf16",
                            }
                        ],
                    }
                )
            )
            candidate = output / "candidates/gptq"
            candidate.mkdir(parents=True)
            save_json(candidate / "probe.json", {"status": "worker_failed"})
            save_json(
                candidate / "result.json",
                {"status": "worker_failed", "candidate": "gptq"},
            )
            stale = output / ".work/candidate-gptq/attempt-1"
            stale.mkdir(parents=True)
            (stale / "request.json").write_text("stale")

            def fake_launch(request, directory):
                directory.mkdir(parents=True)
                save_json(
                    directory / "probe.json",
                    {
                        "schema": "a4fvlm.quantization-probe.v1",
                        "candidate": "gptq",
                        "status": "framework_ready",
                    },
                )
                return {
                    "status": "completed",
                    "probe_status": "framework_ready",
                    "report": str(directory / "probe.json"),
                }

            class Present:
                pass

            with (
                patch(
                    "search.quantization.importlib.util.find_spec",
                    return_value=Present(),
                ),
                patch("search.quantization.hardware", return_value="CPU"),
                patch("search.quantization.launch", side_effect=fake_launch) as launch,
            ):
                result = execute(config, output=output)

            launch.assert_called_once()
            self.assertEqual(result["results"]["gptq"]["status"], "framework_ready")
            self.assertFalse((output / ".work").exists())


if __name__ == "__main__":
    unittest.main()
