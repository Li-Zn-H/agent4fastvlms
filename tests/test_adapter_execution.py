"""Exercise real dispatch and result isolation using CPU-only model doubles."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from adapters import AdapterError, ModelAdapter, HardwareAdapter, models, hardware
from adapters.capabilities import check_capabilities, STAGE_OPERATIONS
from search.context import dense_root, search_root, validate_result_scope
from search.model import load_model, load_processor, prepare_inputs, generate_outputs
from search.registry import preflight


class TestModelAdapter(ModelAdapter):
    id = "test_vlm"
    model_types = ("test_vlm",)
    stages = ("dense",)
    hardware_ids = ("cuda", "test_cpu")
    entrypoints = {key: "test_adapter_execution:fake_entry" for key in STAGE_OPERATIONS["dense"]}

    def validate_checkpoint(self, checkpoint, *, pipeline=False):
        if checkpoint.get("model_type") != self.id:
            raise AdapterError("wrong test checkpoint")

    def describe(self):
        return {"id": self.id}


def fake_entry(*args, **kwargs):
    raise AssertionError("Test must supply an implementation")


class AdapterExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        checkpoint = self.root / "model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({"model_type": "test_vlm"}))
        self.model = {"adapter": "test_vlm", "hardware": "cuda", "path": str(checkpoint),
                      "device": "cuda:0", "dtype": "bfloat16", "attn_implementation": "sdpa"}
        self.adapter = TestModelAdapter()
        self.registration = patch.dict(models._items, {self.adapter.id: self.adapter})
        self.registration.start()
        self.addCleanup(self.registration.stop)

    def test_dense_dispatches_loader_processor_and_generation_to_registered_model(self):
        import torch
        class Inputs(dict):
            def to(self, device):
                return self

        processor = SimpleNamespace(batch_decode=Mock(return_value=["answer"]))
        inputs = Inputs(input_ids=torch.tensor([[2, 3]]))
        model = SimpleNamespace(eval=Mock(), device=torch.device("cpu"),
                                generate=Mock(return_value=torch.tensor([[2, 3, 4]])))
        loader = Mock(return_value=(model, processor))
        prepare = Mock(return_value=inputs)
        dispatch = {"dense_load": loader, "prepare_inputs": prepare}
        with patch.object(self.adapter, "resolve", side_effect=dispatch.__getitem__), \
                patch("adapters.qwen3vl_dense.load_model", side_effect=AssertionError("no Qwen fallback")):
            loaded, proc = load_model(self.model)
            self.assertIs(loaded, model)
            sample = {"stable_row_id": "row"}
            self.assertIs(prepare_inputs(proc, sample, self.root), inputs)
            outputs = generate_outputs(loaded, proc, [sample], media_root=self.root,
                                       generation={"max_new_tokens": 8, "do_sample": False, "ignore_eos": False})
            self.assertEqual(outputs, [{"sample_id": "row", "output_text": "answer"}])
            loader.assert_called_once_with(self.model)
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(processor.batch_decode.call_args.args[0].tolist(), [[4]])

    def test_processor_only_loading_dispatches_without_model_or_backend_import(self):
        processor = SimpleNamespace()
        load = Mock(return_value=processor)
        with patch.object(self.adapter, "resolve", return_value=load) as resolve:
            self.assertIs(load_processor(self.model), processor)
            resolve.assert_called_once_with("processor_load")
        self.assertEqual(processor._agent4vlm_adapter_id, self.adapter.id)

    def test_explicit_adapter_mismatch_and_unknown_model_fail_before_import(self):
        with patch("adapters.base.import_module", side_effect=AssertionError("no backend import")):
            with self.assertRaisesRegex(AdapterError, "does not match"):
                load_model(self.model | {"adapter": "qwen3_vl"})
            with self.assertRaisesRegex(AdapterError, "unknown"):
                load_model(self.model | {"adapter": "missing"})
            with self.assertRaisesRegex(AdapterError, "no implemented sp"):
                load_model(self.model, structured_pruning={"widths": [8]})

    def test_static_matrix_distinguishes_implementation_from_environment_evidence(self):
        with patch("adapters.base.import_module", side_effect=AssertionError("no imports")), \
                patch("subprocess.check_output", side_effect=AssertionError("no GPU")):
            matrix = preflight(self.model, device=0)
        self.assertEqual(matrix["status"], "blocked_preflight")
        self.assertEqual(matrix["stages"]["dense"], {"status": "ready_for_probe", "implementation": "available",
                                                    "environment": "unverified", "quality": "unverified"})
        self.assertEqual(matrix["stages"]["token"]["status"], "blocked_preflight")
        self.assertEqual(preflight(self.model, stages=["dense"], device=0)["status"], "ready_for_probe")
        for override in ({"hardware": "missing"}, {"device": "cpu:0"}):
            with self.assertRaises(AdapterError):
                preflight(self.model | override)

    def test_missing_binding_and_unported_dimensions_block_the_affected_stages(self):
        with patch.object(self.adapter, "entrypoints", {}):
            self.assertIn("missing entry", preflight(self.model)["stages"]["dense"]["reason"])
        qwen = models.get("qwen3_vl")
        checkpoint = {"model_type": "qwen3_vl", "text_config": dict(qwen.pipeline_dimensions)}
        checkpoint["text_config"]["num_hidden_layers"] = 48
        report = check_capabilities(qwen, checkpoint, hardware.get("cuda"))
        self.assertEqual(report["stages"]["dense"]["status"], "ready_for_probe")
        self.assertEqual(report["stages"]["stack"]["status"], "blocked_preflight")

    def test_model_scope_requires_an_independent_result_root(self):
        for evaluation in ({}, {"result_root": "results/model/qwen3vl/new-name"}):
            with self.assertRaisesRegex(AdapterError, "own evaluation.result_root"):
                validate_result_scope(self.model, evaluation)
        evaluation = {"result_root": str(self.root / "experiment"), "scoring": {"baseline": "dense"}}
        self.assertEqual(validate_result_scope(self.model, evaluation), self.root / "experiment")
        self.assertEqual(dense_root(evaluation), self.root / "experiment/search/dense")
        self.assertEqual(search_root(evaluation), self.root / "experiment/search")
        with self.assertRaises(AdapterError):
            dense_root(evaluation | {"scoring": {"baseline": "../foreign-model"}})

    def test_configured_denominator_is_used_by_real_scoring_and_no_legacy_fallback(self):
        from search.optimize import stage_score
        sample = {"stable_row_id": "one", "primary_category": "task", "logical_question": "Q",
                  "answers": ["yes"], "scoring_type": "deterministic_rule_or_structure",
                  "media": {"logical_path": "image.jpg"}}
        (self.root / "image.jpg").touch()
        samples = self.root / "data.jsonl"
        samples.write_text(json.dumps(sample) + "\n")
        evaluation = {"result_root": str(self.root / "experiment"),
                      "data": {"search": str(samples), "media_root": str(self.root)},
                      "scoring": {"baseline": "dense", "maximum_response_tokens": 64, "negation_window_tokens": 3},
                      "quality": {"minimum_retention": .95, "search_direct_pass_minimum_retention": .975}}
        baseline = dense_root(evaluation) / "quality/search.jsonl"
        baseline.parent.mkdir(parents=True)
        (baseline.parent / "config.json").write_text(json.dumps({"model": self.model, "evaluation": evaluation}))
        baseline.write_text(json.dumps({"sample_id": "one", "output_text": "yes"}) + "\n")
        candidate = self.root / "candidate.jsonl"
        candidate.write_text(baseline.read_text())
        snap = {"model": self.model, "evaluation": evaluation}
        self.assertEqual(stage_score(snap, "search", candidate)["retention"], 1)
        baseline.unlink()
        with self.assertRaises(FileNotFoundError):
            stage_score(snap, "search", candidate)

    def test_new_root_rejects_unbound_or_foreign_candidate_evidence(self):
        from search.optimize import compatible
        snap = {"model": self.model, "evaluation": {"result_root": str(self.root / "experiment")},
                "vision": {"selection": {}}}
        request = {"model": self.model, "evaluation": snap["evaluation"], "keep_ratio": .8, "selection": {}}
        self.assertTrue(compatible(request, snap, .8))
        for changes in ({"model": {}}, {"evaluation": {}}, {"keep_ratio": .9}):
            self.assertFalse(compatible(request | changes, snap, .8))
        self.assertFalse(compatible({"keep_ratio": .8}, snap, .8))

    def test_missing_dense_receipt_and_foreign_baseline_cannot_be_reused(self):
        from search.context import validate_dense_reference
        from search.baseline import run_baseline
        evaluation = {"result_root": str(self.root / "experiment"), "scoring": {"baseline": "dense"}}
        destination = dense_root(evaluation) / "quality"
        destination.mkdir(parents=True)
        (destination / "dev.jsonl").write_text('{"sample_id":"one","output_text":"yes"}\n')
        with patch("search.baseline.load_model", side_effect=AssertionError("no model load")):
            with self.assertRaisesRegex(ValueError, "no model/evaluation receipt"):
                run_baseline(self.model, evaluation, destination, resume=True)
            with self.assertRaisesRegex(AdapterError, "no model/evaluation receipt"):
                validate_dense_reference(self.model, evaluation)
            manifest = destination / "config.json"
            manifest.write_text(json.dumps({"model": self.model | {"path": "foreign"}, "evaluation": evaluation}))
            with self.assertRaisesRegex(AdapterError, "config mismatch"):
                validate_dense_reference(self.model, evaluation)
            with self.assertRaisesRegex(ValueError, "config changed"):
                run_baseline(self.model, evaluation, destination, resume=True)

    def test_pipeline_paths_and_dense_worker_environment_follow_jsons(self):
        from search.pipeline import Pipeline, plan
        from search.context import PROJECT_ROOT

        new_root = self.root / "results"
        cfg_dir = self.root / "configs"
        cfg_dir.mkdir()
        mapping = {}

        def replace(value):
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, str):
                return value.replace("results/model/qwen3vl", str(new_root))
            return value

        def copy_config(path):
            if path in mapping:
                return mapping[path]
            saved = str(cfg_dir / Path(path).name)
            mapping[path] = saved
            config = replace(json.loads((PROJECT_ROOT / path).read_text()))
            for key, value in list(config.items()):
                if key.endswith("_config") and isinstance(value, str) and value.startswith("configs/"):
                    config[key] = copy_config(value)
            if path == "configs/evaluation.json":
                config["result_root"] = str(new_root)
            if path == "configs/optimization.json":
                config["worker_python"] = "/selected/dense-env/bin/python"
            Path(saved).write_text(json.dumps(config))
            return saved

        job = copy_config("configs/pipeline.json")
        with patch("subprocess.Popen", side_effect=AssertionError("read only")), \
                patch("subprocess.check_output", side_effect=AssertionError("no GPU query")):
            pipeline = Pipeline(job)
            planned = plan(pipeline)
        self.assertEqual(pipeline.dense, new_root / "search/dense")
        self.assertEqual(planned["next_stage"], "dense")
        self.assertEqual(pipeline.command("dense")[0], "/selected/dense-env/bin/python")
        self.assertEqual(pipeline.token_snapshot()["worker_python"], "/selected/dense-env/bin/python")
        self.assertFalse(new_root.exists())
        self.assertNotIn("/selected/dense-env", pipeline.configs["quantization"]["candidates"][0].get("worker_python", ""))

    def test_backend_import_failure_is_a_preflight_block(self):
        with patch("adapters.base.import_module", side_effect=ImportError("incompatible backend")):
            with self.assertRaisesRegex(AdapterError, "blocked_preflight.*selected Worker environment"):
                self.adapter.resolve("dense_load")

    def test_hardware_dispatch_and_selected_worker_python_are_preserved(self):
        from search.optimize import hardware as identify, launch

        class TestHardware(HardwareAdapter):
            id = "test_cpu"
            worker_device = "cpu:0"
            stages = ("dense",)
            entrypoints = {name: "test_adapter_execution:fake_entry" for name in ("probe_identity", "launch_worker")}

            def describe(self):
                return {"id": self.id}

            def validate_device(self, device):
                if device != 3:
                    raise AdapterError("bad test device")

        platform = TestHardware()
        model = self.model | {"hardware": "test_cpu", "device": "cpu:0"}
        run = Mock(return_value={"status": "completed"})
        funcs = {"probe_identity": Mock(return_value="test cpu"), "launch_worker": run}
        request = {"operation": "control", "model": model, "device": 3,
                   "worker_python": "/isolated/environment/bin/python"}
        with patch.dict(hardware._items, {platform.id: platform}), \
                patch.object(platform, "resolve", side_effect=funcs.__getitem__), \
                patch("subprocess.Popen", side_effect=AssertionError("no GPU Worker")):
            self.assertEqual(identify(3, model), "test cpu")
            self.assertEqual(launch(request, self.root / "worker"), {"status": "completed"})
            run.assert_called_once_with(request, self.root / "worker")
            with self.assertRaisesRegex(AdapterError, "no implemented token"):
                launch(request | {"operation": "candidate"}, self.root / "blocked")
        self.assertFalse((self.root / "blocked").exists())

    def test_preflight_cli_is_json_only_and_no_inference_imports(self):
        script = """
import json, sys
from search.registry import preflight
model = json.load(open('configs/model.json'))
result = preflight(model, device=0)
assert result['status'] == 'ready_for_probe'
assert not {'torch', 'transformers', 'vllm', 'gptqmodel', 'PIL'} & set(sys.modules)
print(json.dumps(result))
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(set(json.loads(result.stdout)["stages"]), set(STAGE_OPERATIONS))


if __name__ == "__main__":
    unittest.main()
