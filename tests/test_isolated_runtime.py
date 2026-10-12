"""Split quantizer/runtime process boundaries, without GPU or backend imports."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search.quantization import _preflight, _candidate_digest, _quantization_identity
from search.worker import _runtime_import_status, _vllm_reload_subprocess, run_quantization_vllm_benchmark


class IsolatedRuntimeTests(unittest.TestCase):
    def test_framework_probe_imports_in_runtime_and_retains_child_failures(self):
        receipt = {"vllm": {"available": True, "version": "test"}}
        result = SimpleNamespace(stdout="dependency startup log\na4fvlm_runtime_imports=" + json.dumps(receipt))
        with patch("search.worker.subprocess.run", return_value=result) as run:
            self.assertEqual(_runtime_import_status(["vllm"], {"runtime_python": "/runtime/bin/python"}), receipt)
        self.assertEqual(run.call_args.args[0][0], "/runtime/bin/python")
        self.assertTrue(run.call_args.kwargs["env"]["PATH"].startswith("/runtime/bin"))
        with patch("search.worker.subprocess.run", side_effect=RuntimeError("runtime failed")):
            result = _runtime_import_status(["vllm"], {"runtime_python": "/runtime/bin/python"})
        self.assertFalse(result["vllm"]["available"])
        self.assertIn("runtime failed", result["vllm"]["error"])

    def test_preflight_checks_each_dependency_in_its_own_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant, runtime, data, requirements = [root / name for name in ("quant", "runtime", "data.jsonl", "requirements.txt")]
            for path in (quant, runtime, data, requirements):
                path.touch()
            candidate = {"id": "split", "route": "gptqmodel_w4a16", "framework": "GPTQModel",
                         "runtime": "vllm_gptq_marlin", "worker_python": str(quant),
                         "imports": ["gptqmodel"], "runtime_imports": ["vllm"],
                         "runtime_config": {"quantize": True, "runtime_python": str(runtime)},
                         "calibration": {"data": str(data)}, "requirements": str(requirements)}
            metadata = {"versions": {}, "constraints": {}, "missing_distributions": [],
                        "dependency_conflicts": [], "probe_error": None}
            with patch("search.quantization._module_available", return_value=True) as imports, \
                 patch("search.quantization._dependency_preflight", return_value=metadata) as dependencies:
                result = _preflight({"model_path": str(root), "model": {"adapter": "internvl35"},
                                     "output": str(root / "output")}, candidate)
            self.assertTrue(result["ready_for_worker"])
            self.assertEqual([call.args for call in imports.call_args_list],
                             [("gptqmodel", str(quant)), ("vllm", str(runtime))])
            self.assertEqual(dependencies.call_args_list[0].args, (candidate, str(quant), False))
            self.assertEqual(dependencies.call_args_list[1].args, ({"imports": ["vllm"]}, str(runtime), False))

    def test_reload_uses_runtime_interpreter_and_its_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            python = str(Path(directory) / "runtime/bin/python")
            def child(*args, **kwargs):
                kwargs['stdout'].write('a4fvlm_stack_adapter={"architecture":"native"}\nvllm_reload_ok\n')
                return SimpleNamespace(returncode=0)
            with patch("search.worker.subprocess.run", side_effect=child) as run:
                observed = _vllm_reload_subprocess(Path(directory) / "artifact", {"runtime_python": python})
            self.assertEqual(observed["architecture"], "native")
            self.assertEqual(run.call_args.args[0][0], python)
            self.assertTrue(run.call_args.kwargs["env"]["PATH"].startswith(str(Path(python).parent)))

    def test_reload_timeout_preserves_readable_startup_log(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            def child(*args, **kwargs):
                kwargs['stdout'].write('Reached distributed initialization\n')
                kwargs['stdout'].flush()
                logs = list((Path(directory) / 'evidence').glob('vllm-reload-*.log'))
                self.assertEqual(len(logs), 1)
                self.assertIn('distributed initialization', logs[0].read_text())
                raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
            with patch('search.worker.subprocess.run', side_effect=child), self.assertRaises(subprocess.TimeoutExpired):
                _vllm_reload_subprocess(Path(directory) / 'artifact', {'serve_timeout_seconds': 1})
            logs = list((Path(directory) / 'evidence').glob('vllm-reload-*.log'))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].read_text(), 'Reached distributed initialization\n')

    def test_benchmark_handoff_preserves_request_and_uses_existing_worker_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = str(root / "runtime/bin/python")
            request = {"operation": "quantization_vllm_benchmark", "device": 0,
                       "hardware": "GPU-TEST, Example, driver", "model": {"path": str(root / "artifact")},
                       "runtime": {"runtime_python": python, "vllm_max_model_len": 8192},
                       "probe_output": str(root / "benchmark"),
                       "vision_token": {"keep_ratio": .8, "selection": {"test": True}}}
            result = {"status": "completed", "probe_status": "vllm_benchmark_passed"}
            with patch("search.optimize._cuda_launch", return_value=result) as launch:
                self.assertEqual(run_quantization_vllm_benchmark(request), result)
            child, destination = launch.call_args.args
            self.assertEqual(child, {**request, "worker_python": python})
            self.assertEqual(destination, root / "benchmark/runtime-worker")
            self.assertFalse(root.joinpath("benchmark").exists())

    def test_changed_runtime_invalidates_runtime_receipt_but_not_export_identity(self):
        first = {"route": "gptqmodel_w4a16", "worker_python": "/quant/python",
                 "runtime_config": {"quantize": True, "runtime_python": "/runtime-a/python"}}
        second = {**first, "runtime_config": {**first["runtime_config"], "runtime_python": "/runtime-b/python"}}
        self.assertNotEqual(_candidate_digest(first), _candidate_digest(second))
        self.assertEqual(_quantization_identity(first), _quantization_identity(second))


if __name__ == "__main__":
    unittest.main()
