"""CPU contracts for acceleration layers on a frozen Stack."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from search.stack_acceleration import _comparison_profile, _dense_comparison, _freeze, plan, snapshot
from search.stack_serving import execute_pipeline
from skills.quantization import digest
from skills.stack_kv_cache import validate_config as validate_kv_config
from skills.stack_runtime import (
    build_vllm_runtime_kwargs,
    candidate_key,
    select_candidate,
    validate_runtime_settings,
)


class StackAccelerationTests(unittest.TestCase):
    def test_runtime_kwargs_are_translated_without_vllm_import(self):
        kwargs = build_vllm_runtime_kwargs(
            {
                "attention_backend": "FLASHINFER",
                "optimization_level": 1,
                "kv_cache_dtype": "fp8",
                "compile_mm_encoder": True,
                "cudagraph_mm_encoder": False,
                "qk_norm_rope_fusion": True,
                "vllm_gpu_memory_utilization": 0.8,
                "vllm_max_model_len": 8192,
            }
        )
        self.assertEqual(kwargs["attention_backend"], "FLASHINFER")
        self.assertEqual(kwargs["optimization_level"], 1)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["compilation_config"]["pass_config"]["enable_qk_norm_rope_fusion"], True)
        self.assertTrue(kwargs["compilation_config"]["compile_mm_encoder"])

    def test_runtime_settings_reject_unknown_backend(self):
        with self.assertRaises(ValueError):
            validate_runtime_settings({"attention_backend": "TRITON"})

    def test_runtime_candidate_identity_includes_parent(self):
        first = candidate_key("parent-a", {"id": "fp8", "runtime": {"kv_cache_dtype": "fp8"}})
        second = candidate_key("parent-b", {"id": "fp8", "runtime": {"kv_cache_dtype": "fp8"}})
        self.assertNotEqual(first, second)

    def test_runtime_plan_is_read_only_and_ready(self):
        result = plan("configs/stack_runtime.json", kind="runtime")
        self.assertEqual(result["status"], "ready")
        configured = json.loads(Path("configs/stack_runtime.json").read_text())
        self.assertEqual(result["candidate_count"], len(configured["candidates"]))
        snap = snapshot("configs/stack_runtime.json", kind="runtime")
        parent = json.loads(Path(snap["parent"]["path"]).read_text())
        self.assertEqual(snap["parent"]["freeze"]["token_keep_ratio"], parent["token_keep_ratio"])

    def test_kv_candidates_inherit_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path("configs/stack_kv_cache.json").read_text())
            config["parent_freeze"] = "results/model/qwen3vl/stack-runtime/final/runtime-freeze.json"
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            snap = snapshot(path, kind="kv_cache")
        runtime = snap["candidates"][0]["runtime"]
        self.assertEqual(runtime["attention_backend"], "FLASHINFER")
        self.assertEqual(runtime["optimization_level"], 2)
        self.assertEqual(runtime["kv_cache_dtype"], "auto")

    def test_runtime_freeze_preserves_stack_token_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            snap = {
                "kind": "runtime",
                "parent": {
                    "path": "/tmp/stack-freeze.json",
                    "digest": "parent-digest",
                    "artifact": "/tmp/artifact",
                    "freeze": {"token_keep_ratio": 0.5},
                },
            }
            selected = {
                "candidate_id": "flashinfer",
                "candidate_key": "candidate-key",
                "runtime": {"attention_backend": "FLASHINFER"},
                "benchmarks": {},
                "quality": {},
            }
            freeze = _freeze(snap, selected, Path(directory))
            self.assertEqual(freeze["token_keep_ratio"], 0.5)

    def test_kv_plan_waits_for_runtime_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path("configs/stack_kv_cache.json").read_text())
            config["parent_freeze"] = str(Path(directory) / "missing-runtime-freeze.json")
            config_path = Path(directory) / "stack_kv_cache.json"
            config_path.write_text(json.dumps(config))
            result = plan(config_path, kind="kv_cache")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("FileNotFoundError", result["blockers"][0])

    def test_kv_plan_requires_token_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path("configs/stack_kv_cache.json").read_text())
            runtime_freeze = json.loads(
                Path("results/model/qwen3vl/stack-runtime/final/runtime-freeze.json").read_text()
            )
            runtime_freeze.pop("token_keep_ratio", None)
            runtime_path = Path(directory) / "runtime-freeze.json"
            runtime_path.write_text(json.dumps(runtime_freeze))
            config["parent_freeze"] = str(runtime_path)
            config_path = Path(directory) / "stack_kv_cache.json"
            config_path.write_text(json.dumps(config))
            result = plan(config_path, kind="kv_cache")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("token_keep_ratio", result["blockers"][0])

    def test_kv_config_requires_supported_dtype(self):
        config = {
            "parent_freeze": "parent.json",
            "output": "out",
            "device": 0,
            "evaluation_config": "evaluation.json",
            "vision_token_config": "vision.json",
            "quality": {"minimum_dev_retention": 0.95},
            "candidates": [{"id": "bad", "kv_cache_dtype": "int2"}],
        }
        with self.assertRaises(ValueError):
            validate_kv_config(config)

    def test_runtime_selector_uses_search_output_throughput(self):
        rows = [
            {
                "candidate_id": "slow",
                "status": "performance_ready",
                "quality": {"status": "pending_full_lmms"},
                "benchmarks": {"search": {"status": "passed", "summary": {"elapsed_seconds": 10, "request_count": 128, "output_token_throughput": 10}}},
            },
            {
                "candidate_id": "fast",
                "status": "performance_ready",
                "quality": {"status": "pending_full_lmms"},
                "benchmarks": {"search": {"status": "passed", "summary": {"elapsed_seconds": 8, "request_count": 128, "output_token_throughput": 12}}},
            },
        ]
        self.assertEqual(select_candidate(rows)["candidate_id"], "fast")

    def test_kv_comparison_uses_batch_one_profile(self):
        self.assertEqual(_comparison_profile("runtime"), "search")
        self.assertEqual(_comparison_profile("kv_cache"), "batch1")

    def test_dense_comparison_normalizes_repeats_and_labels_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.json"
            path.write_text(json.dumps({"rounds": [
                {"sample_count": 2, "total_seconds": 10, "output_tokens": 100},
                {"sample_count": 2, "total_seconds": 10, "output_tokens": 100},
            ]}))
            snap = {"config": {"dense_benchmark": str(path)}, "benchmark": {"search_limit": 2}}
            probe = {"benchmarks": {"batch4": {
                "summary": {"request_count": 6, "elapsed_seconds": 6, "output_token_throughput": 40},
                "runtime": {"vllm_benchmark_batch_size": 4},
            }}}
            comparison = _dense_comparison(snap, probe)
            profile = comparison["profiles"]["batch4"]
            self.assertEqual(profile["scope"], "concurrent_serving")
            self.assertEqual(profile["request_throughput_speedup"], 5)
            self.assertEqual(profile["output_token_throughput_speedup"], 4)
            snap["benchmark"]["search_limit"] = 3
            with self.assertRaises(ValueError):
                _dense_comparison(snap, probe)

    def test_serving_chain_uses_new_runtime_freeze_and_stops_without_winner(self):
        for runtime_pass in (False, True):
            with self.subTest(runtime_pass=runtime_pass), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent_path = root / "stack.json"
                parent = {"candidate_id": "stack-new", "token_keep_ratio": .8,
                          "lmms": {"quality_gate": "passed"}}
                parent_path.write_text(json.dumps(parent))
                runtime_output = root / "runtime"
                runtime_freeze = runtime_output / "final/runtime-freeze.json"
                kv_config = root / "kv.json"
                kv_config.write_text(json.dumps({"parent_freeze": str(runtime_freeze)}))
                snap = {"config": {"output": str(runtime_output)}, "parent": {
                    "freeze": parent, "path": str(parent_path), "digest": digest(parent),
                }}

                def stage_execute(config, *, kind):
                    if kind == "runtime":
                        runtime_freeze.parent.mkdir(parents=True, exist_ok=True)
                        runtime_freeze.write_text(json.dumps({"parent_digest": digest(parent)}))
                        return {"selected_candidate": "runtime-new" if runtime_pass else None}
                    self.assertEqual(json.loads(runtime_freeze.read_text())["parent_digest"], digest(parent))
                    return {"selected_candidate": "kv-new"}

                with patch("search.stack_serving.snapshot", return_value=snap), \
                        patch("search.stack_serving.execute", side_effect=stage_execute) as execute_stage:
                    result = execute_pipeline("runtime.json", kv_config)
                self.assertEqual(execute_stage.call_count, 2 if runtime_pass else 1)
                self.assertEqual(result["status"], "completed" if runtime_pass else "blocked")
                self.assertEqual(result["stack_candidate"], "stack-new")


if __name__ == "__main__":
    unittest.main()
