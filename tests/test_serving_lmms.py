"""Serving LMMS must bind real settings and distinguish quality from errors."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search.lmms import build_command, cache_environment
from search.lmms_gate import DEFAULT_TASKS, _mechanism_file, aggregate_lmms
from search.lmms_vllm_entry import compact_prompt, runtime_receipt, sampling_kwargs
from search.serving_lmms import execute, plan, validate_context_capacity
from search.worker import save_json
from skills.quantization import digest


class ServingLmmsTests(unittest.TestCase):
    def test_native_large_image_needs_the_lmms_context_profile(self):
        # TextVQA doc 1681 has 13,635 image tokens; keep .8 leaves a
        # 10,950-token prompt. The local Search/Dev 8K profile is insufficient.
        with self.assertRaisesRegex(ValueError, "32768.*8192"):
            validate_context_capacity({"vllm_max_model_len": 8192}, 32768)
        validate_context_capacity({"vllm_max_model_len": 32768}, 32768)

    def test_native_prompt_compaction_preserves_text_and_images(self):
        self.assertEqual(compact_prompt([1, 9, 9, 9, 2, 9, 9, 3], 9, [(3, 2), (2, 1)]),
                         [1, 9, 9, 2, 9, 3])
        with self.assertRaisesRegex(ValueError, "disagree"):
            compact_prompt([1, 9, 9, 2], 9, [(3, 2)])
        with self.assertRaisesRegex(ValueError, "Missing"):
            compact_prompt([1, 2], 9, [(3, 2)])

    def test_generation_and_launcher_keep_native_protocol(self):
        self.assertEqual(sampling_kwargs({"eos_token_id": 42, "max_new_tokens": 128}),
                         {"temperature": 0., "max_tokens": 128, "stop_token_ids": [42]})
        with self.assertRaises(ValueError):
            sampling_kwargs({"num_beams": 2})
        model = {"path": "/artifact", "device": "cuda:0", "dtype": "bfloat16",
                 "attn_implementation": "sdpa", "stack_adapter": True,
                 "execution_backend": "vllm", "runtime": {"kv_cache_dtype": "auto"}}
        command = build_command(model, "pope", Path("/out"))
        self.assertTrue(command[1].endswith("lmms_vllm_entry.py"))
        self.assertNotIn("--gen_kwargs", command)
        self.assertEqual(json.loads(cache_environment(model)["A4FVLM_LMMS_VLLM_MODEL"]), model)
        with patch.dict("os.environ", {"A4FVLM_LMMS_VLLM_MODEL": "stale"}):
            self.assertNotIn("A4FVLM_LMMS_VLLM_MODEL", cache_environment())

    def test_effective_runtime_mismatch_is_not_silently_accepted(self):
        compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=False,
            pass_config=SimpleNamespace(enable_qk_norm_rope_fusion=False), mode=3, cudagraph_mode=2)
        config = SimpleNamespace(optimization_level=1, compilation_config=compilation,
            cache_config=SimpleNamespace(cache_dtype="auto"),
            attention_config=SimpleNamespace(backend="FLASH_ATTN"),
            model_config=SimpleNamespace(max_model_len=8192))
        self.assertEqual(runtime_receipt(config, {"optimization_level": 1})["kv_cache_dtype"], "auto")
        with self.assertRaisesRegex(ValueError, "kv_cache_dtype"):
            runtime_receipt(config, {"kv_cache_dtype": "fp8"})
        with self.assertRaisesRegex(ValueError, "attention"):
            runtime_receipt(config, {"attention_backend": "FLASHINFER"})

    def test_vllm_mechanism_cannot_reuse_native_or_uncompacted_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "pope-token-mechanism.jsonl"
            image = {"image_tokens_before": 10, "image_tokens_after": 8,
                     "mrope_positions": 8, "selected_indices_sha256": "hash", "token_keep_ratio": .8}
            row = {"backend": "vllm", "controller": "vllm_score_based_post_encoder",
                   "token_keep_ratio": .8, "image_tokens_before": 10, "image_tokens_after": 8,
                   "images": [image], "sample_index": 0, "prompt_verified": True,
                   "native_prompt_sha256": "hash", "native_prompt_tokens": 20, "decoder_tokens_after": 18}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(_mechanism_file(root, "pope", 1, .8, backend="vllm"), path)
            for update in ({"backend": "native_transformers"}, {"decoder_tokens_after": 20},
                           {"prompt_verified": False}, {"sample_index": 1}):
                path.write_text(json.dumps(row | update) + "\n")
                with self.assertRaisesRegex(ValueError, "vLLM"):
                    _mechanism_file(root, "pope", 1, .8, backend="vllm")

    def fixture(self, root):
        for directory in ("stack/final", "runtime/final", "kv/final", "runtime/candidates/baseline-auto-o2"):
            (root / directory).mkdir(parents=True)
        model_path = root / "model.json"
        model = {"path": str(root / "artifact"), "token_keep_ratio": .8,
                 "token_selection": {"score_dtype": "float32", "coverage_grid": {"rows": 6, "columns": 6}},
                 "attn_implementation": "sdpa"}
        save_json(model_path, model)
        stack = {"artifact": model["path"], "token_keep_ratio": .8,
                 "lmms": {"quality_gate": "passed", "model_config": str(model_path)}}
        stack_path = root / "stack/final/stack-freeze.json"
        save_json(stack_path, stack)
        quality = {"status": "pending_full_lmms"}
        runtime = {"status": "runtime_frozen", "candidate_id": "compile-o1",
                   "runtime": {"optimization_level": 1}, "artifact": model["path"],
                   "token_keep_ratio": .8, "quality": quality,
                   "benchmarks": {"search": {"status": "passed", "summary": {"elapsed_seconds": 8, "request_count": 128, "output_token_throughput": 12}}},
                   "parent_freeze": str(stack_path), "parent_digest": digest(stack)}
        runtime_path = root / "runtime/final/runtime-freeze.json"
        save_json(runtime_path, runtime)
        kv = {"status": "kv_cache_frozen", "candidate_id": "kv-auto", "runtime": runtime["runtime"] | {"kv_cache_dtype": "auto"},
              "artifact": model["path"], "token_keep_ratio": .8, "quality": quality,
              "benchmarks": runtime["benchmarks"],
              "parent_freeze": str(runtime_path), "parent_digest": digest(runtime)}
        kv_path = root / "kv/final/kv-cache-freeze.json"
        save_json(kv_path, kv)
        fallback = runtime | {"status": "performance_ready", "runtime": {"optimization_level": 2}}
        save_json(root / "runtime/candidates/baseline-auto-o2/probe.json", fallback)
        chain_path = root / "runtime/serving-state.json"
        save_json(chain_path, {"status": "completed", "stop_reason": "serving_qualified",
            "stack_digest": digest(stack), "runtime_selected": "compile-o1", "kv_selected": "kv-auto"})
        config = {"serving_state": str(chain_path), "kv_freeze": str(kv_path),
                  "output": str(root / "gate"), "tasks": list(DEFAULT_TASKS), "minimum_retention": .95,
                  "smoke_samples_per_task": 2, "baseline_root": str(root / "dense"),
                  "fallback_runtime_candidates": ["baseline-auto-o2"]}
        config_path = root / "job.json"
        save_json(config_path, config)
        return config_path, config, kv_path

    def test_full_gate_checks_observed_settings_and_dense_sample_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = {"token_keep_ratio": .8, "runtime": {"kv_cache_dtype": "auto"}}
            model_path = root / "model.json"
            save_json(model_path, model)
            for name in ("candidate", "dense"):
                directory = root / name
                directory.mkdir()
                save_json(directory / "run_results.json", {
                    "results": {"pope": {"pope_f1_score,none": .8}},
                    "n-samples": {"pope": {"effective": 1}}, "task_hashes": {"pope": "same"}})
                (directory / "run_samples_pope.jsonl").write_text('{"doc_id": 0, "doc_hash": "same"}\n')
            row = {"backend": "vllm", "controller": "vllm_score_based_post_encoder",
                   "token_keep_ratio": .8, "image_tokens_before": 10, "image_tokens_after": 8,
                   "sample_index": 0, "prompt_verified": True, "native_prompt_sha256": "hash",
                   "native_prompt_tokens": 20, "decoder_tokens_after": 18,
                   "images": [{"image_tokens_before": 10, "image_tokens_after": 8, "mrope_positions": 8,
                               "selected_indices_sha256": "hash", "token_keep_ratio": .8}]}
            (root / "candidate/pope-token-mechanism.jsonl").write_text(json.dumps(row) + "\n")
            receipt = {"backend": "vllm", "model": model,
                       "observed_model": {"token_pruning_enabled": True, "token_runtime": {"keep_ratio": .8},
                           "kv_cache_dtypes": ["auto"], "attention_backends": ["FLASH_ATTN"], "physical_widths": [128]}}
            receipt_path = root / "candidate/pope-token-mechanism.runtime.json"
            save_json(receipt_path, receipt)

            def evaluate():
                return aggregate_lmms(root / "candidate", root / "dense", tasks=[DEFAULT_TASKS[-1]],
                    token_keep_ratio=.8, model_config=str(model_path), execution_backend="vllm")

            result = evaluate()
            self.assertEqual(result["quality_gate"], "passed")
            self.assertEqual(result["runtime_scope"], "vllm_stack_runtime_kv_lmms")
            save_json(receipt_path, receipt | {"observed_model": {}})
            with self.assertRaisesRegex(ValueError, "actual"):
                evaluate()
            save_json(receipt_path, receipt)
            (root / "candidate/run_samples_pope.jsonl").write_text('{"doc_id": 1, "doc_hash": "same"}\n')
            with self.assertRaisesRegex(ValueError, "alignment"):
                evaluate()

    def test_plan_rejects_wrong_parent_and_excludes_unqualified_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, kv_path = self.fixture(root)
            self.assertEqual(len(plan(config)["candidates"]), 2)
            fallback_path = root / "runtime/candidates/baseline-auto-o2/probe.json"
            fallback = json.loads(fallback_path.read_text())
            save_json(fallback_path, fallback | {"status": "runtime_blocked"})
            self.assertEqual(len(plan(config)["candidates"]), 1)
            kv = json.loads(kv_path.read_text())
            save_json(kv_path, kv | {"parent_digest": "wrong"})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                plan(config)

    def test_quality_failure_recovers_but_execution_error_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, config, _ = self.fixture(root)
            with patch("search.serving_lmms._run_tasks", side_effect=lambda c, m, o, **kw: o.mkdir(parents=True, exist_ok=True)) as run, \
                 patch("search.serving_lmms._task_complete", return_value=True), \
                 patch("search.serving_lmms.aggregate_lmms", side_effect=[
                     {"quality_gate": "quality_rejected", "retention": .9},
                     {"quality_gate": "passed", "retention": .96}]):
                result = execute(config_path)
                self.assertEqual(result["stop_reason"], "lmms_quality_passed")
                self.assertEqual(run.call_count, 4)
                self.assertEqual(len(result["candidates"]), 2)
            self.assertTrue((root / "gate/final/serving-freeze.json").is_file())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, _, _ = self.fixture(root)
            with patch("search.serving_lmms._run_tasks", side_effect=RuntimeError("API mismatch")) as run:
                with self.assertRaisesRegex(RuntimeError, "API mismatch"):
                    execute(config_path)
                self.assertEqual(run.call_count, 1)
            state = json.loads((root / "gate/state.json").read_text())
            self.assertEqual(state["stop_reason"], "blocked_preflight")
            self.assertFalse((root / "gate/final/serving-freeze.json").exists())


if __name__ == "__main__":
    unittest.main()
