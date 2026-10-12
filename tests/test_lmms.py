"""Verify native LMMS arguments, local caches, and failure propagation."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from search.lmms import TASKS, build_command, cache_environment, run_lmms
from search.lmms_gate import aggregate_lmms
from search.lmms_stack_entry import _install_token_controller

MODEL = {"path": "models/Qwen3-VL-4B-Instruct", "device": "cuda:0",
         "dtype": "bfloat16", "attn_implementation": "sdpa"}


class LmmsTests(unittest.TestCase):
    def test_resume_reuses_complete_tasks_and_restarts_interrupted_task(self):
        model = MODEL | {"stack_adapter": True, "token_keep_ratio": .7}
        tasks = ["chartqa", "docvqa_val", "pope"]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"
            destination.mkdir()
            (destination / "config.json").write_text(json.dumps({
                "model": model, "tasks": tasks, "limit": None,
            }))

            def write_task(task):
                (destination / f"{task}_results.json").write_text(json.dumps({
                    "results": {task: {}}, "n-samples": {task: {"effective": 1}},
                }))
                (destination / f"{task}_samples_{task}.jsonl").write_text('{"doc_id":0}\n')
                (destination / f"{task}-token-mechanism.jsonl").write_text(json.dumps({
                    "token_keep_ratio": .7, "image_tokens_before": 10, "image_tokens_after": 7,
                    "decoder_tokens_after": 9, "decode_steps": 1, "kv_tokens_after_last_forward": 10,
                }) + "\n")

            write_task("chartqa")
            completed_bytes = (destination / "chartqa_results.json").read_bytes()
            partial = destination / "docvqa_val-token-mechanism.jsonl"
            partial.write_text('{"interrupted":true}\n')

            def run(command, **kwargs):
                task = command[command.index("--tasks") + 1]
                self.assertNotEqual(task, "chartqa")
                self.assertFalse((destination / f"{task}-token-mechanism.jsonl").exists())
                write_task(task)

            with patch("search.lmms.subprocess.run", side_effect=run) as launch:
                run_lmms(model, tasks, output_dir=destination, resume=True)
                self.assertEqual(launch.call_count, 2)
                run_lmms(model, tasks, output_dir=destination, resume=True)
                self.assertEqual(launch.call_count, 2)
            self.assertEqual((destination / "chartqa_results.json").read_bytes(), completed_bytes)
            self.assertEqual(len(list(destination.glob('docvqa_val-token-mechanism.partial-*.jsonl'))), 1)
            with patch("search.lmms.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "config"):
                    run_lmms(model | {"token_keep_ratio": .75}, tasks, output_dir=destination, resume=True)
                launch.assert_not_called()
            (destination / 'pope_samples_pope.jsonl').write_text('')
            with patch("search.lmms.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "sample"):
                    run_lmms(model, tasks, output_dir=destination, resume=True)
                launch.assert_not_called()

    def test_lmms_gate_aggregates_matching_native_results_and_token_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense"
            candidate = root / "candidate"
            for folder, score in ((dense, .8), (candidate, .7)):
                folder.mkdir()
                (folder / "run_results.json").write_text(json.dumps({
                    "results": {"pope": {"pope_f1_score,none": score}},
                    "n-samples": {"pope": {"effective": 2}},
                    "task_hashes": {"pope": "same"},
                }))
                (folder / "run_samples_pope.jsonl").write_text(
                    '{"doc_id":"a","doc_hash":"hash-a"}\n'
                    '{"doc_id":"b","doc_hash":"hash-b"}\n'
                )
            (candidate / "pope-token-mechanism.jsonl").write_text(
                '{"token_keep_ratio":0.6,"image_tokens_before":10,"image_tokens_after":6,'
                '"decoder_tokens_after":8,"decode_steps":1,"kv_tokens_after_last_forward":9}\n'
                * 2
            )
            summary = aggregate_lmms(
                candidate, dense,
                tasks=[{"task": "pope", "metric": "pope_f1_score,none"}],
                token_keep_ratio=.6,
                minimum_retention=.95,
            )
            self.assertEqual(summary["quality_gate"], "quality_rejected")
            self.assertAlmostEqual(summary["retention"], .875)
            self.assertTrue(summary["token_mechanism_verified"])

    def test_lmms_gate_allows_coverage_floor_samples_without_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense"
            candidate = root / "candidate"
            for folder, score in ((dense, 1.0), (candidate, 1.0)):
                folder.mkdir()
                (folder / "run_results.json").write_text(json.dumps({
                    "results": {"pope": {"pope_f1_score,none": score}},
                    "n-samples": {"pope": {"effective": 2}},
                    "task_hashes": {"pope": "same"},
                }))
                (folder / "run_samples_pope.jsonl").write_text(
                    '{"doc_id":"a","doc_hash":"hash-a"}\n'
                    '{"doc_id":"b","doc_hash":"hash-b"}\n'
                )
            (candidate / "pope-token-mechanism.jsonl").write_text(
                '{"token_keep_ratio":0.5,"image_tokens_before":10,"image_tokens_after":10,'
                '"decoder_tokens_after":8,"decode_steps":1,"kv_tokens_after_last_forward":9}\n'
                '{"token_keep_ratio":0.5,"image_tokens_before":10,"image_tokens_after":5,'
                '"decoder_tokens_after":8,"decode_steps":1,"kv_tokens_after_last_forward":9}\n'
            )
            summary = aggregate_lmms(
                candidate, dense,
                tasks=[{"task": "pope", "metric": "pope_f1_score,none"}],
                token_keep_ratio=.5,
            )
            self.assertEqual(summary["quality_gate"], "passed")

    def test_native_gate_requires_complete_aligned_sample_files(self):
        samples = [{"doc_id": 0, "doc_hash": "hash-0"},
                   {"doc_id": 1, "doc_hash": "hash-1"}]
        invalid = [
            ("truncated", samples[:1], "IDs/count"),
            ("duplicate", [samples[0], samples[0]], "IDs/count"),
            ("different_id", [samples[0], samples[1] | {"doc_id": 2}], "alignment"),
            ("different_document", [samples[0], samples[1] | {"doc_hash": "wrong"}], "alignment"),
            ("missing_hash", [{"doc_id": 0}, samples[1]], "hash"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("candidate", "dense"):
                folder = root / name
                folder.mkdir()
                (folder / "run_results.json").write_text(json.dumps({
                    "results": {"pope": {"pope_f1_score,none": .8}},
                    "n-samples": {"pope": {"effective": 2}},
                    "task_hashes": {"pope": "same"},
                }))

            def write_samples(name, rows):
                (root / name / "run_samples_pope.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )

            def evaluate():
                return aggregate_lmms(root / "candidate", root / "dense", tasks=[
                    {"task": "pope", "metric": "pope_f1_score,none"},
                ])

            for folder in ("candidate", "dense"):
                for case, rows, error in invalid:
                    with self.subTest(folder=folder, case=case):
                        for name in ("candidate", "dense"):
                            write_samples(name, samples)
                        write_samples(folder, rows)
                        with self.assertRaisesRegex(ValueError, error):
                            evaluate()
            write_samples("dense", samples)
            write_samples("candidate", list(reversed(samples)))
            self.assertEqual(evaluate()["quality_gate"], "passed")

    def test_native_gate_rejects_nonfinite_scores_and_empty_reported_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("candidate", "dense"):
                folder = root / name
                folder.mkdir()
                (folder / "run_samples_pope.jsonl").write_text(
                    '{"doc_id":0,"doc_hash":"hash-0"}\n'
                )

            def write_result(name, score=.8, count=1):
                (root / name / "run_results.json").write_text(json.dumps({
                    "results": {"pope": {"pope_f1_score,none": score}},
                    "n-samples": {"pope": {"effective": count}},
                    "task_hashes": {"pope": "same"},
                }))

            def evaluate():
                return aggregate_lmms(root / "candidate", root / "dense", tasks=[
                    {"task": "pope", "metric": "pope_f1_score,none"},
                ])

            for folder in ("candidate", "dense"):
                for score in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(folder=folder, score=score):
                        for name in ("candidate", "dense"):
                            write_result(name)
                        write_result(folder, score=score)
                        with self.assertRaisesRegex(ValueError, "metric|denominator"):
                            evaluate()
            for count in (0, True):
                with self.subTest(count=count):
                    for name in ("candidate", "dense"):
                        write_result(name, count=count)
                    with self.assertRaisesRegex(ValueError, "count"):
                        evaluate()

    def test_five_tasks_and_native_generation(self):
        self.assertEqual(set(TASKS), {"mmstar", "textvqa_val", "docvqa_val", "chartqa", "pope"})
        command = build_command(MODEL, "pope", Path("results"))
        self.assertNotIn("--limit", command)
        self.assertNotIn("--gen_kwargs", command)
        args = command[command.index("--model_args") + 1]
        self.assertIn("device_map=cuda:0", args)
        self.assertNotIn("max_pixels", args)
        self.assertNotIn("dtype=", args)
        self.assertEqual(build_command(MODEL, "chartqa", Path("results"), limit=8)[-2:], ["--limit", "8"])

    def test_stack_uses_topology_binding_launcher(self):
        command = build_command(MODEL | {"stack_adapter": True}, "pope", Path("results"))
        self.assertTrue(command[1].endswith("search/lmms_stack_entry.py"))
        self.assertNotEqual(command[1:3], ["-m", "lmms_eval"])

    def test_project_local_caches(self):
        with patch("search.lmms.PROJECT_ROOT", Path("/project")):
            env = cache_environment()
        self.assertEqual(env["HF_HUB_CACHE"], "/project/data/lmms/hub")
        self.assertEqual(env["HF_DATASETS_CACHE"], "/project/data/lmms/datasets")

    def test_stack_token_settings_are_handed_to_native_launcher(self):
        model = MODEL | {
            "stack_adapter": True,
            "token_keep_ratio": 0.55,
            "token_selection": {
                "score_dtype": "float32",
                "coverage_grid": {"rows": 6, "columns": 6},
            },
        }
        env = cache_environment(model)
        self.assertEqual(env["A4FVLMSP_TOKEN_KEEP_RATIO"], "0.55")
        self.assertEqual(
            json.loads(env["A4FVLMSP_TOKEN_SELECTION"])["coverage_grid"]["rows"],
            6,
        )

    def test_dense_clears_inherited_token_settings(self):
        inherited = {key: "stale" for key in (
            "A4FVLMSP_TOKEN_KEEP_RATIO", "A4FVLMSP_TOKEN_SELECTION",
            "A4FVLMSP_TOKEN_RECEIPT",
        )}
        with patch.dict(os.environ, inherited):
            env = cache_environment(MODEL)
        for key in inherited:
            self.assertFalse(key in env, f"Inherited Token setting leaked: {key}")

    def test_chat_override_runs_controller_and_records_compact_kv(self):
        class Model:
            def generate(self, **kwargs):
                if kwargs.get("fail"):
                    raise RuntimeError("generation failed")
                return "answer"

        class SimpleAdapter:
            def __init__(self):
                self.model = Model()

            def generate_until(self, requests):
                return self.model.generate(**requests)

        class ChatAdapter(SimpleAdapter):
            # The real chat adapter overrides the base generation method.
            def generate_until(self, requests):
                return self.model.generate(**requests)

        class Controller:
            def __init__(self, model, selection, ratio):
                self.model = model
                self.calls = 0
                self.last_metrics = None

            def generate(self, inputs, options):
                self.calls += 1
                result = self.model.generate(**inputs, **options)
                self.last_metrics = {
                    "image_tokens_before": 100, "image_tokens_after": 55,
                    "decoder_tokens_before": 120, "decoder_tokens_after": 75,
                    "decode_steps": 2, "kv_tokens_after_last_forward": 77,
                }
                return result

        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "token.jsonl"
            with (
                patch.dict(os.environ, {
                    "A4FVLMSP_TOKEN_KEEP_RATIO": "0.55",
                    "A4FVLMSP_TOKEN_SELECTION": "{}",
                    "A4FVLMSP_TOKEN_RECEIPT": str(receipt),
                }),
                patch("skills.vision_token.VisionTokenController", Controller),
            ):
                _install_token_controller(SimpleNamespace(Qwen3_VL=SimpleAdapter))
                adapter = ChatAdapter()
                self.assertEqual(adapter.generate_until({"input_ids": [1]}), "answer")
                self.assertEqual(adapter._a4fvlm_token_controller.calls, 1)
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    adapter.generate_until({"input_ids": [1], "fail": True})
                self.assertEqual(adapter.generate_until({"input_ids": [1]}), "answer")
                self.assertEqual(adapter._a4fvlm_token_controller.calls, 3)
            records = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_tokens_after"], 55)
            self.assertEqual(records[0]["kv_tokens_after_last_forward"], 77)

    def test_bad_arguments_raise(self):
        for task, model, limit in (("pope_adv", MODEL, None), ("pope", MODEL, 0), ("pope", MODEL | {"dtype": "float16"}, None)):
            with self.assertRaises(ValueError):
                build_command(model, task, Path("results"), limit=limit)

    def test_process_failure_stops_remaining_tasks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("search.lmms.subprocess.run", side_effect=subprocess.CalledProcessError(1, "lmms")) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                run_lmms(MODEL, ["chartqa", "pope"], output_dir=Path(directory) / "output")
            self.assertEqual(run.call_count, 1)

    def test_missing_result_is_failure_even_with_zero_exit(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("search.lmms.subprocess.run"),
            self.assertRaisesRegex(RuntimeError, "did not produce"),
        ):
            run_lmms(MODEL, ["pope"], output_dir=Path(directory) / "output")

    def test_native_result_is_accepted_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"

            def run(command, **kwargs):
                self.assertTrue(kwargs["check"])
                task = command[command.index("--tasks") + 1]
                (destination / f"{task}_results.json").write_text(json.dumps({"results": {task: {"metric": .5}}}))

            with patch("search.lmms.subprocess.run", side_effect=run):
                run_lmms(MODEL, ["chartqa", "pope"], output_dir=destination, limit=8)
                with self.assertRaises(FileExistsError):
                    run_lmms(MODEL, ["pope"], output_dir=destination)

    def test_token_task_requires_mechanism_for_every_sample(self):
        model = MODEL | {"stack_adapter": True, "token_keep_ratio": .55}
        for count, message in ((0, "did not produce Token"), (1, "count mismatch"), (2, None)):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "output"

                def run(command, **kwargs):
                    (destination / "pope_results.json").write_text(json.dumps({
                        "results": {"pope": {}}, "n-samples": {"pope": {"effective": 2}},
                    }))
                    if count:
                        Path(kwargs["env"]["A4FVLMSP_TOKEN_RECEIPT"]).write_text(
                            (json.dumps({"image_tokens_before": 100, "image_tokens_after": 55}) + "\n") * count
                        )

                with patch("search.lmms.subprocess.run", side_effect=run):
                    if message:
                        with self.assertRaisesRegex(RuntimeError, message):
                            run_lmms(model, ["pope"], output_dir=destination, limit=2)
                    else:
                        run_lmms(model, ["pope"], output_dir=destination, limit=2)


if __name__ == "__main__":
    unittest.main()
