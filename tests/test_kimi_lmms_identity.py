"""Reject same-ID Kimi evaluations with changed rasters or weak input evidence."""

import json
from pathlib import Path
import tempfile
import unittest

from search.kimi_lmms_identity import (
    bind_receipt, completed_input, digest, input_identity, paths, validated_inputs,
)
from search.lmms_gate import aggregate_lmms


class Raster:
    size = (2, 1)

    def __init__(self, content=b"\xff\x00\x00\x00\xff\x00"):
        self.content = content

    def convert(self, mode):
        assert mode == "RGB"
        return self

    def tobytes(self):
        return self.content


class IdentityTests(unittest.TestCase):
    def write_task(self, root, *, image=None, generation=None, origin="observed_generation", split="test"):
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"model": {"adapter": "kimi_vl"}}))
        (root / "run_results.json").write_text(json.dumps({
            "results": {"pope": {"pope_f1_score,none": 1.}},
            "task_hashes": {"pope": "same-native-task"},
            "n-samples": {"pope": {"effective": 1}},
            "configs": {"pope": {"test_split": split}},
        }))
        (root / "run_samples_pope.jsonl").write_text(json.dumps({
            "doc_id": 0, "doc_hash": digest(None), "input": "Is it red?",
            "target": "yes", "filtered_resps": "yes",
        }) + "\n")
        identity = input_identity("pope", split, 0, "Is it red?", image or Raster(),
                                  generation or {"max_new_tokens": 16}, [])
        row_path, _ = paths(root, "pope")
        row_path.write_text(json.dumps(completed_input(identity, "yes")) + "\n")
        bind_receipt(root, "pope", origin=origin,
                     dataset_fingerprint="cached-native-revision" if origin != "observed_generation" else None)

    def aggregate(self, candidate, dense):
        return aggregate_lmms(candidate, dense, tasks=[{"task": "pope", "metric": "pope_f1_score,none"}])

    def test_original_reference_reconstruction_and_observed_candidate_align(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root / "dense", origin="reconstructed_original_dense")
            self.write_task(root / "candidate")
            summary = self.aggregate(root / "candidate", root / "dense")
            self.assertEqual(summary["quality_gate"], "passed")
            self.assertEqual(summary["retention"], 1.)

    def test_same_null_document_hash_cannot_hide_changed_image_or_generation(self):
        for change in ({"image": Raster(b"\x00" * 6)}, {"generation": {"max_new_tokens": 8}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_task(root / "dense")
                self.write_task(root / "candidate", **change)
                with self.assertRaisesRegex(ValueError, "native input alignment"):
                    self.aggregate(root / "candidate", root / "dense")

    def test_reconstructed_candidate_is_not_inference_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root / "dense")
            self.write_task(root / "candidate", origin="reconstructed_original_dense")
            with self.assertRaisesRegex(ValueError, "receipt binding"):
                self.aggregate(root / "candidate", root / "dense")

    def test_changed_raw_score_samples_config_and_receipt_are_rejected(self):
        for filename in ("run_results.json", "run_samples_pope.jsonl", "config.json", "pope-native-inputs.jsonl"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "task"
                self.write_task(root)
                path = root / filename
                path.write_text(path.read_text() + " ")
                # Whitespace in config preserves its object, so change the model.
                if filename == "config.json":
                    path.write_text(json.dumps({"model": {"adapter": "kimi_vl", "path": "changed"}}))
                with self.assertRaisesRegex(ValueError, "receipt binding"):
                    validated_inputs(root, "pope")

    def test_duplicate_missing_or_forged_rows_cannot_be_bound(self):
        for change in ("duplicate", "missing", "prompt", "response", "hash"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "task"
                self.write_task(root)
                row_path, receipt_path = paths(root, "pope")
                row = json.loads(row_path.read_text())
                receipt_path.unlink()
                rows = [row]
                if change == "duplicate": rows *= 2
                elif change == "missing": rows = []
                elif change == "prompt": row["prompt"] = "Different question"
                elif change == "response": row["response_sha256"] = digest("no")
                elif change == "hash": row["image_rgb_sha256"] = "0" * 64
                row_path.write_text("".join(json.dumps(value) + "\n" for value in rows))
                with self.assertRaises(ValueError):
                    bind_receipt(root, "pope")

    def test_missing_receipt_and_reconstructed_optimized_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task"
            self.write_task(root)
            paths(root, "pope")[1].unlink()
            with self.assertRaises(FileNotFoundError):
                validated_inputs(root, "pope")
            (root / "config.json").write_text(json.dumps({"model": {"adapter": "kimi_vl", "stack_adapter": True}}))
            with self.assertRaisesRegex(ValueError, "Only original Dense"):
                bind_receipt(root, "pope", origin="reconstructed_original_dense", dataset_fingerprint="revision")

    def test_native_val_split_is_accepted_and_changed_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "task"
            self.write_task(root, split="val")
            self.assertEqual(len(validated_inputs(root, "pope")), 1)
            row_path, receipt_path = paths(root, "pope")
            receipt_path.unlink()
            identity = input_identity("pope", "test", 0, "Is it red?", Raster(), {"max_new_tokens": 16}, [])
            row_path.write_text(json.dumps(completed_input(identity, "yes")) + "\n")
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                bind_receipt(root, "pope")


if __name__ == "__main__":
    unittest.main()
