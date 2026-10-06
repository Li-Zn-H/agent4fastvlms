"""Exercise stage routing with real scoring and persisted results, without a GPU."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search.run import evaluate_candidate


class CandidateTests(unittest.TestCase):
    def run_case(self, search_correct, dev_correct, expected_action, expected_dev, force_dev=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").touch()
            baseline_dir = root / "results/model/qwen3vl/search/dense/quality"
            baseline_dir.mkdir(parents=True)
            for stage, count in (("search", 40), ("dev", 80)):
                rows = [{
                    "stable_row_id": str(i), "primary_category": "task",
                    "logical_question": "Answer?", "answers": ["yes"],
                    "scoring_type": "deterministic_rule_or_structure",
                    "media": {"logical_path": "image.jpg"},
                } for i in range(count)]
                (root / f"{stage}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows)
                )
                (baseline_dir / f"{stage}.jsonl").write_text("\n".join(
                    json.dumps({"sample_id": str(i), "output_text": "yes"})
                    for i in range(count)
                ))
            config = {
                "data": {"search": "search.jsonl", "dev": "dev.jsonl", "media_root": "."},
                "generation": {"max_new_tokens": 64, "do_sample": False, "ignore_eos": False},
                "scoring": {"baseline": "dense", "maximum_response_tokens": 64, "negation_window_tokens": 3},
                "quality": {"minimum_retention": .95, "search_direct_pass_minimum_retention": .975},
            }
            model, processor = object(), object()
            calls = []

            def generate(actual_model, actual_processor, samples, **kwargs):
                self.assertIs(actual_model, model)
                self.assertIs(actual_processor, processor)
                self.assertEqual(kwargs["generation"], config["generation"])
                sample_id = samples[0]["stable_row_id"]
                limit = search_correct if len(calls) < 40 else dev_correct
                calls.append(sample_id)
                return [{"sample_id": sample_id, "output_text": "yes" if int(sample_id) < limit else "no"}]

            destination = root / "candidate"
            with patch("search.run.PROJECT_ROOT", root), patch("search.run.generate_outputs", generate), redirect_stdout(io.StringIO()):
                result = evaluate_candidate(model, processor, config, output_dir=destination, force_dev=force_dev)
                with self.assertRaises(FileExistsError):
                    evaluate_candidate(model, processor, config, output_dir=destination)
            self.assertEqual(result["action"], expected_action)
            self.assertEqual(set(result["stages"]), {"search", "dev"} if expected_dev else {"search"})
            self.assertEqual(calls[:40], [str(i) for i in range(40)])
            self.assertEqual(calls[40:], [str(i) for i in range(80)] if expected_dev else [])
            self.assertEqual((destination / "dev.jsonl").exists(), expected_dev)
            saved = json.loads((destination / "summary.json").read_text())
            self.assertEqual(saved["action"], expected_action)
            for stage, details in saved["stages"].items():
                expected_count = 40 if stage == "search" else 80
                self.assertEqual(len((destination / f"{stage}.jsonl").read_text().splitlines()), expected_count)
                self.assertEqual(details["tasks"][0]["sample_count"], expected_count)
                self.assertEqual(details["retention"], details["candidate_mean"])

    def test_low_search_requires_dev(self):
        self.run_case(37, 0, "reject", True)

    def test_low_search_can_recover_on_dev(self):
        self.run_case(20, 76, "pass", True)

    def test_search_direct_pass_skips_dev(self):
        self.run_case(39, 0, "pass", False)

    def test_borderline_search_full_dev_pass(self):
        self.run_case(38, 76, "pass", True)

    def test_borderline_search_full_dev_reject(self):
        self.run_case(38, 75, "reject", True)

    def test_forced_parent_confirmation_uses_dev_even_when_search_passes(self):
        self.run_case(40, 75, "reject", True, force_dev=True)


if __name__ == "__main__":
    unittest.main()
