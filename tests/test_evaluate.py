"""CPU-only regression tests, independent of local datasets and model packages."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from search.evaluate import compare_outputs, load_samples, score_answer
from search.run import quality_action


OPTIONS = {"maximum_response_tokens": 64, "negation_window_tokens": 3}
QUALITY = {"minimum_retention": 0.95, "search_direct_pass_minimum_retention": 0.975}
STRICT = "deterministic_rule_or_structure"
SHORT = "exact_or_normalized_short_answer"
OPEN = "open_response_or_explicit_review"


def sample(sample_id="s1", task="task", *, answers=None, scoring_type=SHORT):
    return {
        "stable_row_id": sample_id,
        "primary_category": task,
        "answers": ["left"] if answers is None else answers,
        "scoring_type": scoring_type,
        "logical_question": "Where is the chair?",
        "media": {"logical_path": "image.jpg"},
    }


def output(sample_id="s1", text="left"):
    return {"sample_id": sample_id, "output_text": text}


class AnswerTests(unittest.TestCase):
    def test_strict_unicode_whitespace_and_punctuation(self):
        for text, answers, expected in [
            ("  ＣＯＵＮＴＲＹ\nName ", ["Country Name"], 1),
            ("Country Name.", ["Country Name"], 0),
            ("The answer is 52%.", ["52%", "52"], 0),
            ("52%", ["52%", "52"], 1),
            ("", ["left"], 0),
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    score_answer(text, answers, STRICT, **OPTIONS), expected
                )

    def test_short_and_open_answer_spans(self):
        for kind in (SHORT, OPEN):
            for text, answers, expected in [
                ("The chair is on the left side.", ["right", "left"], 1),
                ("leftover", ["left"], 0),
                ("red small car", ["red car"], 0),
                ("The RED car.", ["a red car"], 1),
                ("The chair is not on the left side.", ["left"], 0),
                ("not one two left", ["left"], 0),
                ("not one two three left", ["left"], 1),
                ("not left; instead the chair is left", ["left"], 1),
                ("", ["left"], 0),
                ("the a an", ["left"], 0),
                ("left", ["the"], 0),
            ]:
                with self.subTest(kind=kind, text=text):
                    self.assertEqual(
                        score_answer(text, answers, kind, **OPTIONS), expected
                    )

    def test_limits_count_normalized_tokens_and_only_apply_to_spans(self):
        at_limit = " ".join(["word"] * 63 + ["left"])
        over_limit = "word " + at_limit
        for kind in (SHORT, OPEN):
            self.assertEqual(score_answer(at_limit, ["left"], kind, **OPTIONS), 1)
            self.assertEqual(score_answer(over_limit, ["left"], kind, **OPTIONS), 0)
            self.assertEqual(
                score_answer("the " * 80 + at_limit, ["left"], kind, **OPTIONS), 1
            )
        self.assertEqual(score_answer(over_limit, [over_limit], STRICT, **OPTIONS), 1)

    def test_parameters_are_used(self):
        self.assertEqual(
            score_answer(
                "word left",
                ["left"],
                SHORT,
                maximum_response_tokens=1,
                negation_window_tokens=3,
            ),
            0,
        )
        self.assertEqual(
            score_answer(
                "not left",
                ["left"],
                SHORT,
                maximum_response_tokens=64,
                negation_window_tokens=0,
            ),
            1,
        )

    def test_preserved_legacy_limitations(self):
        # These are compatibility cases, not claims of semantic correctness.
        for text, answer, expected in [
            ("3.14", "3", 1),
            ("not red, blue", "blue", 0),
            ("red, blue or green", "blue", 1),
            ("isn't left", "left", 1),
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    score_answer(text, [answer], SHORT, **OPTIONS), expected
                )

    def test_invalid_targets_outputs_and_parameters_raise(self):
        for answers in ([], "left", [""], ["  "]):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                score_answer("left", answers, SHORT, **OPTIONS)
        for kind in ("short_answer", "unknown", None):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                score_answer("left", ["left"], kind, **OPTIONS)
        with self.assertRaises(TypeError):
            score_answer(None, ["left"], SHORT, **OPTIONS)
        for key, value in [
            ("maximum_response_tokens", 0),
            ("negation_window_tokens", -1),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                score_answer("left", ["left"], SHORT, **(OPTIONS | {key: value}))


class PairingAndAggregationTests(unittest.TestCase):
    def test_output_order_does_not_change_pairing(self):
        result = compare_outputs(
            [sample("a", "a"), sample("b", "b")],
            [output("a"), output("b")],
            [output("b", "right"), output("a")],
            **OPTIONS,
        )
        self.assertEqual(
            [(r.task_id, r.candidate_mean) for r in result.tasks], [("a", 1), ("b", 0)]
        )
        self.assertEqual(result.retention, 0.5)

    def test_incomplete_duplicate_extra_and_invalid_outputs_raise(self):
        for outputs in [
            [],
            [output(), output()],
            [output("extra")],
            [output(), output("extra")],
        ]:
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                compare_outputs([sample()], [output()], outputs, **OPTIONS)
        with self.assertRaises(TypeError):
            compare_outputs([sample()], [output()], [output(text=None)], **OPTIONS)
        with self.assertRaises(ValueError):
            compare_outputs([sample(), sample()], [output()], [output()], **OPTIONS)
        with self.assertRaises(ValueError):
            compare_outputs([], [], [], **OPTIONS)

    def test_ratio_of_equal_task_means_with_unequal_task_sizes(self):
        samples = [sample("a", "a")] + [sample(f"b{i}", "b") for i in range(3)]
        baseline = [
            output("a"),
            output("b0"),
            output("b1", "right"),
            output("b2", "right"),
        ]
        candidate = [output("a", "right")] + [output(f"b{i}") for i in range(3)]
        result = compare_outputs(samples, baseline, candidate, **OPTIONS)
        self.assertAlmostEqual(result.baseline_mean, 2 / 3)
        self.assertEqual(result.candidate_mean, 0.5)
        self.assertAlmostEqual(result.retention, 0.75)
        # Sample-weighted ratio and mean of task ratios would both be 1.5 here.
        self.assertNotAlmostEqual(result.retention, 1.5)
        self.assertEqual([t.sample_count for t in result.tasks], [1, 3])

    def test_zero_task_is_included_and_retention_can_exceed_one(self):
        result = compare_outputs(
            [sample("a", "a"), sample("b", "b")],
            [output("a", "right"), output("b")],
            [output("a"), output("b")],
            **OPTIONS,
        )
        self.assertEqual(result.baseline_mean, 0.5)
        self.assertEqual(result.retention, 2.0)

    def test_zero_baseline_macro_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "zero-score baseline"):
            compare_outputs([sample()], [output(text="right")], [output()], **OPTIONS)


class LoaderTests(unittest.TestCase):
    def test_existing_row_format_and_media_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.jpg").write_bytes(b"path check only")
            path = root / "samples.jsonl"
            row = sample()
            path.write_text(json.dumps(row) + "\n\n")
            self.assertEqual(load_samples(path, media_root=root), (row,))
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_samples(path, media_root=root)
            path.write_text(json.dumps(row))
            (root / "image.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "Image is missing"):
                load_samples(path, media_root=root)


class QualityGateTests(unittest.TestCase):
    def test_exact_search_and_dev_boundaries(self):
        for retention, search_action, dev_action in [
            (0.0, "dev", "reject"),
            (math.nextafter(0.95, 0), "dev", "reject"),
            (0.95, "dev", "pass"),
            (0.96, "dev", "pass"),
            (math.nextafter(0.975, 0), "dev", "pass"),
            (0.975, "pass", "pass"),
            (1.1, "pass", "pass"),
        ]:
            with self.subTest(retention=retention):
                self.assertEqual(
                    quality_action(retention, stage="search", **QUALITY), search_action
                )
                self.assertEqual(
                    quality_action(retention, stage="dev", **QUALITY), dev_action
                )

    def test_custom_thresholds_are_used(self):
        self.assertEqual(
            quality_action(
                0.8,
                stage="search",
                minimum_retention=0.7,
                search_direct_pass_minimum_retention=0.8,
            ),
            "pass",
        )

    def test_invalid_inputs_raise_instead_of_passing(self):
        for retention in (-1, float("nan"), float("inf")):
            with self.subTest(retention=retention), self.assertRaises(ValueError):
                quality_action(retention, stage="search", **QUALITY)
        with self.assertRaises(ValueError):
            quality_action(1, stage="unknown", **QUALITY)
        with self.assertRaises(ValueError):
            quality_action(
                1,
                stage="search",
                minimum_retention=0.98,
                search_direct_pass_minimum_retention=0.95,
            )


if __name__ == "__main__":
    unittest.main()
