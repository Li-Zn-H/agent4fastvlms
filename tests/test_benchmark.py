"""Check measurement aggregation and exclusion of warmup from saved rounds."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from search.model import GenerationTimer

from search.benchmark import benchmark, percentile, summarize


def output(sample_id, seconds):
    return {
        "sample_id": sample_id,
        "output_text": "yes",
        "performance": {
            "end_to_end_seconds": seconds,
            "generate_seconds": seconds / 2,
            "input_tokens": 20,
            "image_tokens": 10,
            "output_tokens": 4,
            "peak_allocated_bytes": int(seconds * 100),
            "vision_encode_seconds": seconds * 0.05,
            "pruning_seconds": seconds * 0.01,
            "prefill_seconds": seconds * 0.04,
            "decode_seconds": seconds * 0.3,
            "decode_tokens": 3,
        },
    }


class BenchmarkTests(unittest.TestCase):
    def test_summary_uses_sample_times_and_maximum_memory(self):
        result = summarize([output("a", 1), output("b", 3)])
        self.assertEqual(result["total_seconds"], 4)
        self.assertEqual(result["mean_seconds"], 2)
        self.assertEqual(result["p50_seconds"], 2)
        self.assertAlmostEqual(result["p95_seconds"], 2.9)
        self.assertEqual(result["generate_total_seconds"], 2)
        self.assertEqual(result["output_tokens"], 8)
        self.assertEqual(result["peak_allocated_bytes"], 300)
        self.assertEqual(percentile([7], 0.95), 7)

    def test_phase_totals_use_token_weighted_decode_mean(self):
        first, second = output("a", 1), output("b", 3)
        first["performance"]["decode_tokens"] = 1
        second["performance"]["decode_tokens"] = 9
        result = summarize([first, second])
        self.assertAlmostEqual(result["vision_encode_seconds"], 0.2)
        self.assertAlmostEqual(result["pruning_seconds"], 0.04)
        self.assertAlmostEqual(result["prefill_seconds"], 0.16)
        self.assertAlmostEqual(result["decode_seconds"], 1.2)
        self.assertEqual(result["decode_tokens"], 10)
        self.assertAlmostEqual(result["decode_mean_seconds_per_token"], 0.12)
        first["performance"].update(decode_tokens=0, decode_seconds=0)
        self.assertIsNone(summarize([first])["decode_mean_seconds_per_token"])

    def test_warmup_is_excluded_and_candidate_reused(self):
        samples = [{"stable_row_id": "a"}, {"stable_row_id": "b"}]
        model, processor = object(), object()
        calls = []
        config = {
            "data": {"search": "unused", "media_root": "."},
            "generation": {"ignore_eos": False},
        }

        def generate(actual_model, actual_processor, rows, **kwargs):
            self.assertIs(actual_model, model)
            self.assertIs(actual_processor, processor)
            self.assertEqual(kwargs["generation"], config["generation"])
            calls.append((rows, kwargs.get("measure", False)))
            return [output(row["stable_row_id"], 1) for row in rows]

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "benchmark"
            with (
                patch("search.benchmark.load_samples", return_value=samples),
                patch("search.benchmark.generate_outputs", generate),
                redirect_stdout(io.StringIO()),
            ):
                result = benchmark(
                    model,
                    processor,
                    config,
                    output_dir=destination,
                    warmup=1,
                    repeats=2,
                )
            self.assertEqual(calls[0], (samples[:1], False))
            self.assertEqual(
                [rows[0]["stable_row_id"] for rows, measured in calls[1:] if measured],
                ["a", "b", "a", "b"],
            )
            self.assertEqual(result["round_time_range_over_mean"], 0)
            self.assertEqual(result["mean_round_seconds"], 2)
            self.assertEqual(
                json.loads((destination / "summary.json").read_text()), result
            )
            rows = [
                json.loads(line)
                for line in (destination / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [(row["round"], row["sample_id"]) for row in rows],
                [(1, "a"), (1, "b"), (2, "a"), (2, "b")],
            )
            self.assertTrue(all("output_text" not in row for row in rows))


class TimingTests(unittest.TestCase):
    def make_timer(self, pruning):
        clock = [0.0]

        class Event:
            def __init__(self):
                self.at = clock[0]

            def elapsed_time(self, end):
                return end.at - self.at

        class Module(torch.nn.Module):
            def __init__(self, durations):
                super().__init__()
                self.durations = iter(durations)

            def forward(self):
                clock[0] += next(self.durations)

        visual, decoder = Module([10]), Module([20, 3])
        if pruning:

            def compact(module, args, kwargs):
                clock[0] += 2

            decoder.register_forward_pre_hook(compact, with_kwargs=True)
        timer = GenerationTimer(
            SimpleNamespace(
                model=SimpleNamespace(visual=visual, language_model=decoder)
            ),
            pruning=pruning,
        )
        timer.event = Event
        return timer, visual, decoder

    def test_pruning_is_separate_from_prefill_and_decode(self):
        timer, visual, decoder = self.make_timer(True)
        with timer:
            visual()
            decoder()
            decoder()
        result = timer.metrics()
        self.assertEqual(result["vision_encode_seconds"], 0.01)
        self.assertEqual(result["pruning_seconds"], 0.004)
        self.assertEqual(result["prefill_seconds"], 0.02)
        self.assertEqual(result["decode_seconds"], 0.003)
        self.assertEqual(result["decode_tokens"], 1)
        self.assertEqual(result["decode_mean_seconds_per_token"], 0.003)
        self.assertEqual(len(decoder._forward_pre_hooks), 1)
        self.assertFalse(decoder._forward_hooks)
        self.assertFalse(visual._forward_pre_hooks)

    def test_dense_one_token_and_exception_cleanup(self):
        timer, visual, decoder = self.make_timer(False)
        with timer:
            visual()
            decoder()
        result = timer.metrics()
        self.assertEqual(result["pruning_seconds"], 0)
        self.assertEqual(result["decode_seconds"], 0)
        self.assertEqual(result["decode_tokens"], 0)
        self.assertIsNone(result["decode_mean_seconds_per_token"])
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with timer:
                raise RuntimeError("failed")
        self.assertFalse(decoder._forward_pre_hooks)
        self.assertFalse(decoder._forward_hooks)
        self.assertFalse(visual._forward_pre_hooks)
        self.assertFalse(visual._forward_hooks)


if __name__ == "__main__":
    unittest.main()
