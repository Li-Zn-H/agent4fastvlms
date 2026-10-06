"""Pure contracts for the isolated vLLM comparison benchmark."""

import unittest
import os
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import torch

from search.vllm_benchmark import (
    chat_prompt,
    compare_reports,
    summarize_round,
    visual_token_budget,
)
from skills.vllm_token import (
    TOKEN_RUNTIME_ENV,
    configured_runtime,
    configure_runtime,
    image_keep_count,
    select_image_indices,
)
from skills.vision_token import select_indices


class VllmBenchmarkTests(unittest.TestCase):
    def test_streaming_counts_llava_image_token_index(self):
        from search.worker import _stream_vllm_request

        async def generate(*args):
            yield SimpleNamespace(outputs=[SimpleNamespace(text="answer", token_ids=[5])])

        engine = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(image_token_index=9)),
            renderer=SimpleNamespace(render_chat=lambda *args: (None, [{"prompt_token_ids": [1, 9, 9, 2]}])),
            generate=generate)
        renderers = SimpleNamespace(ChatParams=lambda **kwargs: kwargs)
        with patch.dict("sys.modules", {"vllm.renderers": renderers}):
            result = asyncio.run(_stream_vllm_request(engine, {"conversation": []}, None, "test"))
        self.assertEqual(result["image_tokens"], 2)
        self.assertEqual(result["tokens"], 1)

    def test_chat_prompt_uses_local_openai_multimodal_shape(self):
        prompt = chat_prompt({"logical_question": "What is shown?"}, "/tmp/a.jpg")
        self.assertEqual(prompt[0]["role"], "user")
        self.assertEqual(
            prompt[0]["content"][0]["image_url"]["url"], "file:///tmp/a.jpg"
        )

    def test_summary_keeps_unavailable_ttft_explicit(self):
        summary = summarize_round(
            elapsed_seconds=2.0,
            output_tokens=20,
            request_latencies=[0.5, 1.0],
            ttfts=[None, None],
            request_count=2,
        )
        self.assertFalse(summary["ttft_available"])
        self.assertIsNone(summary["mean_ttft_seconds"])
        self.assertEqual(summary["output_token_throughput"], 10.0)

    def test_comparison_reports_speed_and_memory_ratios(self):
        dense = {
            "summary": summarize_round(
                elapsed_seconds=4,
                output_tokens=40,
                request_latencies=[2, 2],
                ttfts=[None, None],
                request_count=2,
            ),
            "peak_gpu_memory_bytes": 100,
        }
        quantized = {
            "summary": summarize_round(
                elapsed_seconds=2,
                output_tokens=40,
                request_latencies=[1, 1],
                ttfts=[None, None],
                request_count=2,
            ),
            "peak_gpu_memory_bytes": 75,
        }
        result = compare_reports(dense, quantized)
        self.assertEqual(result["output_token_speedup"], 2.0)
        self.assertEqual(result["latency_ratio"], 0.5)
        self.assertEqual(result["memory_saving_ratio"], 0.25)
        self.assertIsNone(result["ttft_ratio"])

    def test_visual_token_budget_is_aligned_and_ratio_bounded(self):
        budget = visual_token_budget(
            image_width=1024,
            image_height=768,
            keep_ratio=0.5,
            min_pixels=64 * 1024,
            max_pixels=16 * 1024 * 1024,
            patch_size=16,
            merge_size=2,
        )
        self.assertEqual(budget["target_pixels"] % (16 * 2) ** 2, 0)
        self.assertEqual(budget["source_pixels"], 1024 * 768)
        self.assertLessEqual(budget["target_pixels"], budget["source_pixels"])
        self.assertGreaterEqual(budget["target_keep_ratio"], 0.49)
        self.assertLessEqual(budget["target_keep_ratio"], 0.51)

    def test_visual_token_budget_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            visual_token_budget(
                image_width=32,
                image_height=32,
                keep_ratio=0,
                min_pixels=1,
                max_pixels=4096,
            )

    def test_score_runtime_round_trips_and_uses_shared_keep_count(self):
        runtime = {
            "keep_ratio": 0.5,
            "selection": {
                "score_dtype": "float32",
                "coverage_grid": {"rows": 6, "columns": 6},
            },
        }
        with patch.dict("os.environ", {}, clear=False):
            configure_runtime(runtime)
            self.assertEqual(configured_runtime(), runtime)
            self.assertEqual(
                image_keep_count(rows=12, columns=20, runtime=runtime), 120
            )
            configure_runtime(None)
            self.assertIsNone(configured_runtime())
            self.assertNotIn(TOKEN_RUNTIME_ENV, os.environ)

    def test_vllm_selector_is_the_shared_score_selector(self):
        runtime = {
            "keep_ratio": 0.5,
            "selection": {
                "score_dtype": "float32",
                "coverage_grid": {"rows": 2, "columns": 2},
            },
        }
        features = torch.arange(12 * 20 * 4, dtype=torch.float32).reshape(
            12 * 20, 4
        )
        expected = select_indices(
            features,
            12,
            20,
            runtime["keep_ratio"],
            runtime["selection"],
        )
        actual = select_image_indices(
            features,
            rows=12,
            columns=20,
            runtime=runtime,
        )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
