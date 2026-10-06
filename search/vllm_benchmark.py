"""Small, backend-neutral helpers for the vLLM comparison benchmark."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable


def chat_prompt(sample: dict[str, Any], image_path: str) -> list[dict[str, Any]]:
    """Build the OpenAI-compatible multimodal chat shape accepted by vLLM."""

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"},
                },
                {"type": "text", "text": sample["logical_question"]},
            ],
        }
    ]


def visual_token_budget(
    *,
    image_width: int,
    image_height: int,
    keep_ratio: float,
    min_pixels: int,
    max_pixels: int,
    patch_size: int = 16,
    merge_size: int = 2,
) -> dict[str, int | float]:
    """Return a legacy image processor budget for compatibility experiments.

    The score-based vLLM Worker does not call this helper.  It remains available
    for old receipts and isolated processor-budget experiments; resizing pixels
    is not equivalent to selecting post-encoder visual features.
    """

    if (
        type(image_width) is not int
        or type(image_height) is not int
        or image_width < 1
        or image_height < 1
    ):
        raise ValueError("image dimensions must be positive integers")
    if not 0 < float(keep_ratio) <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if (
        type(min_pixels) is not int
        or type(max_pixels) is not int
        or min_pixels < 1
        or max_pixels < min_pixels
    ):
        raise ValueError("pixel bounds are invalid")
    if (
        type(patch_size) is not int
        or type(merge_size) is not int
        or patch_size < 1
        or merge_size < 1
    ):
        raise ValueError("patch and merge sizes must be positive integers")

    source_pixels = min(max(image_width * image_height, min_pixels), max_pixels)
    alignment = (patch_size * merge_size) ** 2
    target_pixels = int(round(source_pixels * float(keep_ratio)))
    target_pixels = max(alignment, target_pixels)
    # Qwen's image processor rounds the resized dimensions to patch/merge
    # boundaries.  Aligning the requested area avoids a silent ratio drift
    # caused solely by an impossible processor target.
    target_pixels = max(alignment, (target_pixels // alignment) * alignment)
    target_pixels = min(max_pixels, max(min_pixels, target_pixels))
    return {
        "source_pixels": source_pixels,
        "target_pixels": target_pixels,
        "requested_keep_ratio": float(keep_ratio),
        "target_keep_ratio": target_pixels / source_pixels,
    }


def percentile(values: Iterable[float], fraction: float) -> float | None:
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_round(
    *,
    elapsed_seconds: float,
    output_tokens: int,
    request_latencies: Iterable[float],
    ttfts: Iterable[float | None],
    request_count: int,
) -> dict[str, Any]:
    """Return compact, comparable metrics for one vLLM benchmark round."""

    latencies = [float(value) for value in request_latencies]
    first_tokens = [float(value) for value in ttfts if value is not None]
    if request_count < 1 or len(latencies) != request_count:
        raise ValueError("request latency count must match request_count")
    if first_tokens and len(first_tokens) != request_count:
        raise ValueError("TTFT must be present for every request or none")
    elapsed = float(elapsed_seconds)
    if elapsed <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return {
        "request_count": request_count,
        "elapsed_seconds": elapsed,
        "output_tokens": int(output_tokens),
        "request_throughput": request_count / elapsed,
        "output_token_throughput": int(output_tokens) / elapsed,
        "mean_request_latency_seconds": fmean(latencies),
        "p50_request_latency_seconds": percentile(latencies, 0.50),
        "p95_request_latency_seconds": percentile(latencies, 0.95),
        "ttft_available": bool(first_tokens),
        "mean_ttft_seconds": fmean(first_tokens) if first_tokens else None,
        "p50_ttft_seconds": percentile(first_tokens, 0.50),
        "p95_ttft_seconds": percentile(first_tokens, 0.95),
    }


def compare_reports(dense: dict[str, Any], quantized: dict[str, Any]) -> dict[str, Any]:
    """Calculate quantized/Dense ratios from two completed reports."""

    dense_round = dense["summary"]
    quant_round = quantized["summary"]
    dense_output = float(dense_round["output_token_throughput"])
    dense_request = float(dense_round["request_throughput"])
    dense_latency = float(dense_round["mean_request_latency_seconds"])
    dense_ttft = dense_round.get("mean_ttft_seconds")
    dense_memory = dense.get("peak_inference_memory_bytes", dense.get("peak_gpu_memory_bytes"))
    quant_memory = quantized.get("peak_inference_memory_bytes", quantized.get("peak_gpu_memory_bytes"))
    dense_engine_memory = dense.get("engine_loaded_memory_bytes")
    quant_engine_memory = quantized.get("engine_loaded_memory_bytes")
    result = {
        "output_token_speedup": float(quant_round["output_token_throughput"])
        / dense_output,
        "request_speedup": float(quant_round["request_throughput"]) / dense_request,
        "latency_ratio": float(quant_round["mean_request_latency_seconds"])
        / dense_latency,
        "ttft_ratio": (
            float(quant_round["mean_ttft_seconds"]) / float(dense_ttft)
            if dense_ttft is not None and quant_round.get("mean_ttft_seconds") is not None
            else None
        ),
    }
    if isinstance(dense_memory, (int, float)) and isinstance(quant_memory, (int, float)):
        result["memory_ratio"] = float(quant_memory) / float(dense_memory)
        result["memory_saving_ratio"] = 1.0 - result["memory_ratio"]
    else:
        result["memory_ratio"] = None
        result["memory_saving_ratio"] = None
    if isinstance(dense_engine_memory, (int, float)) and isinstance(
        quant_engine_memory, (int, float)
    ):
        result["engine_memory_ratio"] = float(quant_engine_memory) / float(
            dense_engine_memory
        )
        result["engine_memory_saving_ratio"] = 1.0 - result["engine_memory_ratio"]
    else:
        result["engine_memory_ratio"] = None
        result["engine_memory_saving_ratio"] = None
    return result
