"""Shared native/vLLM selection for LLaVA's packed and fixed-grid features."""


def clip_grid(count):
    """LLaVA-1.5 selects the native CLIP grid after removing its CLS token."""
    if count != 576:
        raise ValueError("LLaVA-1.5 selection requires the native 24x24 CLIP grid")
    return 24, 24


def select_clip_indices(features, selection, keep_ratio):
    from skills.vision_token import select_indices

    rows, columns = clip_grid(int(features.shape[0]))
    return select_indices(features, rows, columns, keep_ratio, selection)


def select_packed_indices(features, selection, keep_ratio):
    import torch
    from skills.vision_token import keep_count

    count = int(features.shape[0])
    keep = keep_count(1, count, keep_ratio, selection["coverage_grid"])
    scores = torch.linalg.vector_norm(features.to(getattr(torch, selection["score_dtype"])), dim=-1)
    torch._assert_async(torch.isfinite(scores).all(), "Nonfinite visual importance scores")
    return torch.argsort(scores, descending=True, stable=True)[:keep].sort().values
