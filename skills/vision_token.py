"""Single-image Qwen3-VL post-encoder selection and cached generation adapter."""

from contextlib import nullcontext
from decimal import Decimal, ROUND_CEILING


def validate_selection(config):
    if config["score_dtype"] not in ("float32", "float16"):
        raise ValueError("score_dtype must be float32 or float16")
    grid = config["coverage_grid"]
    if any(
        type(grid[key]) is not int or not 1 <= grid[key] <= 6
        for key in ("rows", "columns")
    ):
        raise ValueError("Coverage dimensions must be integers between 1 and 6")


def keep_count(rows, columns, ratio, grid):
    if rows < 1 or columns < 1 or not 0 < ratio <= 1:
        raise ValueError("Invalid visual grid or keep ratio")
    count = rows * columns
    requested = int(
        (Decimal(str(ratio)) * count).to_integral_value(rounding=ROUND_CEILING)
    )
    floor = min(rows, grid["rows"]) * min(columns, grid["columns"])
    return min(count, max(requested, floor))


def geometry_keep_count(geometry, ratio, grid):
    """Count selected grid features plus native protected layout tokens."""
    repeats = geometry.get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Geometry repeats must be a positive integer")
    protected = geometry.get("protected_tokens", 0)
    if type(protected) is not int or protected < 0:
        raise ValueError("Protected layout token count must be a nonnegative integer")
    return repeats * keep_count(geometry["rows"], geometry["columns"], ratio, grid) + protected


def select_indices(features, rows, columns, ratio, config):
    """Select coverage representatives, fill by score, then restore spatial order."""
    import torch

    count = rows * columns
    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError("Visual features do not match the merged image grid")
    keep = keep_count(rows, columns, ratio, config["coverage_grid"])
    scores = torch.linalg.vector_norm(
        features.to(getattr(torch, config["score_dtype"])), dim=-1
    )
    # Fail on invalid features without a per-cell device-to-host synchronization.
    torch._assert_async(
        torch.isfinite(scores).all(), "Nonfinite visual importance scores"
    )
    positions = torch.arange(count, device=features.device)
    if keep == count:
        return positions
    grid = config["coverage_grid"]
    nr, nc = min(rows, grid["rows"]), min(columns, grid["columns"])
    cells = ((positions // columns) * nr // rows) * nc + (
        positions % columns
    ) * nc // columns
    ranked = torch.argsort(scores, descending=True, stable=True)
    first = torch.full((nr * nc,), count, device=features.device, dtype=torch.long)
    first.scatter_reduce_(0, cells[ranked], positions, reduce="amin", include_self=True)
    representatives = ranked[first]
    is_representative = torch.zeros(count, device=features.device, dtype=torch.bool)
    is_representative[representatives] = True
    rest = ranked[~is_representative[ranked]][: keep - nr * nc]
    return torch.cat((representatives, rest)).sort().values


def compact_visual(features, deepstack, rows, columns, ratio, config):
    indices = select_indices(features, rows, columns, ratio, config)
    return (
        indices,
        features.index_select(0, indices),
        [x.index_select(0, indices) for x in deepstack],
    )


class VisionTokenController:
    """Compact every decoder call while generation retains the original prompt IDs.

    Transformers 5.14 passes one new token on cached decode and advances the
    original four-channel positions. Only the causal/text channel is renumbered;
    the three rotary channels keep their original coordinates. Its growing
    attention mask is compacted on every call to match the physical KV length.
    Hooks exist only for the duration of one generate call, including failures.
    """

    def __init__(self, model, selection, keep_ratio=1.0):
        validate_selection(selection)
        self.model = model
        self.selection = dict(selection)
        self.keep_ratio = keep_ratio
        self.rows = self.columns = 0
        self.last_metrics = None
        self.reset()

    def reset(self):
        self.kept = None
        self.original_length = None
        self.expected_cache_length = None
        self.last_metrics = None

    def generate(self, inputs, options, *, forward_context=None):
        """Enter external instrumentation after installing compaction hooks."""
        self.reset()
        context = forward_context if forward_context is not None else nullcontext()
        if self.keep_ratio == 1:
            with context:
                return self.model.generate(**inputs, **options)
        if inputs["input_ids"].shape[0] != 1 or "pixel_values_videos" in inputs:
            raise ValueError(
                "Vision token selection supports one image and batch size one"
            )
        grid = inputs["image_grid_thw"]
        if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError("Exactly one still image is required")
        if options.get("do_sample") or options.get("use_cache") is False:
            raise ValueError("Vision token selection requires greedy cached generation")
        merge = self.model.config.vision_config.spatial_merge_size
        self.rows, self.columns = (int(grid[0, 1]) // merge, int(grid[0, 2]) // merge)
        self.original_length = inputs["input_ids"].shape[1]
        self.model.model.rope_deltas = None
        language = self.model.model.language_model
        pre = language.register_forward_pre_hook(self._before, with_kwargs=True)
        post = language.register_forward_hook(self._after, with_kwargs=True)
        try:
            with context:
                result = self.model.generate(**inputs, **options)
            if self.last_metrics is None:
                raise RuntimeError("Vision token prefill was not observed")
            return result
        finally:
            pre.remove()
            post.remove()
            self.kept = None
            self.model.model.rope_deltas = None

    def _before(self, module, args, kwargs):
        import torch

        embeds = kwargs["inputs_embeds"]
        mask = kwargs["attention_mask"]
        positions = kwargs["position_ids"]
        if mask.ndim != 2 or positions.shape[0] != 4 or embeds.shape[0] != 1:
            raise ValueError("Unsupported decoder mask or position layout")
        cache = kwargs.get("past_key_values")
        past = 0 if cache is None else cache.get_seq_length()
        visual = kwargs.get("visual_pos_masks")
        if visual is not None:
            if past or self.kept is not None:
                raise ValueError("Visual selection must occur once on an empty cache")
            locations = visual[0].nonzero().flatten()
            features = embeds[0].index_select(0, locations)
            deepstack = kwargs["deepstack_visual_embeds"]
            if len(deepstack) != len(
                self.model.config.vision_config.deepstack_visual_indexes
            ):
                raise ValueError("DeepStack feature count drift")
            if any(x.shape != features.shape for x in deepstack):
                raise ValueError("DeepStack and main visual features differ")
            indices, _, selected_deepstack = compact_visual(
                features,
                deepstack,
                self.rows,
                self.columns,
                self.keep_ratio,
                self.selection,
            )
            keep = ~visual[0].clone()
            keep[locations[indices]] = True
            self.kept = keep.nonzero().flatten()
            kwargs["inputs_embeds"] = embeds.index_select(1, self.kept)
            kwargs["visual_pos_masks"] = visual.index_select(1, self.kept)
            kwargs["deepstack_visual_embeds"] = selected_deepstack
            positions = positions.index_select(-1, self.kept)
            self.last_metrics = {
                "requested_keep_ratio": self.keep_ratio,
                "image_tokens_before": features.shape[0],
                "image_tokens_after": indices.numel(),
                "actual_keep_ratio": indices.numel() / features.shape[0],
                "decoder_tokens_before": self.original_length,
                "decoder_tokens_after": self.kept.numel(),
                "decode_steps": 0,
            }
        else:
            if self.kept is None or embeds.shape[1] != 1:
                raise ValueError("Expected single-token decode after visual prefill")
            if past != self.expected_cache_length:
                raise ValueError("Decode KV cache length drift")
            assert self.last_metrics is not None
            self.last_metrics["decode_steps"] = (
                int(self.last_metrics["decode_steps"] or 0) + 1
            )
        tail = mask[:, self.original_length :]
        kwargs["attention_mask"] = torch.cat(
            (mask.index_select(1, self.kept), tail), dim=1
        )
        positions = positions.clone()
        length = kwargs["inputs_embeds"].shape[1]
        positions[0] = torch.arange(past, past + length, device=positions.device)
        kwargs["position_ids"] = positions
        self.expected_cache_length = past + length
        if kwargs["attention_mask"].shape[1] != self.expected_cache_length:
            raise ValueError("Compacted mask and KV cache length differ")
        return args, kwargs

    def _after(self, module, args, kwargs, output):
        cache = output.past_key_values
        if cache is None or cache.get_seq_length() != self.expected_cache_length:
            raise ValueError("Decoder did not produce the expected compacted KV cache")
        assert self.last_metrics is not None
        self.last_metrics["kv_tokens_after_last_forward"] = cache.get_seq_length()
