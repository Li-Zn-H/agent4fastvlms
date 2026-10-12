"""Select native DeepSeek tiles while retaining all row/view separators."""

from contextlib import nullcontext
from math import ceil


def image_layout(side, nx, ny):
    """Return each tile's raster indices in the native 2D packed stream."""
    if any(type(x) is not int or x < 1 for x in (side, nx, ny)):
        raise ValueError("Native image layout needs positive integer dimensions")
    stride = side + 1
    tiles = [[y * stride + x for y in range(side) for x in range(side)]]
    protected = [y * stride + side for y in range(side)]
    offset = side * stride
    protected.append(offset)
    offset += 1
    stride = nx * side + 1
    for ty in range(ny):
        for tx in range(nx):
            tiles.append([offset + (ty * side + y) * stride + tx * side + x
                          for y in range(side) for x in range(side)])
    protected.extend(offset + y * stride + nx * side for y in range(ny * side))
    return tiles, protected, offset + ny * side * stride


def token_geometry(processor, inputs):
    import torch
    native = processor.native
    side = ceil((native.image_size // native.patch_size) / native.downsample_ratio)
    crop = inputs["images_spatial_crop"]
    if tuple(crop.shape) != (1, 1, 2) or inputs["input_ids"].shape[0] != 1:
        raise ValueError("DeepSeek-VL2 Token requires exactly one native image")
    nx, ny = (int(x) for x in crop[0, 0].tolist())
    tiles, protected, count = image_layout(side, nx, ny)
    if tuple(inputs["images"].shape) != (1, len(tiles), 3, native.image_size, native.image_size):
        raise ValueError("Native DeepSeek tile geometry changed")
    mask = inputs["images_seq_mask"]
    if mask.dtype != torch.bool or mask.shape != inputs["input_ids"].shape or int(mask.sum()) != count:
        raise ValueError("Native DeepSeek image mask and tile geometry disagree")
    if not torch.equal(mask, inputs["input_ids"] == native.image_token_id):
        raise ValueError("DeepSeek image mask and placeholders disagree")
    return {"rows": side, "columns": side, "repeats": len(tiles),
            "protected_tokens": len(protected), "height": native.image_size,
            "width": native.image_size, "patches": len(tiles)}


def select_image_indices(features, side, nx, ny, ratio, selection):
    import torch
    from skills.vision_token import select_indices
    tiles, protected, count = image_layout(side, nx, ny)
    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError("DeepSeek visual stream differs from its native tile layout")
    selected = [torch.tensor(protected, dtype=torch.long, device=features.device)]
    for tile in tiles:
        positions = torch.tensor(tile, dtype=torch.long, device=features.device)
        keep = select_indices(features.index_select(0, positions), side, side, ratio, selection)
        selected.append(positions.index_select(0, keep))
    return torch.cat(selected).sort().values


class DeepSeekTokenController:
    def __init__(self, model, selection, keep_ratio=1.0):
        from skills.vision_token import validate_selection
        validate_selection(selection)
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.model, self.selection, self.keep_ratio = model, dict(selection), keep_ratio
        self.last_metrics = None

    def generate(self, inputs, options, *, forward_context=None):
        import torch
        self.last_metrics = None
        context = forward_context if forward_context is not None else nullcontext()
        if self.keep_ratio == 1:
            with context:
                return self.model.generate(**inputs, **options)
        if options.get("do_sample") or options.get("use_cache", True) is not True or options.get("num_beams", 1) != 1:
            raise ValueError("DeepSeek Token requires greedy cached single-beam generation")
        ids, mask = inputs["input_ids"], inputs["images_seq_mask"]
        if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape or mask.dtype != torch.bool:
            raise ValueError("DeepSeek Token requires one native image sequence")
        if tuple(inputs["images_spatial_crop"].shape) != (1, 1, 2):
            raise ValueError("DeepSeek Token requires a single image crop layout")
        if inputs["attention_mask"].shape != ids.shape or not bool(inputs["attention_mask"].all()):
            raise ValueError("DeepSeek Token requires an unpadded single prompt")
        if ids.shape[1] + options.get("max_new_tokens", 0) > self.model.config.language_config.max_position_embeddings:
            raise ValueError("Original DeepSeek input exceeds its native context")
        locations = mask[0].nonzero().flatten()
        if not locations.numel() or int(locations[-1]) - int(locations[0]) + 1 != locations.numel():
            raise ValueError("DeepSeek Token requires one contiguous image run")
        vision = self.model.config.vision_config
        side = ceil((vision.image_size // vision.patch_size) / self.model.config.projector_config.downsample_ratio)
        nx, ny = (int(x) for x in inputs["images_spatial_crop"][0, 0].tolist())
        observed = {"forwards": 0, "kv_length": None}

        def cache(module, args, kwargs, output):
            value = getattr(output, "past_key_values", None)
            if value is None:
                raise RuntimeError("DeepSeek Token requires an observed KV cache")
            observed["forwards"] += 1
            observed["kv_length"] = int(value.get_seq_length() if hasattr(value, "get_seq_length") else value[0][0].shape[-2])

        handle = self.model.language.model.register_forward_hook(cache, with_kwargs=True)
        try:
            with context:
                embeddings = self.model.prepare_inputs_embeds(**inputs)
                features = embeddings[0].index_select(0, locations)
                selected = select_image_indices(features, side, nx, ny, self.keep_ratio, self.selection)
                keep = ~mask[0].clone()
                keep[locations.index_select(0, selected)] = True
                positions = keep.nonzero().flatten()
                compacted = {**inputs, **{key: inputs[key].index_select(1, positions)
                                         for key in ("input_ids", "attention_mask", "images_seq_mask")}}
                result = self.model.language.generate(
                    inputs_embeds=embeddings.index_select(1, positions),
                    attention_mask=compacted["attention_mask"], **{**options, "use_cache": True})
            length = int(positions.numel())
            steps = observed["forwards"] - 1
            if steps < 0 or observed["kv_length"] != length + steps:
                raise RuntimeError("DeepSeek KV cache did not match the compacted prompt")
            protected = len(image_layout(side, nx, ny)[1])
            self.last_metrics = {
                "requested_keep_ratio": self.keep_ratio,
                "image_tokens_before": int(locations.numel()), "image_tokens_after": int(selected.numel()),
                "actual_keep_ratio": int(selected.numel()) / int(locations.numel()),
                "protected_layout_tokens": protected, "tiles_selected": 1 + nx * ny,
                "decoder_tokens_before": int(ids.shape[1]), "decoder_tokens_after": length,
                "decode_steps": steps, "output_tokens": int(result.shape[1]),
                "kv_tokens_after_last_forward": observed["kv_length"],
            }
            inputs.update(compacted)
            return result
        finally:
            handle.remove()
