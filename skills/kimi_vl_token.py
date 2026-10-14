"""Native Kimi merged-grid selection, prompt compaction and cache auditing."""

from contextlib import nullcontext
import hashlib
from types import MethodType

from skills.vision_token import keep_count, select_indices, validate_selection


def token_geometry(processor, inputs):
    grid = inputs['image_grid_hws']
    if tuple(grid.shape) != (1, 2):
        raise ValueError('Kimi Token requires one native image grid')
    mh, mw = processor.image_processor.merge_kernel_size
    height, width = (int(x) for x in grid[0].tolist())
    if height % mh or width % mw:
        raise ValueError('Kimi native patch grid is not aligned to merging')
    rows, columns = height // mh, width // mw
    if int((inputs['input_ids'] == 163605).sum()) != rows * columns:
        raise ValueError('Kimi native placeholders differ from merged-grid features')
    return {'rows': rows, 'columns': columns, 'patch_rows': height, 'patch_columns': width}


class KimiTokenController:
    def __init__(self, model, selection, keep_ratio=1.0):
        validate_selection(selection)
        if not 0 < keep_ratio <= 1:
            raise ValueError('Kimi Token keep ratio must be in (0, 1]')
        self.model = model
        self.selection = dict(selection)
        self.keep_ratio = keep_ratio
        self.last_metrics = None

    def generate(self, inputs, options, *, forward_context=None):
        import torch

        self.last_metrics = None
        context = forward_context if forward_context is not None else nullcontext()
        if self.keep_ratio == 1:
            with context:
                return self.model.generate(**inputs, **options)
        if inputs['input_ids'].shape[0] != 1 or inputs['image_grid_hws'].shape != (1, 2):
            raise ValueError('Kimi Token requires a single image and batch size one')
        if options.get('do_sample') or options.get('use_cache', True) is not True:
            raise ValueError('Kimi Token requires greedy cached generation')
        original_ids = inputs['input_ids']
        if original_ids.shape[1] + options.get('max_new_tokens', 0) > self.model.config.text_config.max_position_embeddings:
            raise ValueError('Kimi original input plus output exceeds native context')
        mh, mw = self.model.config.vision_config.merge_kernel_size
        height, width = (int(x) for x in inputs['image_grid_hws'][0].tolist())
        if height % mh or width % mw:
            raise ValueError('Kimi Token requires a native aligned merged grid')
        rows, columns = height // mh, width // mw
        image_id = self.model.config.media_placeholder_token_id
        locations = (original_ids[0] == image_id).nonzero().flatten()
        before = rows * columns
        if locations.numel() != before or before < 1:
            raise ValueError('Kimi native merged grid and placeholders differ')
        start, end = int(locations[0]), int(locations[-1]) + 1
        if end - start != before:
            raise ValueError('Kimi Token requires one contiguous placeholder run')
        count = keep_count(rows, columns, self.keep_ratio, self.selection['coverage_grid'])
        compacted_ids = torch.cat((original_ids[:, :start], original_ids[:, start:start + count],
                                   original_ids[:, end:]), dim=1)
        compacted = {**inputs, 'input_ids': compacted_ids}
        mask = inputs.get('attention_mask')
        if mask is not None:
            if mask.shape != original_ids.shape or mask.ndim != 2:
                raise ValueError('Kimi Token requires the native 2D attention mask')
            compacted['attention_mask'] = torch.cat((mask[:, :start], mask[:, start:start + count],
                                                     mask[:, end:]), dim=1)
        original_extract = self.model._extract_image_features
        selected = {}
        observed = {'calls': 0, 'kv_length': None}

        def extract(owner, *args, **kwargs):
            if selected:
                raise RuntimeError('Kimi encoded the image more than once during cached generation')
            features = original_extract(*args, **kwargs)
            indices = select_indices(features, rows, columns, self.keep_ratio, self.selection)
            if indices.numel() != count:
                raise RuntimeError('Kimi selected-feature count differs from compacted placeholders')
            selected['indices'] = indices.detach().cpu().tolist()
            return features.index_select(0, indices)

        def observe(module, args, kwargs, output):
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError('Kimi Token requires observed physical KV cache')
            length = int(cache.get_seq_length() if hasattr(cache, 'get_seq_length') else cache[0][0].shape[-2])
            expected = compacted_ids.shape[1] + observed['calls']
            if length != expected:
                raise RuntimeError('Kimi physical KV cache differs from compacted decoder input')
            observed['calls'] += 1
            observed['kv_length'] = length

        self.model._extract_image_features = MethodType(extract, self.model)
        handle = self.model.language_model.model.register_forward_hook(observe, with_kwargs=True)
        try:
            with context:
                result = self.model.generate(**compacted, **{**options, 'use_cache': True})
            if not selected or observed['calls'] < 1:
                raise RuntimeError('Kimi feature selection or cached decoding was not observed')
            inputs.update(compacted)
            self.last_metrics = {
                'requested_keep_ratio': self.keep_ratio,
                'image_tokens_before': before, 'image_tokens_after': count,
                'actual_keep_ratio': count / before,
                'decoder_tokens_before': int(original_ids.shape[1]),
                'decoder_tokens_after': int(compacted_ids.shape[1]),
                'decode_steps': observed['calls'] - 1,
                'output_tokens': int(result.shape[1] - compacted_ids.shape[1]),
                'kv_tokens_after_last_forward': observed['kv_length'],
                'selected_indices_sha256': hashlib.sha256(str(selected['indices']).encode()).hexdigest(),
            }
            return result
        finally:
            handle.remove()
            self.model._extract_image_features = original_extract
