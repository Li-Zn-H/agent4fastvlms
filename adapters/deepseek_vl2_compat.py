"""Explicit Transformers 5 compatibility for the isolated GPTQ Worker.

The original Transformers 4 Dense environment never enters this module. Keep
native eager MLA/MoE forward code and its append-only latent cache semantics;
only bridge removed imports and generation lifecycle APIs.
"""

from contextlib import contextmanager


@contextmanager
def legacy_imports():
    import torch
    import transformers.models.llama.modeling_llama as llama
    import transformers.utils.import_utils as imports
    import transformers.modeling_utils as modeling
    from transformers.utils import is_flash_attn_2_available

    class UnsupportedLegacyMHA(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise ValueError('DeepSeek compatibility supports native eager MLA only')

    additions = [(llama, 'LlamaFlashAttention2', UnsupportedLegacyMHA),
                 (modeling, 'is_flash_attn_2_available', is_flash_attn_2_available),
                 (imports, 'is_torch_fx_available', lambda: hasattr(torch, 'fx'))]
    added = []
    try:
        for module, name, value in additions:
            if not hasattr(module, name):
                setattr(module, name, value)
                added.append((module, name))
        yield
    finally:
        for module, name in reversed(added):
            delattr(module, name)


class LegacyMLACache:
    """Unbounded append-only cache matching the upstream Transformers 4 API."""

    def __init__(self):
        self.key_cache = []
        self.value_cache = []
        self.seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_max_length(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        import torch
        if layer_idx > len(self.key_cache):
            raise ValueError('Native MLA cache layers must arrive in order')
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        for index, (key, value) in enumerate(past_key_values or ()):
            cache.update(key, value, index)
        return cache

    def to_legacy_cache(self):
        return tuple(zip(self.key_cache, self.value_cache))


def bind_native_generation():
    import deepseek_vl2.models.modeling_deepseek as decoder
    import deepseek_vl2.models.modeling_deepseek_vl_v2 as composite
    from transformers.generation import GenerationMixin
    if getattr(composite.DeepseekV2ForCausalLM, '_agent4vlm_legacy_mla', False):
        return
    native = decoder.DeepseekV2ForCausalLM

    class DeepSeekLegacyGeneration(native, GenerationMixin):
        _agent4vlm_legacy_mla = True

        def __init__(self, config):
            config._attn_implementation = 'eager'
            rope = config.rope_scaling
            if isinstance(rope, dict) and 'type' not in rope:
                if rope.get('rope_type') == 'default':
                    config.rope_scaling = None
                elif rope.get('rope_type') in ('linear', 'dynamic', 'yarn'):
                    config.rope_scaling = {**rope, 'type': rope['rope_type']}
                else:
                    raise ValueError('Unsupported native DeepSeek rotary scaling')
            if not config.use_mla:
                raise ValueError('DeepSeek quantization compatibility requires MLA')
            for name, value in (('output_attentions', False), ('output_hidden_states', False),
                                ('return_dict', True), ('use_cache', True)):
                if not hasattr(config, name):
                    setattr(config, name, value)
            super().__init__(config)

        @classmethod
        def _supports_default_dynamic_cache(cls):
            return False

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                          attention_mask=None, inputs_embeds=None, **kwargs):
            # Transformers 5 pre-populates a full position_ids tensor. The
            # original method slices positions for cached decode only when it
            # derives them itself; preserve that native derivation and verify
            # the newer API did not request different positional semantics.
            import torch
            supplied = kwargs.pop('position_ids', None)
            result = super().prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, **kwargs)
            if supplied is not None and not torch.equal(
                    supplied[:, -result['position_ids'].shape[-1]:], result['position_ids']):
                raise ValueError('Native DeepSeek position IDs differ from the generation request')
            return result

    decoder.Cache = decoder.DynamicCache = LegacyMLACache
    composite.DeepseekV2ForCausalLM = DeepSeekLegacyGeneration


def finish_model(model):
    from transformers import PreTrainedModel
    for module in reversed(list(model.modules())):
        if isinstance(module, PreTrainedModel) and not hasattr(module, 'all_tied_weights_keys'):
            module.post_init()
