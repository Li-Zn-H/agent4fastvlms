"""Bridge removed Transformers 4 APIs only in Kimi's isolated modern Worker."""

from contextlib import contextmanager
import sys

from adapters.deepseek_vl2_compat import LegacyMLACache


@contextmanager
def legacy_imports():
    import torch
    import transformers.utils.import_utils as imports

    missing = not hasattr(imports, "is_torch_fx_available")
    if missing:
        imports.is_torch_fx_available = lambda: hasattr(torch, "fx")
    try:
        yield
    finally:
        if missing:
            del imports.is_torch_fx_available


def modern_model_class(native):
    """Keep native forwards, legacy tuple cache, positional IDs and image lifecycle."""
    import torch

    module = sys.modules[native.__module__]
    if hasattr(module, "_agent4vlm_kimi_modern_class"):
        return module._agent4vlm_kimi_modern_class
    module.Cache = module.DynamicCache = LegacyMLACache
    prepare_decoder = module.DeepseekV3ForCausalLM.prepare_inputs_for_generation

    def prepare_language(self, input_ids, past_key_values=None, attention_mask=None,
                         inputs_embeds=None, **kwargs):
        supplied = kwargs.pop("position_ids", None)
        result = prepare_decoder(self, input_ids, past_key_values=past_key_values,
                                 attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        if supplied is not None and not torch.equal(
                supplied[:, -result["position_ids"].shape[-1]:], result["position_ids"]):
            raise ValueError("Kimi native position IDs differ from modern generation")
        return result

    module.DeepseekV3ForCausalLM.prepare_inputs_for_generation = prepare_language
    module.DeepseekV3ForCausalLM._supports_default_dynamic_cache = classmethod(lambda cls: False)

    class KimiModernGeneration(native):
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return False

        def tie_weights(self, **kwargs):
            return self.language_model.tie_weights(**kwargs)

        def __init__(self, config):
            for section in (config, config.text_config, config.vision_config):
                for name, value in (("output_attentions", False), ("output_hidden_states", False),
                                    ("return_dict", True), ("use_cache", True)):
                    if not hasattr(section, name):
                        setattr(section, name, value)
            config._attn_implementation = config.text_config._attn_implementation = "eager"
            config.vision_config._attn_implementation = "eager"
            rope = config.text_config.rope_scaling
            if isinstance(rope, dict) and "type" not in rope:
                kind = rope.get("rope_type")
                if kind == "default":
                    config.text_config.rope_scaling = None
                elif kind in ("linear", "dynamic", "yarn"):
                    config.text_config.rope_scaling = {**rope, "type": kind}
                else:
                    raise ValueError("Unsupported Kimi native rotary scaling")
            super().__init__(config)

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                          inputs_embeds=None, pixel_values=None,
                                          attention_mask=None, image_grid_hws=None,
                                          cache_position=None, **kwargs):
            if cache_position is None:
                past = (past_key_values.get_seq_length() if isinstance(past_key_values, LegacyMLACache)
                        else past_key_values[0][0].shape[-2] if past_key_values else 0)
                cache_position = torch.tensor([past], device=input_ids.device)
            return super().prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values, attention_mask=attention_mask, image_grid_hws=image_grid_hws,
                cache_position=cache_position, **kwargs)

    module._agent4vlm_kimi_modern_class = KimiModernGeneration
    return KimiModernGeneration


def restore_native_rotary_buffers(model):
    """Rebuild native derived buffers after Transformers 5's meta-based loading.

    Kimi's custom weight initializer does not initialize nonpersistent rotary
    buffers. The modern loader materializes them with uninitialized storage.
    Reuse the original constructor and scaling code instead of inventing tables
    or trusting historical checkpoint buffers from a different rotary layout.
    """
    for layer in model.language_model.model.layers:
        attention = layer.self_attn
        device = next(attention.parameters()).device
        attention._init_rope()
        attention.rotary_emb.to(device)
