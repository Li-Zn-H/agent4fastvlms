"""Scoped vLLM 0.26 packed-kv_b dtype repair; preserve native MLA computation."""

_PREFILL = None


def bind_packed_prefill(implementation):
    """Repair the installed prefill's stale weight access on this Kimi instance.

    Upstream already computes a dtype fallback for GPTQ's missing ``weight``,
    then accidentally ignores it in one conversion. Retain the whole installed
    method, changing only that operand. Never create a BF16 projection fallback
    or modify the isolated package. Refuse an unfamiliar implementation.
    """
    import inspect
    import textwrap
    from types import MethodType
    from vllm.model_executor.layers.attention.mla_attention import MLACommonBaseImpl
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
    global _PREFILL
    projection = implementation.kv_b_proj
    if not isinstance(projection.quant_method, AutoGPTQLinearMethod) or hasattr(projection, "weight"):
        raise ValueError("Kimi packed MLA prefill requires its actual GPTQ kv_b projection")
    original = MLACommonBaseImpl._compute_prefill_context
    if getattr(type(implementation), "_compute_prefill_context") is not original:
        raise ValueError("Kimi packed MLA has an unvalidated prefill implementation")
    if _PREFILL is None:
        source = textwrap.dedent(inspect.getsource(original))
        stale = "kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)"
        repaired = "kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)"
        if source.count(stale) != 1 or "else self.kv_b_proj.params_dtype" not in source:
            raise ValueError("Kimi packed MLA compatibility requires the validated vLLM dtype fallback")
        namespace = dict(original.__globals__)
        exec(compile(source.replace(stale, repaired), __file__, "exec"), namespace)
        _PREFILL = namespace[original.__name__]
    implementation._compute_prefill_context = MethodType(_PREFILL, implementation)
