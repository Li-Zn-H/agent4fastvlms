"""Adapter-dispatched loading, input processing and local generation."""

from pathlib import Path
from time import perf_counter

from adapters import AdapterError, models, select_model
from search.context import model_adapter, require_stages

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def __getattr__(name):
    # Compatibility for callers of the old Qwen-specific timer; dispatch below
    # uses each model's own registered timer instead.
    if name == "GenerationTimer":
        return models.get("qwen3_vl").resolve("generation_timer")
    raise AttributeError(name)


def load_model(config, *, structured_pruning=None):
    require_stages(config, ("dense", "sp") if structured_pruning is not None else ("dense",),
                   project_root=PROJECT_ROOT)
    adapter = model_adapter(config, project_root=PROJECT_ROOT)
    apply = adapter.resolve("sp_apply") if structured_pruning is not None else None
    model, processor = adapter.resolve("dense_load")(config)
    processor._agent4vlm_adapter_id = adapter.id
    if apply is not None:
        apply(model, structured_pruning)
    return model, processor


def load_processor(config):
    require_stages(config, ("dense",), project_root=PROJECT_ROOT)
    adapter = model_adapter(config, project_root=PROJECT_ROOT)
    processor = adapter.resolve("processor_load")(config)
    processor._agent4vlm_adapter_id = adapter.id
    return processor


def processor_adapter(processor, model=None):
    registered = getattr(processor, "_agent4vlm_adapter_id", None)
    if registered is not None:
        return models.get(registered)
    if model is not None:
        config = model.config.to_dict()
        return select_model(config)
    raise AdapterError("processor has no model adapter; use load_processor/load_model")


def prepare_inputs(processor, sample, media_root):
    return processor_adapter(processor).resolve("prepare_inputs")(processor, sample, media_root)


def generate_outputs(
    model, processor, samples, *, media_root, generation, measure=False
):
    """Generate responses using the supplied model, including optimized candidates.

    Paths are relative to the project root. Sampling and length come only from
    evaluation.json; responses exclude the input prompt and special tokens.
    """
    import torch

    adapter = processor_adapter(processor, model)
    prepare = adapter.resolve("prepare_inputs")
    timer_factory = adapter.resolve("generation_timer") if measure else None
    count_images = adapter.resolve("image_token_count") if measure else None
    media_root = PROJECT_ROOT / media_root
    options = {
        "max_new_tokens": generation["max_new_tokens"],
        "do_sample": generation["do_sample"],
        "return_dict_in_generate": False,
    }
    # ignore_eos is our config field, not a Transformers generate() argument.
    if generation["ignore_eos"]:
        options.update(eos_token_id=None, forced_eos_token_id=None)

    outputs = []
    model.eval()
    if measure and model.device.type != "cuda":
        raise ValueError("Performance measurement requires a CUDA model")
    with torch.inference_mode():
        for sample in samples:
            started = generation_started = generation_seconds = 0.0
            input_tokens = image_tokens = 0
            if measure:
                torch.cuda.synchronize(model.device)
                torch.cuda.reset_peak_memory_stats(model.device)
                started = perf_counter()
            inputs = prepare(processor, sample, media_root)
            if measure:
                input_tokens = inputs["input_ids"].shape[1]
                image_tokens = count_images(model, inputs)
            inputs = inputs.to(model.device)
            if measure:
                torch.cuda.synchronize(model.device)
                generation_started = perf_counter()
            controller = getattr(model, "vision_token_controller", None)
            timer = (
                timer_factory(
                    model, pruning=controller is not None and controller.keep_ratio < 1
                )
                if measure
                else None
            )
            if controller is not None:
                generated = controller.generate(inputs, options, forward_context=timer)
            elif timer is not None:
                with timer:
                    generated = model.generate(**inputs, **options)
            else:
                generated = model.generate(**inputs, **options)
            if measure:
                torch.cuda.synchronize(model.device)
                generation_seconds = perf_counter() - generation_started
            answer_ids = generated[:, inputs["input_ids"].shape[1] :]
            text = processor.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            output = {"sample_id": sample["stable_row_id"], "output_text": text}
            if measure:
                torch.cuda.synchronize(model.device)
                elapsed = perf_counter() - started
                output["performance"] = {
                    "end_to_end_seconds": elapsed,
                    "generate_seconds": generation_seconds,
                    "input_tokens": input_tokens,
                    "image_tokens": image_tokens,
                    "output_tokens": answer_ids.shape[1],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        model.device
                    ),
                }
            if measure:
                assert timer is not None
                output["performance"].update(timer.metrics())
            if controller is not None and controller.last_metrics is not None:
                output["vision_token"] = dict(controller.last_metrics)
                if measure:
                    output["performance"].update(controller.last_metrics)
            outputs.append(output)
            del inputs, generated, answer_ids
    return outputs
