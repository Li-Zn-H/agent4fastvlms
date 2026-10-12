"""Diagnostic real GPTQ export and GPU reload, never a search candidate.

Run export/native with the isolated quantizer and runtime with the isolated
vLLM interpreter. Native 384-pixel processing, MLA head geometry, routed/shared
branches and the original tokenizer are retained in a small random model.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 4, "columns": 4}}


def descriptor(root):
    value = json.loads((ROOT / "configs/model-deepseek-vl2.json").read_text())
    return {**value, "path": str(root / "artifact"), "device": "cuda:0"}


def native_reference(model, processor, image, question, ratio):
    import torch
    from adapters.deepseek_vl2_dense import prepare_image
    from skills.deepseek_vl2_token import DeepSeekTokenController
    inputs = prepare_image(processor, image, question).to("cuda:0")
    options = dict(max_new_tokens=4, do_sample=False, use_cache=True,
                   eos_token_id=None, pad_token_id=processor.tokenizer.eos_token_id,
                   return_dict_in_generate=False)
    controller = DeepSeekTokenController(model, SELECTION, ratio)
    with torch.inference_mode():
        result = controller.generate(inputs, options)
        embeddings = model.prepare_inputs_embeds(**prepare_image(processor, image, question).to("cuda:0"))
        # Keep logits from the full native prompt as an independent numeric check.
        full = prepare_image(processor, image, question).to("cuda:0")
        logits = model.language(inputs_embeds=embeddings, attention_mask=full["attention_mask"],
                                use_cache=False).logits[0, -1].float().cpu()
    return {"ids": result.cpu(), "logits": logits, "metrics": controller.last_metrics}


def export(root):
    import torch
    import gptqmodel  # Import before allocating the diagnostic model.
    from adapters.deepseek_vl2_dense import native_model_class, load_processor
    from skills.deepseek_vl2_artifact import write_parent, load_physical_model, QUANT_METHOD
    from skills.deepseek_vl2_pruning import dimensions, units, make_recipe, apply_recipe
    from skills.deepseek_vl2_quantization import export_multimodal_gptq
    from deepseek_vllm_fixture import picture
    from test_deepseek_quantization import header
    source = json.loads((ROOT / "configs/model-deepseek-vl2.json").read_text())
    cls = native_model_class(source)
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
    small = header()
    small["vision_config"]["image_size"] = 384
    small["language_config"].update(vocab_size=129280, max_position_embeddings=4096,
        kv_lora_rank=512, qk_nope_head_dim=128, v_head_dim=128,
        n_routed_experts=4, num_experts_per_tok=2)
    torch.manual_seed(427)
    model = cls(DeepseekVLV2Config(**small)).to(device="cuda:0", dtype=torch.bfloat16).eval()
    model.language.model.layers[1].mlp.gate.e_score_correction_bias.data = torch.zeros(4, device="cuda:0")
    processor = load_processor(source)
    dims = dimensions(model.config.language_config)
    scores = [[[float(i) for i in range(u["original_width"])] for _ in range(u["experts"])]
              for u in units(dims)]
    recipe = make_recipe(scores, [128, 128, 384], dims,
                         {"channel_group_size": 128, "protected_layers": []})
    apply_recipe(model, recipe)
    parent = root / "parent"
    write_parent(model, processor, recipe, parent)
    del model
    model = load_physical_model({**source, "path": str(parent)})
    rows = []
    for index, size in enumerate(((384, 384), (768, 384))):
        picture(size).save(root / f"image-{index}.png")
        rows.append({"stable_row_id": f"diagnostic-{index}", "sample_id": f"diagnostic-{index}",
                     "primary_category": "diagnostic", "logical_question": "Describe the image.",
                     "media": {"logical_path": f"image-{index}.png"}})
    (root / "calibration.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    calibration = {"data": str(root / "calibration.jsonl"), "media_root": str(root), "limit": len(rows)}
    candidate = {"route": "gptqmodel_w4a16", "bits": 4, "group_size": 128,
                 "vision_quantization": QUANT_METHOD, "calibration": calibration}
    names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    export_multimodal_gptq(candidate=candidate, model={**source, "path": str(parent)},
        processor=processor, calibration=calibration, output_dir=root / "artifact",
        module_names=names, loaded_model=model)
    from PIL import Image
    references = []
    for row in rows:
        with Image.open(root / row["media"]["logical_path"]) as image:
            for ratio in (1., .5):
                references.append({"image": row["media"]["logical_path"], "question": row["logical_question"],
                    "ratio": ratio, **native_reference(model, processor, image, row["logical_question"], ratio)})
    torch.save(references, root / "reference.pt")
    print(json.dumps({"status": "exported", "scope": "small random diagnostic model, not target quality",
                      "cases": len(references), "physical_widths": recipe["widths"]}), flush=True)


def native(root):
    import torch
    from PIL import Image
    from skills.deepseek_vl2_artifact import load_artifact
    model, processor = load_artifact(descriptor(root))
    references = torch.load(root / "reference.pt", map_location="cpu", weights_only=True)
    errors, failures = [], []
    for expected in references:
        with Image.open(root / expected["image"]) as image:
            actual = native_reference(model, processor, image, expected["question"], expected["ratio"])
        record = {"image": expected["image"], "ratio": expected["ratio"],
                  "max_abs_error": float((actual["logits"] - expected["logits"]).abs().max()),
                  "actual_ids": actual["ids"].tolist(), "expected_ids": expected["ids"].tolist()}
        for label, check in (
            ("logits", lambda: torch.testing.assert_close(actual["logits"], expected["logits"], rtol=.02, atol=.02)),
            ("greedy_ids", lambda: torch.testing.assert_close(actual["ids"], expected["ids"], rtol=0, atol=0)),
            ("cache_metrics", lambda: torch.testing.assert_close(actual["metrics"], expected["metrics"], rtol=0, atol=0))):
            try:
                check()
                record[label] = "passed"
            except AssertionError as error:
                record[label] = str(error)
                failures.append({"image": expected["image"], "ratio": expected["ratio"], "check": label})
        errors.append(record)
    receipt = {"status": "failed" if failures else "passed", "scope": "small real GPTQ physical MoE native CUDA Marlin reload",
               "cases": len(references), "logit_max_abs_errors": errors,
               "quantized_modules": len(model._agent4vlm_quantized_modules), "failures": failures}
    (root / "native-reload.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if failures:
        raise AssertionError(f"Native GPU reload comparison failed: {failures}")


def runtime(root):
    import torch
    from PIL import Image
    from adapters.deepseek_vl2_dense import load_processor, prepare_image
    from search.deepseek_serving import DeepSeekVllmBackend
    config = {**descriptor(root), "stack_adapter": True, "execution_backend": "vllm",
              "token_keep_ratio": .5, "token_selection": SELECTION,
              "runtime": {"vllm_max_model_len": 4096, "vllm_gpu_memory_utilization": .2,
                          "optimization_level": 0, "compile_mm_encoder": False,
                          "cudagraph_mm_encoder": False, "enable_prefix_caching": False,
                          "kv_cache_dtype": "auto"}}
    processor = load_processor(config)
    backend = DeepSeekVllmBackend(config, root / "runtime-all-cases-token.jsonl")
    references = torch.load(root / "reference.pt", map_location="cpu", weights_only=True)
    records = []
    for expected in references:
        if expected["ratio"] != .5:
            continue
        with Image.open(root / expected["image"]) as image:
            inputs = prepare_image(processor, image, expected["question"])
            generated, receipt = backend.generate(image, expected["question"], inputs, processor,
                {"temperature": 0., "max_tokens": 4, "ignore_eos": True, "detokenize": False})
        records.append({"generated": generated, "native_generated": expected["ids"][0].tolist(),
                        "mechanism": receipt})
    equal = all(row["generated"] == row["native_generated"] for row in records)
    result = {"status": "passed" if equal else "greedy_comparison_failed",
              "mechanisms": "passed", "scope": "small real GPTQ physical MoE vLLM engine and Token",
              "runtime": backend.runtime, "cases": records}
    (root / "runtime-reload.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "mechanisms": "passed", "cases": len(records)}), flush=True)
    if not equal:
        raise AssertionError("vLLM diagnostic greedy IDs differ; all mechanism receipts preserved")


def numerics(root):
    """Diagnose greedy drift at identical decode prefixes and selected positions."""
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from adapters.deepseek_vl2_dense import prepare_image
    from skills.deepseek_vl2_artifact import load_artifact, load_physical_model
    from skills.deepseek_vl2_quantization import pad_vision, unpack_gptq
    from skills.deepseek_vl2_token import select_image_indices
    model, processor = load_artifact(descriptor(root))
    reference = load_physical_model({**descriptor(root), "path": str(root / "parent")})
    pad_vision(reference)
    state = {}
    for shard in (root / "artifact").glob("*.safetensors"):
        state.update(load_file(shard))
    manifest = json.loads((root / "artifact/manifest.json").read_text())
    with torch.no_grad():
        for name, precision in manifest["module_assignment"].items():
            if precision == "int4_w4a16":
                packed = {key: state[name + "." + key] for key in ("qweight", "qzeros", "scales", "g_idx")}
                reference.get_submodule(name).weight.copy_(unpack_gptq(packed))

    def replay(native, image, row, reference_positions=None):
        batch = prepare_image(processor, image, row["question"]).to("cuda:0")
        embeddings = native.prepare_inputs_embeds(**batch)
        locations = batch["images_seq_mask"][0].nonzero().flatten()
        nx, ny = batch["images_spatial_crop"][0, 0].tolist()
        selected = select_image_indices(embeddings[0, locations], 14, nx, ny, row["ratio"], SELECTION)
        keep = ~batch["images_seq_mask"][0].clone()
        keep[locations[selected if reference_positions is None else reference_positions.to("cuda:0")]] = True
        embeddings = embeddings[:, keep]
        mask = batch["attention_mask"][:, keep]
        output = native.language(inputs_embeds=embeddings, attention_mask=mask, use_cache=True)
        logits = [output.logits[0, -1].float().cpu()]
        for token in row["ids"][0, :-1]:
            mask = torch.cat((mask, mask.new_ones(1, 1)), dim=1)
            output = native.language(input_ids=token.reshape(1, 1).to("cuda:0"), attention_mask=mask,
                                     past_key_values=output.past_key_values, use_cache=True)
            logits.append(output.logits[0, -1].float().cpu())
        return torch.stack(logits), selected.cpu()

    records = []
    with torch.inference_mode():
        for row in torch.load(root / "reference.pt", map_location="cpu", weights_only=True):
            with Image.open(root / row["image"]) as image:
                expected, expected_positions = replay(reference, image, row)
                actual, actual_positions = replay(model, image, row, expected_positions)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
            values, indices = expected.topk(2, dim=-1)
            records.append({"image": row["image"], "ratio": row["ratio"],
                "logit_max_abs_error_by_step": (actual - expected).abs().amax(-1).tolist(),
                "reference_top2_ids": indices.tolist(), "reference_top2_margins": (values[:, 0] - values[:, 1]).tolist(),
                "reference_argmax": expected.argmax(-1).tolist(), "marlin_argmax": actual.argmax(-1).tolist(),
                "natural_selected_positions_equal": torch.equal(actual_positions, expected_positions),
                "natural_selection_overlap": len(set(actual_positions.tolist()) & set(expected_positions.tolist())) / len(expected_positions)})
    result = {"status": "passed", "scope": "fixed-prefix native Marlin logits using reference-selected positions; natural selection drift is reported separately, not greedy equality or quality",
              "tolerance": {"atol": .02, "rtol": .02}, "cases": records}
    (root / "native-numerics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "native", "runtime", "numerics"))
    parser.add_argument("path")
    args = parser.parse_args()
    # Match the isolated Worker launch: JIT must find this environment's ninja.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires an actual GPU")
    root = Path(args.path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    globals()[args.mode](root)


if __name__ == "__main__":
    main()
