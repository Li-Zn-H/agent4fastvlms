"""Probe real SigLIP FFN padding and GPTQ packing before a full export.

This synthetic-activation probe is an interface check, never quality evidence.
Use the isolated GPTQ Worker interpreter and --execute to allocate the GPU.
"""

import argparse
import json
from pathlib import Path

from skills.llava_quantization import new_gptq_task, pack_gptq_task, padded_width, pad_siglip_mlp


def execute(model, output, device):
    import torch
    from safetensors import safe_open
    from transformers import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipMLP
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

    config = json.loads((model / "config.json").read_text())
    vision_config = SiglipVisionConfig(**config["vision_config"])
    torch.manual_seed(0)
    mlp = SiglipMLP(vision_config).to(device=device, dtype=torch.float16).eval()
    weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "vision_tower.vision_model.encoder.layers.0.mlp."
    state = {}
    for name, shard in weight_map.items():
        if name.startswith(prefix):
            with safe_open(str(model / shard), framework="pt", device="cpu") as handle:
                state[name[len(prefix):]] = handle.get_tensor(name)
    mlp.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 512, vision_config.hidden_size, device=device, dtype=torch.float16)
    with torch.inference_mode():
        dense = mlp(inputs)
        padding = pad_siglip_mlp(mlp, padded_width(vision_config.intermediate_size))
        padded = mlp(inputs)
        torch.testing.assert_close(padded, dense, atol=0.02, rtol=0.01)
        first_inputs = inputs
        second_inputs = mlp.activation_fn(mlp.fc1(inputs))
        reports = []
        for name, activation in (("fc1", first_inputs), ("fc2", second_inputs)):
            layer = getattr(mlp, name)
            task = new_gptq_task(layer)
            task.add_batch(activation, layer(activation))
            tensors, report = pack_gptq_task(task)
            loaded = TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                                 in_features=layer.in_features, out_features=layer.out_features,
                                 bias=layer.bias is not None)
            loaded.load_state_dict(tensors, strict=True)
            loaded.qzero_format(format=1)
            convert_gptq_v1_to_v2_format_module(loaded, bits=4, pack_dtype=torch.int32)
            loaded.post_init()
            loaded = loaded.to(device)
            result = loaded(activation)
            reference = layer(activation)
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite reloaded GPTQ output")
            report.update(module=name, packed_keys=sorted(tensors),
                          reload_output_nmse=float((result.float()-reference.float()).square().mean() /
                                                   reference.float().square().mean()))
            reports.append(report)
            setattr(mlp, name, loaded)
        quantized = mlp(inputs)
        result = {"status": "passed", "scope": "synthetic_activation_interface_probe_only",
                  "model": str(model), "device": device, "padding": padding,
                  "padding_max_abs_error": float((padded-dense).abs().max()),
                  "modules": reports,
                  "mlp_output_nmse": float((quantized.float()-dense.float()).square().mean() /
                                           dense.float().square().mean())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/LLaVA-OneVision-7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "planned", "model": str(args.model), "device": args.device}))
        return
    print(json.dumps(execute(args.model, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
