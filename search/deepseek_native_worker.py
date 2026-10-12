"""Native DeepSeek generation using the existing isolated LMMS connection.

LMMS owns image preprocessing, prompt construction and scoring. The worker owns
the native model, its tokenizer's stopping criteria and actual Token/KV checks.
No LMMS or quantizer dependency crosses the interpreter boundary.
"""

import json
from multiprocessing.connection import Connection
import sys

from search.internvl_native_worker import (
    NativeBackend as BaseBackend, PROJECT_ROOT, cpu_value, receive_message, send_message,
)
from skills.deepseek_vl2_pruning import dimensions, modules, units
from skills.quantization import digest

INPUT_KEYS = frozenset({"input_ids", "attention_mask", "images", "images_seq_mask", "images_spatial_crop"})


def observed_topology(model):
    """Inspect each expert's logical dimensions, including packed Marlin linears."""
    dims = dimensions(model.config.language_config)
    widths = []
    for experts in modules(model, dims):
        width = experts[0].gate_proj.out_features
        for expert in experts:
            expected = {"gate_proj": (dims["hidden_size"], width),
                        "up_proj": (dims["hidden_size"], width),
                        "down_proj": (width, dims["hidden_size"])}
            for name, shape in expected.items():
                linear = getattr(expert, name)
                if (linear.in_features, linear.out_features) != shape:
                    raise ValueError("Native DeepSeek expert dimensions disagree within a width unit")
        widths.append(width)
    routers = {name: str(value.dtype) for name, value in model.named_parameters()
               if name.endswith("mlp.gate.e_score_correction_bias")}
    required = {f"language.model.layers.{u['layer']}.mlp.gate.e_score_correction_bias"
                for u in units(dims) if u["kind"] == "routed"}
    if set(routers) != required or any(dtype != "torch.float32" for dtype in routers.values()):
        raise ValueError("Native DeepSeek router correction tensors must remain F32")
    return {"dimensions": dims, "physical_widths": widths, "router_dtypes": routers,
            "quantized_modules": getattr(model, "_agent4vlm_quantized_modules", [])}


def serve(connection, config, *, loader=None):
    try:
        import torch
        from adapters.deepseek_vl2_dense import load_model
        from skills.deepseek_vl2_token import DeepSeekTokenController

        # Load GPTQ dependencies before any weights, as in the normal Worker.
        # Dependency initialization can retain caller frames and their tensors.
        if loader is None:
            header = json.loads((PROJECT_ROOT / config["path"] / "config.json").read_text())
            if header.get("a4fvlm_deepseek_gptq") is not None:
                import gptqmodel  # noqa: F401
        model, processor = (loader or load_model)(config)
        controller = DeepSeekTokenController(model, config["token_selection"], config["token_keep_ratio"])
        send_message(connection, {"status": "ready", "model_digest": digest(config), **observed_topology(model)})
        while True:
            request = receive_message(connection)
            if request is None:
                break
            kwargs = request["kwargs"]
            if not INPUT_KEYS.issubset(kwargs) or "tokenizer" in kwargs:
                raise ValueError("Native DeepSeek requires complete image inputs and its worker-local tokenizer")
            inputs = {key: value.to(model.device) for key, value in kwargs.items() if key in INPUT_KEYS}
            options = {key: value for key, value in kwargs.items() if key not in INPUT_KEYS}
            stops = options.get("stop_strings")
            if stops:
                if not isinstance(stops, list) or any(not isinstance(x, str) or not x for x in stops):
                    raise ValueError("Native stopping sequences must be nonempty strings")
                options["tokenizer"] = processor.tokenizer
            with torch.inference_mode():
                output = controller.generate(inputs, options)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("Native DeepSeek must return one suffix-only token sequence")
            if config["token_keep_ratio"] < 1 and not controller.last_metrics:
                raise RuntimeError("Native DeepSeek produced no Token/KV evidence")
            send_message(connection, {"status": "completed", "index": request["index"],
                                      "output_ids": cpu_value(output), "metrics": controller.last_metrics})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            send_message(connection, {"status": "error", "error": f"{type(error).__name__}: {error}"})
        except (OSError, EOFError):
            pass
        raise
    finally:
        connection.close()


class NativeBackend(BaseBackend):
    worker_module = "search.deepseek_native_worker"
    model_name = "DeepSeek-VL2"

    def validate_ready(self, ready):
        from skills.deepseek_vl2_artifact import read_parent, validate_quantized_bindings
        path = PROJECT_ROOT / self.config["path"]
        header = json.loads((path / "config.json").read_text())
        dims = dimensions(header["language_config"])
        widths = [unit["original_width"] for unit in units(dims)]
        if header.get("a4fvlm_stack_widths") is not None:
            widths = read_parent(path)[1]["widths"]
        if ready.get("dimensions") != dims or ready.get("physical_widths") != widths:
            raise RuntimeError("Native DeepSeek worker topology differs from the selected Parent")
        routers = {f"language.model.layers.{u['layer']}.mlp.gate.e_score_correction_bias": "torch.float32"
                   for u in units(dims) if u["kind"] == "routed"}
        if ready.get("router_dtypes") != routers:
            raise RuntimeError("Native DeepSeek worker changed the native F32 routers")
        required = []
        if header.get("a4fvlm_deepseek_gptq") is not None:
            manifest = json.loads((path / "manifest.json").read_text())
            validate_quantized_bindings(path, header, manifest)
            required = sorted(name for name, precision in manifest["module_assignment"].items()
                              if precision == "int4_w4a16")
        if sorted(ready.get("quantized_modules", [])) != required:
            raise RuntimeError("Native DeepSeek worker did not load all declared quantized kernels")
        if self.config.get("stack_adapter") and (not required or header.get("a4fvlm_stack_widths") is None):
            raise RuntimeError("Native DeepSeek Stack requires a freshly quantized physical Parent")


def create_native_proxy(config):
    """Weightless LMMS model; the isolated worker owns all weight allocation."""
    import torch
    from adapters.deepseek_vl2_dense import native_classes

    class NativeProxy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            native_class, _ = native_classes(config)
            self.config = native_class.config_class.from_pretrained(
                str((PROJECT_ROOT / config["path"]).resolve()), local_files_only=True)
            self.backend = NativeBackend(config)

        def generate(self, **kwargs):
            return self.backend.generate(**kwargs)

    return NativeProxy()


def main():
    serve(Connection(int(sys.argv[1])), json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
