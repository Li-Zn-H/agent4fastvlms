"""Local native generation boundary for LMMS and the isolated GPTQ runtime.

LMMS retains its own image transform, native chat, decoding and scoring. Only
CPU tensors and generation options cross the private inherited connection;
backend logs continue on stdout/stderr and cannot corrupt the protocol.
"""
import atexit
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys

from skills.quantization import digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = frozenset({"input_ids", "attention_mask", "pixel_values"})


def send_message(connection, value):
    # Connection.send uses ForkingPickler, whose Torch reducers may send shared
    # storage handles authenticated by multiprocessing. Independent interpreters
    # have different auth keys. Plain pickle transfers owned CPU tensor bytes.
    connection.send_bytes(pickle.dumps(value, protocol=5))


def receive_message(connection):
    return pickle.loads(connection.recv_bytes())


def cpu_value(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_value(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"Unsupported native generation value: {type(value).__name__}")


def serve(connection, config, *, loader=None):
    """Load once, fail the branch on any generation error, and release on EOF."""
    try:
        import torch
        from adapters.internvl35_dense import InternVLTokenController, load_model

        model, _ = (loader or load_model)(config)
        controller = InternVLTokenController(model, config["token_selection"], config["token_keep_ratio"])
        send_message(connection, {"status": "ready", "model_digest": digest(config),
                         "physical_widths": [layer.mlp.gate_proj.out_features
                                             for layer in model.language_model.model.layers],
                         "quantized_modules": getattr(model, "_agent4vlm_quantized_modules", [])})
        while True:
            request = receive_message(connection)
            if request is None:
                break
            kwargs = request["kwargs"]
            inputs = {key: value.to(model.device) for key, value in kwargs.items() if key in INPUT_KEYS}
            options = {key: value for key, value in kwargs.items() if key not in INPUT_KEYS}
            with torch.inference_mode():
                output = controller.generate(inputs, options)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("Native InternVL must return one suffix-only token sequence")
            metrics = controller.last_metrics
            if config["token_keep_ratio"] < 1 and not metrics:
                raise RuntimeError("Native InternVL backend produced no Token/KV evidence")
            send_message(connection, {"status": "completed", "index": request["index"],
                             "output_ids": cpu_value(output), "metrics": metrics})
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


class NativeBackend:
    worker_module = "search.internvl_native_worker"
    model_name = "InternVL"

    def __init__(self, config):
        self.config = dict(config)
        self.index = 0
        self.process = None
        self.connection = None
        self.timeout = float(config.get("native_worker_timeout_seconds", 600))
        if self.timeout <= 0:
            raise ValueError("Native worker timeout must be positive")
        python = (PROJECT_ROOT / config["native_worker_python"]).absolute()
        environment = os.environ.copy()
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        parent, child = socket.socketpair()
        self.connection = Connection(parent.detach())
        try:
            self.process = subprocess.Popen(
                [str(python), "-B", "-m", self.worker_module, str(child.fileno()), json.dumps(config)],
                cwd=PROJECT_ROOT, env=environment, pass_fds=(child.fileno(),),
            )
        except BaseException:
            self.connection.close()
            self.connection = None
            raise
        finally:
            child.close()
        try:
            ready = self.receive()
            if ready.get("status") != "ready" or ready.get("model_digest") != digest(config):
                raise RuntimeError("Native worker loaded a different model descriptor")
            self.validate_ready(ready)
            self.observed_model = ready
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def validate_ready(self, ready):
        header = json.loads((PROJECT_ROOT / self.config["path"] / "config.json").read_text())
        widths = header.get("a4fvlm_stack_widths")
        if widths is not None and ready.get("physical_widths") != widths:
            raise RuntimeError("Native worker physical widths differ from the selected Parent")
        if header.get("a4fvlm_internvl_gptq") is not None:
            manifest = json.loads((PROJECT_ROOT / self.config["path"] / "manifest.json").read_text())
            required = sorted(name for name, precision in manifest["module_assignment"].items()
                              if precision == "int4_w4a16")
            if not required or sorted(ready.get("quantized_modules", [])) != required:
                raise RuntimeError("Native worker did not load all declared quantized kernels")

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"Native {self.model_name} generation worker timed out")
        try:
            response = receive_message(self.connection)
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Native {self.model_name} generation worker exited without a result") from error
        if response.get("status") == "error":
            raise RuntimeError(f"Native {self.model_name} worker: " + response["error"])
        return response

    def generate(self, **kwargs):
        try:
            send_message(self.connection, {"index": self.index, "kwargs": cpu_value(kwargs)})
            response = self.receive()
            if response.get("status") != "completed" or response.get("index") != self.index:
                raise RuntimeError("Native worker generation receipt does not match the request")
            metrics = response["metrics"]
            if self.config["token_keep_ratio"] < 1:
                if not metrics or metrics.get("requested_keep_ratio") != self.config["token_keep_ratio"]:
                    raise RuntimeError("Native worker returned mismatched Token evidence")
                receipt = os.environ.get("A4FVLMSP_TOKEN_RECEIPT")
                if receipt:
                    with Path(receipt).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**metrics, "sample_index": self.index,
                                                 "method": "transformers_score_based_post_encoder",
                                                 "token_keep_ratio": self.config["token_keep_ratio"]}) + "\n")
            self.index += 1
            return response["output_ids"]
        except BaseException:
            self.close()
            raise

    def close(self):
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                send_message(connection, None)
            except (OSError, EOFError):
                pass
            connection.close()
        process, self.process = self.process, None
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        atexit.unregister(self.close)


def create_native_chat_proxy(config):
    """Construct a weightless Module for native LMMS/Accelerate, importing lazily."""
    import torch

    class NativeChatProxy(_NativeChatProxy, torch.nn.Module):
        def __init__(self, config):
            torch.nn.Module.__init__(self)
            super().__init__(config)

    return NativeChatProxy(config)


class _NativeChatProxy:
    """Run the checkpoint's unchanged chat method over isolated generation."""
    def __init__(self, config):
        import torch
        from types import MethodType
        from adapters.internvl35_dense import load_processor, native_model_class

        path = (PROJECT_ROOT / config["path"]).resolve()
        header = json.loads((path / "config.json").read_text())
        artifact = any(header.get(key) is not None for key in ("a4fvlm_stack_widths", "a4fvlm_internvl_gptq"))
        processor = load_processor(config, artifact=artifact)
        self.config = processor.config
        self.template = self.config.template
        self.system_message = processor.conversation_factory(self.template).system_message
        self.num_image_token = processor.tokens_per_tile
        self.img_context_token_id = processor.image_token_id
        self.device = torch.device(config["device"])
        self.chat = MethodType(native_model_class(path).chat, self)
        self.backend = NativeBackend(config)

    def generate(self, **kwargs):
        return self.backend.generate(**kwargs)

    def eval(self):
        return self

    def to(self, device):
        import torch
        if torch.device(device) != self.device:
            raise ValueError("Native LMMS cannot move its isolated backend to a different device")
        return self


def main():
    serve(Connection(int(sys.argv[1])), json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
