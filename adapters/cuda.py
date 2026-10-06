"""Register the existing single-GPU NVIDIA execution path; do not probe on import."""

from types import MappingProxyType

from adapters.base import AdapterError, HardwareAdapter


class CudaAdapter(HardwareAdapter):
    id = "cuda"
    worker_device = "cuda:0"
    stages = ("dense", "token", "sp", "quantization", "stack", "runtime", "kv_cache", "lmms")
    # Preserve current stage CLI limits until device selection is ported together.
    device_indices = (0, 1)
    entrypoints = MappingProxyType({
        "probe_identity": "search.optimize:_cuda_hardware",
        "launch_worker": "search.optimize:_cuda_launch",
        "vllm_runtime_kwargs": "skills.stack_runtime:build_vllm_runtime_kwargs",
    })

    def validate_device(self, device):
        if type(device) is not int or device not in self.device_indices:
            raise AdapterError("current CUDA stage entry points require device 0 or 1")

    def describe(self):
        return {
            "id": self.id, "vendor": "nvidia", "worker_device": self.worker_device,
            "device_indices": list(self.device_indices), "entrypoints": dict(self.entrypoints),
            "implemented_stages": list(self.stages),
            "execution": "single isolated GPU selected by UUID",
            "capabilities": "unprobed; kernel/dtype/backend support requires environment preflight and actual receipts",
        }
