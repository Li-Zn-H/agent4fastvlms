"""Explicit built-ins. Add future registrations here after implementing their contracts."""

from adapters.base import AdapterError, HardwareAdapter, ModelAdapter
from adapters.registry import hardware, models, select_model
from adapters.cuda import CudaAdapter
from adapters.qwen3vl import Qwen3VLAdapter
from adapters.llava_onevision import LlavaOneVisionAdapter
from adapters.llava15 import Llava15Adapter


models.register(Qwen3VLAdapter())
models.register(LlavaOneVisionAdapter())
models.register(Llava15Adapter())
hardware.register(CudaAdapter())

__all__ = ["AdapterError", "HardwareAdapter", "ModelAdapter", "hardware", "models", "select_model"]
