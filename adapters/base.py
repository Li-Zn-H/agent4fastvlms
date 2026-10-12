"""Small registration contracts; importing them never imports an inference backend."""

from abc import ABC, abstractmethod
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping


class AdapterError(ValueError):
    """An unknown/unimplemented adapter is a preflight block, not a quality failure."""

    def __init__(self, message):
        super().__init__(f"blocked_preflight: {message}")


class Adapter(ABC):
    id: str
    entrypoints: Mapping[str, str] = MappingProxyType({})

    def resolve(self, name: str):
        """Resolve one declared callable only when explicitly used in its environment."""
        target = self.entrypoints.get(name)
        if target is None:
            raise AdapterError(f"{self.id} has no registered {name} entry point")
        module, attribute = target.split(":", 1)
        try:
            value = getattr(import_module(module), attribute)
        except (ImportError, AttributeError) as error:
            raise AdapterError(f"cannot resolve {target} in the selected Worker environment: {error}") from error
        if not callable(value):
            raise AdapterError(f"{target} is not callable")
        return value

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-compatible declarations, without imports, probes or writes."""


class ModelAdapter(Adapter):
    generation_includes_prompt = True
    model_types: tuple[str, ...] = ()
    # Stages must be explicitly implemented; missing declarations never fall back.
    stages: tuple[str, ...] = ()
    hardware_ids: tuple[str, ...] = ()
    # Registration alone must not enable the complete pipeline.
    pipeline_ids: tuple[str, ...] = ()

    @abstractmethod
    def validate_checkpoint(self, checkpoint: Mapping[str, Any], *, pipeline: bool = False) -> None:
        """Check architecture and implemented scope using checkpoint JSON only."""

    def validate_stage(self, stage: str, checkpoint: Mapping[str, Any]) -> None:
        if stage not in self.stages:
            raise AdapterError(f"{self.id} has no implemented {stage} stage")
        self.validate_checkpoint(checkpoint)


class HardwareAdapter(Adapter):
    worker_device: str
    stages: tuple[str, ...] = ()

    @abstractmethod
    def validate_device(self, device: Any) -> None:
        """Validate the configured selector without querying hardware."""
