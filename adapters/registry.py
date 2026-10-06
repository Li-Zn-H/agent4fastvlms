"""Explicit registration with duplicate/unknown checks and no backend discovery."""

import re
from typing import Generic, TypeVar

from adapters.base import Adapter, AdapterError, HardwareAdapter, ModelAdapter


T = TypeVar("T", bound=Adapter)


class Registry(Generic[T]):
    def __init__(self, adapter_type: type[T]):
        self.adapter_type = adapter_type
        self._items: dict[str, T] = {}

    def register(self, adapter: T) -> T:
        if not isinstance(adapter, self.adapter_type):
            raise TypeError(f"Expected {self.adapter_type.__name__}")
        if not isinstance(adapter.id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", adapter.id):
            raise ValueError("Adapter id must be a lowercase identifier")
        if adapter.id in self._items:
            raise ValueError(f"Duplicate adapter id: {adapter.id}")
        for name, target in adapter.entrypoints.items():
            if not isinstance(name, str) or not isinstance(target, str) or not re.fullmatch(
                r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", target
            ):
                raise ValueError(f"Invalid lazy entry point: {name}={target!r}")
        if isinstance(adapter, ModelAdapter):
            if not adapter.model_types or any(not isinstance(x, str) or not x for x in adapter.model_types):
                raise ValueError("Model adapters must declare nonempty model_types")
            for previous in self._items.values():
                if set(adapter.model_types) & set(previous.model_types):
                    raise ValueError("Duplicate model_type ownership")
                if set(adapter.pipeline_ids) & set(previous.pipeline_ids):
                    raise ValueError("Duplicate pipeline model_id ownership")
        self._items[adapter.id] = adapter
        return adapter

    def get(self, adapter_id: str) -> T:
        try:
            return self._items[adapter_id]
        except KeyError as error:
            raise AdapterError(f"unknown {self.adapter_type.__name__}: {adapter_id}") from error

    def entries(self) -> tuple[T, ...]:
        # Built-ins are registered in a deliberate, stable order.  Preserve
        # that order in the read-only catalog so adding a later migration
        # adapter cannot reorder the established Qwen reference entry.
        return tuple(self._items.values())


models = Registry(ModelAdapter)
hardware = Registry(HardwareAdapter)


def select_model(checkpoint, *, pipeline_id=None):
    matches = [item for item in models.entries() if checkpoint.get("model_type") in item.model_types]
    if not matches:
        raise AdapterError(f"no registered adapter for model_type={checkpoint.get('model_type')!r}")
    adapter = matches[0]
    if pipeline_id is not None and pipeline_id not in adapter.pipeline_ids:
        raise AdapterError(f"{adapter.id} is not integrated with pipeline model_id={pipeline_id!r}")
    adapter.validate_checkpoint(checkpoint, pipeline=pipeline_id is not None)
    return adapter
