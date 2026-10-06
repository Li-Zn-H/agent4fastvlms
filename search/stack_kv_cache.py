"""Plan or execute KV-cache candidates on a runtime-frozen Stack."""

from __future__ import annotations

from search.stack_acceleration import execute as _execute
from search.stack_acceleration import main as _main
from search.stack_acceleration import plan as _plan
from search.stack_acceleration import snapshot as _snapshot


def snapshot(config_path="configs/stack_kv_cache.json"):
    return _snapshot(config_path, kind="kv_cache")


def plan(config_path="configs/stack_kv_cache.json"):
    return _plan(config_path, kind="kv_cache")


def execute(config_path="configs/stack_kv_cache.json", *, output=None):
    return _execute(config_path, kind="kv_cache", output=output)


def main():
    _main("kv_cache")


if __name__ == "__main__":
    main()
