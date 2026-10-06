"""Plan or execute runtime acceleration candidates on a frozen Stack."""

from __future__ import annotations

from search.stack_acceleration import execute as _execute
from search.stack_acceleration import main as _main
from search.stack_acceleration import plan as _plan
from search.stack_acceleration import snapshot as _snapshot


def snapshot(config_path="configs/stack_runtime.json"):
    return _snapshot(config_path, kind="runtime")


def plan(config_path="configs/stack_runtime.json"):
    return _plan(config_path, kind="runtime")


def execute(config_path="configs/stack_runtime.json", *, output=None):
    return _execute(config_path, kind="runtime", output=output)


def main():
    _main("runtime")


if __name__ == "__main__":
    main()
