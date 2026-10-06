"""Read-only model/hardware registration catalog; no GPU or inference imports."""

import argparse
import json

from adapters import hardware, models
from adapters.capabilities import check_capabilities
from search.context import PROJECT_ROOT, checkpoint_config, hardware_adapter, model_adapter, validate_result_scope


def catalog(*, model=None, hardware_id=None):
    return {
        "models": [item.describe() for item in ((models.get(model),) if model else models.entries())],
        "hardware": [item.describe() for item in ((hardware.get(hardware_id),) if hardware_id else hardware.entries())],
        "qualification": "registered interfaces only; no capability probes or quality certification performed",
    }


def preflight(model_config, *, evaluation=None, device=None, stages=None):
    model = model_adapter(model_config)
    report = check_capabilities(model, checkpoint_config(model_config), hardware_adapter(model_config),
                                device=device, stages=stages)
    if evaluation is not None:
        report["result_root"] = str(validate_result_scope(model_config, evaluation))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Registered model adapter id")
    parser.add_argument("--hardware", help="Registered hardware adapter id")
    parser.add_argument("--model-config", help="Check a model descriptor without importing a backend")
    parser.add_argument("--evaluation-config", help="Also validate the model's result root")
    parser.add_argument("--device", type=int, help="Physical device selector for static validation")
    parser.add_argument("--stages", help="Comma-separated stages; defaults to all registered contracts")
    args = parser.parse_args()
    try:
        if args.model_config:
            model = json.loads((PROJECT_ROOT / args.model_config).read_text())
            if args.model:
                model["adapter"] = args.model
            if args.hardware:
                model["hardware"] = args.hardware
            evaluation = json.loads((PROJECT_ROOT / args.evaluation_config).read_text()) if args.evaluation_config else None
            value = preflight(model, evaluation=evaluation, device=args.device,
                              stages=args.stages.split(",") if args.stages else None)
        else:
            if args.evaluation_config or args.device is not None or args.stages:
                parser.error("--evaluation-config, --device and --stages require --model-config")
            value = catalog(model=args.model, hardware_id=args.hardware)
    except (OSError, ValueError, KeyError, TypeError) as error:
        value = {"status": "blocked_preflight", "reason": str(error)}
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if value.get("status") == "blocked_preflight":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
