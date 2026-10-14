"""CPU-only native LMMS input/context validation before Kimi GPU evaluation."""

import argparse
import json
import os
from pathlib import Path

from search.model import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model-kimi-vl.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="chartqa,mmstar,textvqa_val,docvqa_val,pope")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from search.lmms import TASKS, cache_environment
    config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    os.environ.update(cache_environment(config))
    from adapters.kimi_vl_dense import load_processor, prepare_image
    from search.deepseek_lmms import generation_options
    from lmms_eval.tasks import TaskManager, get_task_dict

    tasks = args.tasks.split(",")
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("Expected distinct native five-task names")
    destination = PROJECT_ROOT / args.output
    destination.mkdir(parents=True, exist_ok=args.resume)
    processor = load_processor(config)
    manager = TaskManager(model_name="kimi_vl")
    result = {"model": config, "scope": "native LMMS CPU preprocessing only; no quality evaluation",
              "context_limit": 131072, "tasks": {}}
    saved = destination / "summary.json"
    if args.resume and saved.is_file():
        previous = json.loads(saved.read_text())
        if previous.get("model") != config or previous.get("context_limit") != 131072:
            raise ValueError("Kimi context descriptor changed; use a new output")
        result["tasks"] = previous.get("tasks", {})
    for name in tasks:
        task = get_task_dict([name], manager)[name]
        task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False)
        if name in result["tasks"]:
            rows = [json.loads(line) for line in (destination / f"{name}.jsonl").read_text().splitlines()]
            expected = result["tasks"][name]
            if (len(rows) != len(task.instances) or len(rows) != expected["sample_count"]
                    or len({row["doc_id"] for row in rows}) != len(rows)
                    or any(row["task"] != name or row["required_context"] > 131072 or not row["fits"]
                           for row in rows)
                    or max(row["required_context"] for row in rows) != expected["maximum_required_context"]):
                raise ValueError("Kimi saved context rows do not match the complete native task")
            print(f"Reusing complete context audit: {name}", flush=True)
            del task
            continue
        maximum = max_prompt = failures = 0
        count = 0
        with (destination / f"{name}.jsonl").open("w") as stream:
            for request in task.instances:
                text, options, visual, doc_id, task_name, split = request.args
                images = visual(task.dataset[split][doc_id])
                if task_name != name or len(images) != 1:
                    raise ValueError("Native Kimi context probe requires one image per task input")
                inputs = prepare_image(processor, images[0], text)
                options, until = generation_options(options)
                prompt = int(inputs["input_ids"].shape[1])
                total = prompt + options["max_new_tokens"]
                row = {"task": name, "doc_id": int(doc_id), "prompt_tokens": prompt,
                       "image_tokens": int((inputs["input_ids"] == 163605).sum()),
                       "image_grid_hws": inputs["image_grid_hws"].tolist(), "generation": options,
                       "until": until, "required_context": total, "fits": total <= 131072}
                stream.write(json.dumps(row) + "\n")
                failures += int(not row["fits"])
                maximum = max(maximum, total)
                max_prompt = max(max_prompt, prompt)
                count += 1
                if count % 500 == 0:
                    stream.flush()
                    print(f"{name}: {count}/{len(task.instances)}, maximum={maximum}", flush=True)
        result["tasks"][name] = {"sample_count": count, "max_prompt_tokens": max_prompt,
                                  "maximum_required_context": maximum, "over_capacity": failures}
        (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["tasks"][name], flush=True)
        del task
    result["all_fit"] = all(x["over_capacity"] == 0 for x in result["tasks"].values())
    result["complete"] = True
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_fit"]:
        raise SystemExit("Native context envelope exceeds model capacity; do not truncate inputs")


if __name__ == "__main__":
    main()
