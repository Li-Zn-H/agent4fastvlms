"""CPU-only native LMMS input/context validation before DeepSeek GPU evaluation."""

import argparse
import json
import os
from pathlib import Path

from search.model import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model-deepseek-vl2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="chartqa,mmstar,textvqa_val,docvqa_val,pope")
    args = parser.parse_args()
    from search.lmms import TASKS, cache_environment
    config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    os.environ.update(cache_environment(config))
    from adapters.deepseek_vl2_dense import load_processor, prepare_image
    from search.deepseek_lmms import generation_options
    from lmms_eval.tasks import TaskManager, get_task_dict

    tasks = args.tasks.split(",")
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("Expected distinct native five-task names")
    destination = PROJECT_ROOT / args.output
    destination.mkdir(parents=True, exist_ok=False)
    processor = load_processor(config)
    manager = TaskManager(model_name="deepseek_vl2")
    result = {"model": config, "scope": "native LMMS CPU preprocessing only; no quality evaluation",
              "context_limit": 4096, "tasks": {}}
    for name in tasks:
        task = get_task_dict([name], manager)[name]
        task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False)
        maximum = max_prompt = failures = 0
        count = 0
        with (destination / f"{name}.jsonl").open("w") as stream:
            for request in task.instances:
                text, options, visual, doc_id, task_name, split = request.args
                images = visual(task.dataset[split][doc_id])
                if task_name != name or len(images) != 1:
                    raise ValueError("Native DeepSeek context probe requires one image per task input")
                inputs = prepare_image(processor, images[0], text)
                options, until = generation_options(options)
                prompt = int(inputs["input_ids"].shape[1])
                total = prompt + options["max_new_tokens"]
                row = {"task": name, "doc_id": int(doc_id), "prompt_tokens": prompt,
                       "image_tokens": int(inputs["images_seq_mask"].sum()),
                       "tiles": int(inputs["images"].shape[1]), "generation": options,
                       "until": until, "required_context": total, "fits": total <= 4096}
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
