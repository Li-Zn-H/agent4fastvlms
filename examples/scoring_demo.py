"""Exercise the local scorer with synthetic inputs, without a model or GPU."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.evaluate import compare_outputs, load_samples


def main():
    samples = load_samples(ROOT / "examples/samples.jsonl", media_root=ROOT / "examples")
    baseline = [{"sample_id": "synthetic-red-1", "output_text": "red"}]
    retentions = {}
    for answer in ("red", "blue"):
        candidate = [{"sample_id": "synthetic-red-1", "output_text": answer}]
        result = compare_outputs(
            samples, baseline, candidate,
            maximum_response_tokens=64, negation_window_tokens=3,
        )
        retentions[answer] = result.retention
    if retentions != {"red": 1.0, "blue": 0.0}:
        raise RuntimeError(f"Unexpected synthetic scoring result: {retentions}")
    print(json.dumps({"scope": "synthetic_scoring_only", "retention": retentions}, indent=2))


if __name__ == "__main__":
    main()
