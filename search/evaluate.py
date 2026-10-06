"""CPU-only answer scoring and baseline comparison; see docs/evaluation.md."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from statistics import fmean
import unicodedata


_STRICT = "deterministic_rule_or_structure"
_SCORING_TYPES = {
    _STRICT,
    "exact_or_normalized_short_answer",
    "open_response_or_explicit_review",
}
# Preserve the original alternative order, including its decimal-splitting behavior.
_ANSWER_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['’][^\W_]+)*|[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?",
    flags=re.UNICODE,
)
_NEGATIONS = frozenset({"no", "not", "never", "neither", "nor", "without"})
_ARTICLES = frozenset({"a", "an", "the"})


@dataclass(frozen=True)
class TaskQuality:
    task_id: str
    sample_count: int
    baseline_mean: float
    candidate_mean: float


@dataclass(frozen=True)
class QualityResult:
    tasks: tuple[TaskQuality, ...]
    baseline_mean: float
    candidate_mean: float
    retention: float


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _answer_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(
        t for t in _ANSWER_TOKEN_PATTERN.findall(normalized) if t not in _ARTICLES
    )


def score_answer(
    output_text: str,
    answers: Sequence[str],
    scoring_type: str,
    *,
    maximum_response_tokens: int,
    negation_window_tokens: int,
) -> float:
    """Return 0 or 1 using the existing accepted-answer matching rules."""
    if scoring_type not in _SCORING_TYPES:
        raise ValueError(f"Unsupported scoring_type: {scoring_type!r}")
    if not answers or isinstance(answers, str) or any(not a.strip() for a in answers):
        raise ValueError("answers must contain nonempty reference strings")
    if maximum_response_tokens < 1 or negation_window_tokens < 0:
        raise ValueError("Invalid scoring limits")

    if scoring_type == _STRICT:
        return float(
            _normalize_text(output_text) in {_normalize_text(a) for a in answers}
        )

    tokens = _answer_tokens(output_text)
    if not tokens or len(tokens) > maximum_response_tokens:
        return 0.0
    for answer in answers:
        expected = _answer_tokens(answer)
        if not expected:
            continue
        for start in range(len(tokens) - len(expected) + 1):
            if tokens[start : start + len(expected)] != expected:
                continue
            preceding = tokens[max(0, start - negation_window_tokens) : start]
            if not _NEGATIONS.intersection(preceding):
                return 1.0
    return 0.0


def _sample_ids(samples: Sequence[Mapping]) -> set[str]:
    ids = {row["stable_row_id"] for row in samples}
    if not ids or not all(ids) or len(ids) != len(samples):
        raise ValueError("Sample IDs must be nonempty and unique")
    return ids


def load_samples(path: str | Path, *, media_root: str | Path) -> tuple[dict, ...]:
    """Read existing JSONL rows and check image paths without decoding images.

    Callers resolve config paths against the project root. The selected config
    path determines Search versus Dev; the local row format is documented.
    """
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    _sample_ids(rows)
    root = Path(media_root).resolve(strict=True)
    for row in rows:
        if not row["logical_question"].strip() or not row["primary_category"].strip():
            raise ValueError("Question and task must be nonempty")
        logical = Path(row["media"]["logical_path"])
        image_path = (root / logical).resolve()
        if (
            logical.is_absolute()
            or not image_path.is_relative_to(root)
            or not image_path.is_file()
        ):
            raise ValueError(f"Image is missing or outside media_root: {logical}")
    return rows


def _output_texts(sample_ids: set[str], outputs: Sequence[Mapping]) -> dict[str, str]:
    """Require one successful response per sample; input order may differ."""
    by_id = {row["sample_id"]: row["output_text"] for row in outputs}
    if len(by_id) != len(outputs) or set(by_id) != sample_ids:
        raise ValueError("Outputs must match the complete sample set")
    return by_id


def compare_outputs(
    samples: Sequence[Mapping],
    baseline_outputs: Sequence[Mapping],
    candidate_outputs: Sequence[Mapping],
    *,
    maximum_response_tokens: int,
    negation_window_tokens: int,
) -> QualityResult:
    """Divide equal-task macro means on exactly the same samples and rules."""
    options = dict(
        maximum_response_tokens=maximum_response_tokens,
        negation_window_tokens=negation_window_tokens,
    )
    sample_ids = _sample_ids(samples)
    baseline = _output_texts(sample_ids, baseline_outputs)
    candidate = _output_texts(sample_ids, candidate_outputs)
    grouped = defaultdict(list)
    for sample in samples:
        sample_id = sample["stable_row_id"]
        scores = tuple(
            score_answer(
                outputs[sample_id], sample["answers"], sample["scoring_type"], **options
            )
            for outputs in (baseline, candidate)
        )
        grouped[sample["primary_category"]].append(scores)
    tasks = tuple(
        TaskQuality(
            task, len(pairs), fmean(p[0] for p in pairs), fmean(p[1] for p in pairs)
        )
        for task, pairs in sorted(grouped.items())
    )
    baseline_mean = fmean(task.baseline_mean for task in tasks)
    candidate_mean = fmean(task.candidate_mean for task in tasks)
    if baseline_mean <= 0.0:
        raise ValueError("Retention is undefined for a zero-score baseline")
    return QualityResult(
        tasks, baseline_mean, candidate_mean, candidate_mean / baseline_mean
    )
