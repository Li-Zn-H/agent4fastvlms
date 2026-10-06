"""Finite retention-ratio generation for the unified Vision Token runner."""

from decimal import Decimal


def candidate_ratios(config):
    start, lower, step = (
        Decimal(str(config[key]))
        for key in ("start_keep_ratio", "minimum_keep_ratio", "keep_ratio_step")
    )
    if (
        not all(x.is_finite() for x in (start, lower, step))
        or not 0 < lower <= start < 1
        or step <= 0
    ):
        raise ValueError("Require 0 < minimum <= start < 1 and positive step")
    if (
        type(config["consecutive_dev_failures"]) is not int
        or config["consecutive_dev_failures"] < 1
    ):
        raise ValueError("consecutive_dev_failures must be a positive integer")
    values = []
    while start > lower:
        values.append(float(start))
        start -= step
    return values + [float(lower)]
