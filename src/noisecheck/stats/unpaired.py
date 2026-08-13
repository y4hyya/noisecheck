"""Welch comparison for two variants without shared examples."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ttest_ind

from noisecheck.errors import DataError


@dataclass(frozen=True)
class UnpairedResult:
    """Effect size, uncertainty, and evidence for an unpaired comparison."""

    n_baseline: int
    n_candidate: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    level: float
    warnings: tuple[str, ...]


def compare_unpaired(
    baseline: ArrayLike,
    candidate: ArrayLike,
    level: float = 0.95,
) -> UnpairedResult:
    """Compare two independent samples with the welch test.

    Pairing on shared examples is far more sensitive, use this only when the
    two variants were measured on different examples.
    """
    base = _as_sample(baseline, "baseline")
    cand = _as_sample(candidate, "candidate")
    if bool(np.all(base == base[0])) and bool(np.all(cand == cand[0])):
        raise DataError("both variants have zero variance, the welch test needs spread")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        outcome = ttest_ind(cand, base, equal_var=False)
        interval = outcome.confidence_interval(level)
    notes = ["unpaired comparison, pairing on shared examples detects far smaller differences"]
    if caught:
        notes.append("precision loss reported by scipy, some values are nearly identical")

    se = float(np.sqrt(base.var(ddof=1) / base.size + cand.var(ddof=1) / cand.size))
    return UnpairedResult(
        n_baseline=int(base.size),
        n_candidate=int(cand.size),
        estimate=float(cand.mean() - base.mean()),
        se=se,
        ci_low=float(interval.low),
        ci_high=float(interval.high),
        p_value=float(outcome.pvalue),
        level=level,
        warnings=tuple(notes),
    )


def _as_sample(values: ArrayLike, name: str) -> NDArray[np.float64]:
    sample = np.asarray(values, dtype=np.float64).ravel()
    if sample.size < 2:
        raise DataError(f"need at least 2 values per variant, {name} has {sample.size}")
    if not np.isfinite(sample).all():
        raise DataError(f"{name} contains non finite values")
    return sample
