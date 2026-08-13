"""Paired comparison of two variants on shared examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

from noisecheck.errors import DataError
from noisecheck.schema import PairedData
from noisecheck.stats.resample import bootstrap_ci, sign_flip_p

_SMALL_SAMPLE = 30


@dataclass(frozen=True)
class ComparisonResult:
    """Effect size, uncertainty, and evidence for one metric comparison."""

    metric: str
    n: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    mcnemar_p: float | None
    level: float
    b: int
    seed: int
    warnings: tuple[str, ...]


def compare_paired(
    paired: PairedData,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> ComparisonResult:
    """Compare candidate against baseline on paired data for one metric."""
    if paired.n < 2:
        raise DataError(f"need at least 2 paired observations, got {paired.n}")
    baseline = np.asarray(paired.baseline, dtype=np.float64)
    candidate = np.asarray(paired.candidate, dtype=np.float64)
    deltas = candidate - baseline
    n = int(deltas.size)
    estimate = float(deltas.mean())
    warnings: list[str] = []

    sample_se = float(deltas.std(ddof=1) / np.sqrt(n))
    degenerate = bool(np.all(deltas == deltas[0])) or sample_se == 0.0
    if degenerate:
        se = 0.0
        ci_low = estimate
        ci_high = estimate
        if estimate == 0.0:
            warnings.append("every paired delta is zero, the variants are identical on this data")
        else:
            warnings.append("paired deltas have zero variance, the interval is degenerate")
    else:
        se = sample_se
        ci_low, ci_high = bootstrap_ci(deltas, b=b, seed=seed, level=level)

    p_value = sign_flip_p(deltas, b=b, seed=seed)
    mcnemar_p = _mcnemar_p(baseline, candidate) if _is_binary(baseline, candidate) else None
    if n < _SMALL_SAMPLE:
        warnings.append(
            f"only {n} pairs, the permutation p value is more reliable than the "
            f"interval at this size"
        )

    return ComparisonResult(
        metric=paired.metric,
        n=n,
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        mcnemar_p=mcnemar_p,
        level=level,
        b=b,
        seed=seed,
        warnings=tuple(warnings),
    )


def _is_binary(baseline: NDArray[np.float64], candidate: NDArray[np.float64]) -> bool:
    values = np.concatenate([baseline, candidate])
    return bool(np.isin(values, (0.0, 1.0)).all())


def _mcnemar_p(baseline: NDArray[np.float64], candidate: NDArray[np.float64]) -> float:
    zero_to_one = int(((baseline == 0.0) & (candidate == 1.0)).sum())
    one_to_zero = int(((baseline == 1.0) & (candidate == 0.0)).sum())
    flips = zero_to_one + one_to_zero
    if flips == 0:
        return 1.0
    return float(binomtest(zero_to_one, flips, 0.5).pvalue)
