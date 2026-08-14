"""Paired comparison of two variants on shared examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

from noisecheck.errors import DataError
from noisecheck.schema import PairedData
from noisecheck.stats.power import cluster_se, design_effect, effective_sample_size, icc
from noisecheck.stats.resample import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    cluster_sign_flip_p,
    sign_flip_p,
)

_SMALL_SAMPLE = 30
_FEW_CLUSTERS = 10


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
    n_clusters: int | None = None
    icc: float | None = None
    design_effect: float | None = None
    effective_n: float | None = None


def compare_paired(
    paired: PairedData,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> ComparisonResult:
    """Compare candidate against baseline on paired data for one metric.

    Resampling happens at the cluster level whenever cluster ids are present,
    because rows inside one cluster move together and are not independent.
    """
    if paired.n < 2:
        raise DataError(f"need at least 2 paired observations, got {paired.n}")
    baseline = np.asarray(paired.baseline, dtype=np.float64)
    candidate = np.asarray(paired.candidate, dtype=np.float64)
    deltas = candidate - baseline
    n = int(deltas.size)
    estimate = float(deltas.mean())
    warnings: list[str] = []

    keys, missing = cluster_keys(paired.cluster_ids)
    icc_value: float | None = None
    deff: float | None = None
    effective_n: float | None = None
    n_clusters: int | None = None

    if keys is None:
        if len(set(paired.example_ids)) < n:
            warnings.append(
                "the same example appears in several rows but no cluster ids are given, "
                "rows are treated as independent and the interval may be too narrow"
            )
        sample_se = float(deltas.std(ddof=1) / np.sqrt(n))
    else:
        k = len(set(keys))
        if k == 1:
            raise DataError(
                "all rows share a single cluster, between cluster uncertainty cannot be estimated"
            )
        if missing:
            warnings.append(
                f"{missing} rows have no cluster id, each one counts as its own cluster"
            )
        if k < _FEW_CLUSTERS:
            warnings.append(
                f"only {k} clusters, cluster level resampling is unstable below {_FEW_CLUSTERS}"
            )
        n_clusters = k
        icc_value = icc(deltas, keys)
        deff = design_effect(icc_value, n / k)
        effective_n = effective_sample_size(n, deff)
        sample_se = cluster_se(deltas, keys)

    degenerate = bool(np.all(deltas == deltas[0])) or sample_se == 0.0
    if degenerate:
        se = 0.0
        ci_low = estimate
        ci_high = estimate
        if estimate == 0.0:
            warnings.append("every paired delta is zero, the variants are identical on this data")
        else:
            warnings.append("paired deltas have zero variance, the interval is degenerate")
    elif keys is None:
        se = sample_se
        ci_low, ci_high = bootstrap_ci(deltas, b=b, seed=seed, level=level)
    else:
        se = sample_se
        ci_low, ci_high = cluster_bootstrap_ci(deltas, keys, b=b, seed=seed, level=level)

    if keys is None:
        p_value = sign_flip_p(deltas, b=b, seed=seed)
    else:
        p_value = cluster_sign_flip_p(deltas, keys, b=b, seed=seed)

    mcnemar_p = _mcnemar_p(baseline, candidate) if _is_binary(baseline, candidate) else None
    if mcnemar_p is not None and keys is not None:
        warnings.append(
            "mcnemar assumes independent pairs, with clusters trust the permutation p value"
        )
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
        n_clusters=n_clusters,
        icc=icc_value,
        design_effect=deff,
        effective_n=effective_n,
    )


def cluster_keys(cluster_ids: tuple[str | None, ...]) -> tuple[list[str] | None, int]:
    if all(label is None for label in cluster_ids):
        return None, 0
    missing = sum(1 for label in cluster_ids if label is None)
    keys = [label if label is not None else f"\x00solo{i}" for i, label in enumerate(cluster_ids)]
    return keys, missing


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
