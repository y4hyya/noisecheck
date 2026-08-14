"""Seeded resampling engines: studentized bootstrap and sign flip permutation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from noisecheck.stats.power import cluster_sums, group_codes

_BLOCK_CELLS = 2_000_000


def bootstrap_ci(
    deltas: ArrayLike,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    values = _as_deltas(deltas)
    n = values.size
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    if bool(np.all(values == values[0])) or se == 0.0:
        raise ValueError("deltas have zero variance, the interval would be meaningless")
    rng = np.random.default_rng(seed)
    studentized: list[NDArray[np.float64]] = []
    for rows in blocks(b, n):
        indices = rng.integers(0, n, size=(rows, n))
        samples = values[indices]
        means = samples.mean(axis=1)
        errors = samples.std(axis=1, ddof=1) / np.sqrt(n)
        spread = samples.max(axis=1) - samples.min(axis=1)
        errors = np.where((spread == 0.0) | (errors == 0.0), se, errors)
        studentized.append((means - estimate) / errors)
    t_all = np.concatenate(studentized)
    alpha = 1.0 - level
    quantiles = np.quantile(t_all, [alpha / 2.0, 1.0 - alpha / 2.0])
    lo_t = float(quantiles[0])
    hi_t = float(quantiles[1])
    return (estimate - hi_t * se, estimate - lo_t * se)


def sign_flip_p(deltas: ArrayLike, b: int = 10_000, seed: int = 42) -> float:
    values = _as_deltas(deltas)
    n = values.size
    observed = abs(float(values.mean()))
    threshold = observed * (1.0 - 1e-12)
    rng = np.random.default_rng(seed)
    flip_choices = np.array([-1.0, 1.0])
    extreme = 0
    for rows in blocks(b, n):
        signs = rng.choice(flip_choices, size=(rows, n))
        stats = np.abs((signs * values).mean(axis=1))
        extreme += int((stats >= threshold).sum())
    return (extreme + 1) / (b + 1)


def cluster_bootstrap_ci(
    deltas: ArrayLike,
    clusters: Sequence[str],
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    values = _as_deltas(deltas)
    sums, sizes = _cluster_summary(values, clusters)
    k = sums.size
    total = values.size
    estimate = float(values.mean())
    residuals = sums - sizes * estimate
    se = float(np.sqrt((k / (k - 1)) * float((residuals**2).sum()) / total**2))
    if bool(np.all(values == values[0])) or se == 0.0:
        raise ValueError("deltas have zero variance, the interval would be meaningless")
    cluster_means = sums / sizes
    rng = np.random.default_rng(seed)
    studentized: list[NDArray[np.float64]] = []
    for rows in blocks(b, k):
        indices = rng.integers(0, k, size=(rows, k))
        drawn_sums = sums[indices]
        drawn_sizes = sizes[indices]
        totals = drawn_sizes.sum(axis=1)
        means = drawn_sums.sum(axis=1) / totals
        resid = drawn_sums - drawn_sizes * means[:, None]
        variances = (k / (k - 1)) * (resid**2).sum(axis=1) / totals**2
        errors = np.sqrt(variances)
        drawn_means = cluster_means[indices]
        spread = drawn_means.max(axis=1) - drawn_means.min(axis=1)
        errors = np.where((spread == 0.0) | (errors == 0.0), se, errors)
        studentized.append((means - estimate) / errors)
    t_all = np.concatenate(studentized)
    alpha = 1.0 - level
    quantiles = np.quantile(t_all, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (estimate - float(quantiles[1]) * se, estimate - float(quantiles[0]) * se)


def cluster_sign_flip_p(
    deltas: ArrayLike,
    clusters: Sequence[str],
    b: int = 10_000,
    seed: int = 42,
) -> float:
    values = _as_deltas(deltas)
    sums, _ = _cluster_summary(values, clusters)
    k = sums.size
    total = values.size
    observed = abs(float(sums.sum() / total))
    threshold = observed * (1.0 - 1e-12)
    rng = np.random.default_rng(seed)
    flip_choices = np.array([-1.0, 1.0])
    extreme = 0
    for rows in blocks(b, k):
        signs = rng.choice(flip_choices, size=(rows, k))
        stats = np.abs((signs * sums).sum(axis=1)) / total
        extreme += int((stats >= threshold).sum())
    return (extreme + 1) / (b + 1)


def _cluster_summary(
    values: NDArray[np.float64], clusters: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(clusters) != values.size:
        raise ValueError("clusters must match deltas in length")
    codes = group_codes(clusters)
    sums, sizes = cluster_sums(values, codes)
    if sums.size < 2:
        raise ValueError(f"need at least 2 clusters, got {sums.size}")
    return sums, sizes


def _as_deltas(deltas: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(deltas, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError(f"need at least 2 deltas, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("deltas must be finite numbers")
    return values


def blocks(total: int, width: int) -> Iterator[int]:
    if total < 1:
        raise ValueError(f"need at least 1 resample, got {total}")
    rows_per_block = max(1, _BLOCK_CELLS // max(width, 1))
    remaining = total
    while remaining > 0:
        rows = min(rows_per_block, remaining)
        yield rows
        remaining -= rows
