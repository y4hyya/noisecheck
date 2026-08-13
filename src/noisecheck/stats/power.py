"""Cluster diagnostics and power arithmetic for eval comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def group_codes(clusters: Sequence[str]) -> NDArray[np.int64]:
    seen: dict[str, int] = {}
    return np.array([seen.setdefault(label, len(seen)) for label in clusters], dtype=np.int64)


def cluster_sums(
    values: NDArray[np.float64], codes: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    count = int(codes.max()) + 1
    sums = np.zeros(count)
    np.add.at(sums, codes, values)
    sizes = np.bincount(codes, minlength=count).astype(np.float64)
    return sums, sizes


def cluster_se(values: ArrayLike, clusters: Sequence[str]) -> float:
    """Cluster robust standard error of the mean."""
    data = np.asarray(values, dtype=np.float64).ravel()
    codes = group_codes(clusters)
    sums, sizes = cluster_sums(data, codes)
    k = sums.size
    if k < 2:
        raise ValueError(f"need at least 2 clusters, got {k}")
    total = data.size
    mean = float(sums.sum() / total)
    residuals = sums - sizes * mean
    variance = (k / (k - 1)) * float((residuals**2).sum()) / total**2
    return float(np.sqrt(variance))


def icc(values: ArrayLike, clusters: Sequence[str]) -> float:
    """Intraclass correlation from a one way analysis of variance, clipped to [0, 1]."""
    data = np.asarray(values, dtype=np.float64).ravel()
    codes = group_codes(clusters)
    sums, sizes = cluster_sums(data, codes)
    k = sums.size
    n = data.size
    if k < 2:
        raise ValueError(f"need at least 2 clusters, got {k}")
    if k == n:
        return 0.0
    grand_mean = float(data.mean())
    cluster_means = sums / sizes
    between = float((sizes * (cluster_means - grand_mean) ** 2).sum()) / (k - 1)
    within = float(((data - cluster_means[codes]) ** 2).sum()) / (n - k)
    average_size = (n - float((sizes**2).sum()) / n) / (k - 1)
    denominator = between + (average_size - 1.0) * within
    if denominator <= 0.0:
        return 0.0
    return float(min(max((between - within) / denominator, 0.0), 1.0))


def design_effect(icc_value: float, average_cluster_size: float) -> float:
    """How many times less informative each row is because of clustering."""
    return 1.0 + (average_cluster_size - 1.0) * icc_value


def effective_sample_size(n: int, deff: float) -> float:
    """How many independent rows the clustered data is really worth."""
    return n / deff


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true difference this eval would detect with the given power."""
    if se <= 0.0:
        raise ValueError(f"standard error must be positive, got {se}")
    return float((norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)) * se)


def items_needed(
    se: float,
    n: int,
    target: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Total paired items required before the eval can detect the target difference."""
    if target <= 0.0:
        raise ValueError(f"target difference must be positive, got {target}")
    current = mde(se, alpha=alpha, power=power)
    return math.ceil(n * (current / target) ** 2)
