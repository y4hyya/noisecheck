"""Judge versus human agreement with chance corrected kappa."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from noisecheck.errors import DataError
from noisecheck.schema import PairedData
from noisecheck.stats.paired import cluster_keys
from noisecheck.stats.power import group_codes
from noisecheck.stats.resample import blocks

_SMALL_SAMPLE = 30
_FEW_CLUSTERS = 10


@dataclass(frozen=True)
class AgreementResult:
    """How well a judge tracks human labels, corrected for lucky agreement."""

    metric: str
    n: int
    kappa: float
    ci_low: float
    ci_high: float
    observed_agreement: float
    confusion: tuple[tuple[str, str, int], ...]
    worst_disagreements: tuple[str, ...]
    weights: str
    warnings: tuple[str, ...]


def judge_agreement(
    paired: PairedData,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
    weights: str = "none",
) -> AgreementResult:
    """Kappa between human labels (the baseline side) and a judge (the candidate side).

    Use linear or quadratic weights when the labels are ordered grades.
    """
    if weights not in ("none", "linear", "quadratic"):
        raise DataError(f"unknown weights {weights!r}, choose none, linear, or quadratic")
    if paired.n < 2:
        raise DataError(f"need at least 2 paired labels, got {paired.n}")
    human = np.asarray(paired.baseline, dtype=np.float64)
    judge_values = np.asarray(paired.candidate, dtype=np.float64)
    categories = np.unique(np.concatenate([human, judge_values]))
    k = int(categories.size)
    labels = [f"{value:g}" for value in categories]
    human_codes = np.searchsorted(categories, human)
    judge_codes = np.searchsorted(categories, judge_values)

    warnings: list[str] = []
    keys, missing = cluster_keys(paired.cluster_ids)
    if keys is None:
        codes = np.arange(paired.n, dtype=np.int64)
        n_clusters = paired.n
    else:
        if missing:
            warnings.append(
                f"{missing} rows have no cluster id, each one counts as its own cluster"
            )
        codes = group_codes(keys)
        n_clusters = int(codes.max()) + 1
        if n_clusters < 2:
            raise DataError("all rows share a single cluster, the interval cannot be estimated")
        if n_clusters < _FEW_CLUSTERS:
            warnings.append(
                f"only {n_clusters} clusters, cluster level resampling is "
                f"unstable below {_FEW_CLUSTERS}"
            )

    cell = human_codes * k + judge_codes
    per_cluster = np.zeros((n_clusters, k * k))
    np.add.at(per_cluster, (codes, cell), 1.0)
    weight_matrix = _weight_matrix(k, weights)
    total_flat = per_cluster.sum(axis=0)
    kappa_hat = float(_kappa_many(total_flat[None, :], k, weight_matrix)[0])
    total = total_flat.reshape(k, k)
    observed_agreement = float(np.trace(total) / paired.n)

    if k == 1:
        warnings.append("every label is the same, kappa is undefined, reporting 0")
        ci_low = 0.0
        ci_high = 0.0
    else:
        rng = np.random.default_rng(seed)
        samples: list[NDArray[np.float64]] = []
        for rows in blocks(b, n_clusters):
            indices = rng.integers(0, n_clusters, size=(rows, n_clusters))
            confs = per_cluster[indices].sum(axis=1)
            samples.append(_kappa_many(confs, k, weight_matrix))
        kappa_star = np.concatenate(samples)
        alpha = 1.0 - level
        quantiles = np.quantile(kappa_star, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci_low = float(quantiles[0])
        ci_high = float(quantiles[1])
    if paired.n < _SMALL_SAMPLE:
        warnings.append(f"only {paired.n} pairs, the interval is unstable at this size")

    confusion = tuple(
        (labels[i], labels[j], int(total[i, j]))
        for i in range(k)
        for j in range(k)
        if total[i, j] > 0
    )
    disagreement = np.abs(human - judge_values)
    order = sorted(
        (index for index in range(paired.n) if disagreement[index] > 0),
        key=lambda index: (-float(disagreement[index]), paired.example_ids[index]),
    )
    worst = tuple(paired.example_ids[index] for index in order)

    return AgreementResult(
        metric=paired.metric,
        n=paired.n,
        kappa=kappa_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        observed_agreement=observed_agreement,
        confusion=confusion,
        worst_disagreements=worst,
        weights=weights,
        warnings=tuple(warnings),
    )


def _weight_matrix(k: int, weights: str) -> NDArray[np.float64]:
    grid = np.arange(k, dtype=np.float64)
    distance = np.abs(grid[:, None] - grid[None, :])
    if weights == "linear":
        return distance
    if weights == "quadratic":
        return distance**2
    return (distance > 0).astype(np.float64)


def _kappa_many(
    confs: NDArray[np.float64], k: int, weight_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    shaped = confs.reshape(-1, k, k)
    n = shaped.sum(axis=(1, 2))
    rows = shaped.sum(axis=2)
    cols = shaped.sum(axis=1)
    expected = rows[:, :, None] * cols[:, None, :] / n[:, None, None]
    numerator = (weight_matrix * shaped).sum(axis=(1, 2))
    denominator = (weight_matrix * expected).sum(axis=(1, 2))
    safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, 1.0 - numerator / safe, 0.0)
