"""Seeded resampling engines: studentized bootstrap and sign flip permutation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

_BLOCK_CELLS = 2_000_000


def bootstrap_ci(
    deltas: ArrayLike,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    values = _as_deltas(deltas)
    if bool(np.all(values == values[0])):
        raise ValueError("deltas have zero variance, the interval would be meaningless")
    n = values.size
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    studentized: list[NDArray[np.float64]] = []
    for rows in _blocks(b, n):
        indices = rng.integers(0, n, size=(rows, n))
        samples = values[indices]
        means = samples.mean(axis=1)
        errors = samples.std(axis=1, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (means - estimate) / errors
        studentized.append(np.where(np.isnan(t), 0.0, t))
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
    for rows in _blocks(b, n):
        signs = rng.choice(flip_choices, size=(rows, n))
        stats = np.abs((signs * values).mean(axis=1))
        extreme += int((stats >= threshold).sum())
    return (extreme + 1) / (b + 1)


def _as_deltas(deltas: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(deltas, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError(f"need at least 2 deltas, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("deltas must be finite numbers")
    return values


def _blocks(total: int, width: int) -> Iterator[int]:
    if total < 1:
        raise ValueError(f"need at least 1 resample, got {total}")
    rows_per_block = max(1, _BLOCK_CELLS // max(width, 1))
    remaining = total
    while remaining > 0:
        rows = min(rows_per_block, remaining)
        yield rows
        remaining -= rows
