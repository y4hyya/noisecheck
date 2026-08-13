"""Run to run variance: the literal noise floor of a metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t

from noisecheck.errors import DataError
from noisecheck.schema import Dataset


@dataclass(frozen=True)
class NoiseFloor:
    """How much one metric moves between identical runs of one variant."""

    metric: str
    variant: str
    n_runs: int
    run_means: tuple[float, ...]
    run_sd: float
    floor: float
    flip_rates: tuple[tuple[str, float], ...] | None


def noise_floor(dataset: Dataset, variant: str, metric: str) -> NoiseFloor:
    """Measure how much a metric moves across repeated runs with zero changes.

    Differences smaller than the floor are weather, not climate.
    """
    if variant not in dataset.variants():
        raise DataError(f"variant {variant!r} is not present in the data")
    if metric not in dataset.metrics():
        raise DataError(f"metric {metric!r} is not present in the data")
    rows = [r for r in dataset.records if r.variant == variant and r.metric == metric]
    missing = sum(1 for r in rows if r.run_id is None)
    if missing:
        raise DataError(f"{missing} rows have no run id, the floor needs repeated runs")

    by_run: dict[str, dict[str, float]] = {}
    for row in rows:
        assert row.run_id is not None
        by_run.setdefault(row.run_id, {})[row.example_id] = row.value
    runs = sorted(by_run)
    if len(runs) < 2:
        raise DataError(f"need at least 2 runs to measure the floor, got {len(runs)}")
    example_sets = {run: frozenset(by_run[run]) for run in runs}
    first = example_sets[runs[0]]
    if any(example_sets[run] != first for run in runs[1:]):
        raise DataError("runs cover different examples, the floor would be biased")

    run_means = tuple(float(np.mean(list(by_run[run].values()))) for run in runs)
    run_sd = float(np.std(run_means, ddof=1))
    half_width = float(t.ppf(0.975, len(runs) - 1)) * run_sd

    flip_rates: tuple[tuple[str, float], ...] | None = None
    if dataset.is_binary(metric):
        n_runs = len(runs)
        rates: list[tuple[str, float]] = []
        for example in sorted(first):
            passes = sum(int(by_run[run][example] == 1.0) for run in runs)
            rate = min(passes, n_runs - passes) / n_runs
            if rate > 0.0:
                rates.append((example, rate))
        rates.sort(key=lambda item: (-item[1], item[0]))
        flip_rates = tuple(rates)

    return NoiseFloor(
        metric=metric,
        variant=variant,
        n_runs=len(runs),
        run_means=run_means,
        run_sd=run_sd,
        floor=half_width,
        flip_rates=flip_rates,
    )
