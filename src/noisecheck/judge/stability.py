"""Judge stability across repeated judgings of identical inputs."""

from __future__ import annotations

from dataclasses import dataclass

from noisecheck.schema import Dataset
from noisecheck.stats.variance import noise_floor

_FEW_RUNS = 5


@dataclass(frozen=True)
class StabilityResult:
    """How often a judge changes its mind when nothing else changes."""

    metric: str
    variant: str
    n_runs: int
    n_examples: int
    flip_rate: float | None
    run_sd: float
    floor: float
    flip_rates: tuple[tuple[str, float], ...] | None
    warnings: tuple[str, ...]


def judge_stability(dataset: Dataset, variant: str, metric: str) -> StabilityResult:
    """Measure a judge's self consistency across repeated runs on identical inputs."""
    base = noise_floor(dataset, variant, metric)
    rows = sum(1 for r in dataset.records if r.variant == variant and r.metric == metric)
    n_examples = rows // base.n_runs
    flip_rate: float | None = None
    if base.flip_rates is not None:
        flip_rate = sum(rate for _, rate in base.flip_rates) / n_examples
    warnings: list[str] = []
    if base.n_runs < _FEW_RUNS:
        warnings.append(f"only {base.n_runs} runs, stability estimates are rough below {_FEW_RUNS}")
    return StabilityResult(
        metric=metric,
        variant=variant,
        n_runs=base.n_runs,
        n_examples=n_examples,
        flip_rate=flip_rate,
        run_sd=base.run_sd,
        floor=base.floor,
        flip_rates=base.flip_rates,
        warnings=tuple(warnings),
    )
