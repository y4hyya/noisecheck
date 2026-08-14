"""Position bias of a pairwise judge, measured from order swapped runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noisecheck.errors import DataError
from noisecheck.schema import PairedData
from noisecheck.stats.paired import compare_paired


@dataclass(frozen=True)
class PositionResult:
    """Whether the judge's preference follows the seat instead of the answer."""

    n: int
    flip_rate: float
    bias: float
    ci_low: float
    ci_high: float
    p_value: float
    mcnemar_p: float | None
    warnings: tuple[str, ...]


def judge_position(
    swapped: PairedData,
    b: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> PositionResult:
    """Position preference from two judgings with the candidate order swapped.

    The baseline side holds verdicts with the candidate shown second and the
    candidate side the verdicts after swapping it to first. A value of 1 means
    the judge preferred the candidate answer. A fair judge gives the same
    verdicts in both orders, so the bias estimate should sit near zero.
    """
    values = np.concatenate([np.asarray(swapped.baseline), np.asarray(swapped.candidate)])
    if not bool(np.isin(values, (0.0, 1.0)).all()):
        raise DataError("position analysis needs binary preferred or not verdicts, 0 or 1")
    comparison = compare_paired(swapped, b=b, seed=seed, level=level)
    original = np.asarray(swapped.baseline)
    reordered = np.asarray(swapped.candidate)
    flip_rate = float((original != reordered).mean())
    return PositionResult(
        n=comparison.n,
        flip_rate=flip_rate,
        bias=comparison.estimate,
        ci_low=comparison.ci_low,
        ci_high=comparison.ci_high,
        p_value=comparison.p_value,
        mcnemar_p=comparison.mcnemar_p,
        warnings=comparison.warnings,
    )
