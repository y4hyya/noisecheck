"""Verdicts, gate policies, and exit codes for eval comparisons."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum

from noisecheck.stats.multiplicity import benjamini_hochberg
from noisecheck.stats.paired import ComparisonResult
from noisecheck.stats.power import items_needed, mde


class Outcome(StrEnum):
    """What one metric comparison concluded."""

    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NO_DETECTABLE_DIFFERENCE = "no_detectable_difference"
    UNDERPOWERED = "underpowered"


class Gate(StrEnum):
    """Which policy turns verdicts into an exit code."""

    NON_REGRESSION = "non_regression"
    IMPROVEMENT = "improvement"


_SEVERITY = {
    Outcome.IMPROVEMENT: 0,
    Outcome.NO_DETECTABLE_DIFFERENCE: 1,
    Outcome.UNDERPOWERED: 2,
    Outcome.REGRESSION: 3,
}


@dataclass(frozen=True)
class MetricVerdict:
    """One metric's outcome with the numbers that justify it."""

    metric: str
    outcome: Outcome
    estimate: float
    p_value: float
    q_value: float
    mde: float
    items_needed: int | None
    reason: str


@dataclass(frozen=True)
class Assessment:
    """All verdicts, the overall outcome, and the exit code for the chosen gate."""

    verdicts: tuple[MetricVerdict, ...]
    overall: Outcome
    gate: Gate
    exit_code: int
    alpha: float
    min_effect: float
    power: float


def judge(
    results: Sequence[ComparisonResult],
    gate: Gate = Gate.NON_REGRESSION,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    power: float = 0.8,
    lower_is_better: Collection[str] = (),
) -> Assessment:
    """Turn comparisons into verdicts and one exit code: 0 pass, 1 fail, 2 cannot tell."""
    if not results:
        raise ValueError("need at least one comparison to judge")
    metrics = [comparison.metric for comparison in results]
    if len(set(metrics)) < len(metrics):
        raise ValueError("duplicate metric names, judge each metric once")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if min_effect < 0.0:
        raise ValueError(f"min effect must not be negative, got {min_effect}")

    flipped = set(lower_is_better)
    q_values = benjamini_hochberg([comparison.p_value for comparison in results])
    verdicts: list[MetricVerdict] = []
    cannot_rule_out_regression: list[bool] = []
    for comparison, q_value in zip(results, q_values, strict=True):
        sign = -1.0 if comparison.metric in flipped else 1.0
        estimate = sign * comparison.estimate
        low, _ = sorted((sign * comparison.ci_low, sign * comparison.ci_high))
        detectable = mde(comparison.se, alpha=alpha, power=power) if comparison.se > 0.0 else 0.0
        significant = q_value < alpha
        needed: int | None = None
        if significant and estimate > 0.0 and estimate >= min_effect:
            outcome = Outcome.IMPROVEMENT
            reason = f"a real improvement of about {estimate:.4g}"
        elif significant and estimate < 0.0 and -estimate >= min_effect:
            outcome = Outcome.REGRESSION
            reason = f"a real regression of about {estimate:.4g}"
        elif significant:
            outcome = Outcome.NO_DETECTABLE_DIFFERENCE
            reason = (
                f"a real difference of about {estimate:.4g}, smaller than the "
                f"minimum effect that matters ({min_effect:g})"
            )
        elif min_effect > 0.0 and detectable <= min_effect:
            outcome = Outcome.NO_DETECTABLE_DIFFERENCE
            reason = (
                f"no difference found, and this eval can detect anything above {detectable:.4g}"
            )
        elif comparison.se == 0.0:
            outcome = Outcome.NO_DETECTABLE_DIFFERENCE
            reason = "the variants are identical on this data"
        else:
            outcome = Outcome.UNDERPOWERED
            target = max(min_effect, abs(estimate))
            if target > 0.0:
                needed = items_needed(comparison.se, comparison.n, target, alpha=alpha, power=power)
            reason = f"this eval cannot detect differences below {detectable:.4g}"
            if needed is not None:
                reason = f"{reason}, about {needed} items would"
        verdicts.append(
            MetricVerdict(
                metric=comparison.metric,
                outcome=outcome,
                estimate=comparison.estimate,
                p_value=comparison.p_value,
                q_value=q_value,
                mde=detectable,
                items_needed=needed,
                reason=reason,
            )
        )
        cannot_rule_out_regression.append(outcome is not Outcome.REGRESSION and low < -min_effect)

    overall = max((verdict.outcome for verdict in verdicts), key=lambda outcome: _SEVERITY[outcome])
    if gate is Gate.NON_REGRESSION:
        if any(verdict.outcome is Outcome.REGRESSION for verdict in verdicts):
            exit_code = 1
        elif any(cannot_rule_out_regression):
            exit_code = 2
        else:
            exit_code = 0
    else:
        if all(verdict.outcome is Outcome.IMPROVEMENT for verdict in verdicts):
            exit_code = 0
        elif any(verdict.outcome is Outcome.UNDERPOWERED for verdict in verdicts):
            exit_code = 2
        else:
            exit_code = 1

    return Assessment(
        verdicts=tuple(verdicts),
        overall=overall,
        gate=gate,
        exit_code=exit_code,
        alpha=alpha,
        min_effect=min_effect,
        power=power,
    )
