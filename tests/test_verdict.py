from __future__ import annotations

import pytest

from noisecheck import Assessment, ComparisonResult, Gate, Outcome, judge


def result(
    metric: str = "accuracy",
    estimate: float = 0.05,
    se: float = 0.01,
    p: float = 0.001,
    ci: tuple[float, float] = (0.03, 0.07),
    n: int = 200,
) -> ComparisonResult:
    return ComparisonResult(
        metric=metric,
        n=n,
        estimate=estimate,
        se=se,
        ci_low=ci[0],
        ci_high=ci[1],
        p_value=p,
        mcnemar_p=None,
        level=0.95,
        b=1000,
        seed=42,
        warnings=(),
    )


class TestOutcomes:
    def test_clear_improvement_passes_both_gates(self) -> None:
        assessment = judge([result()])
        assert assessment.verdicts[0].outcome is Outcome.IMPROVEMENT
        assert assessment.overall is Outcome.IMPROVEMENT
        assert assessment.exit_code == 0
        assert judge([result()], gate=Gate.IMPROVEMENT).exit_code == 0

    def test_clear_regression_fails_both_gates(self) -> None:
        bad = result(estimate=-0.05, ci=(-0.07, -0.03))
        assert judge([bad]).verdicts[0].outcome is Outcome.REGRESSION
        assert judge([bad]).exit_code == 1
        assert judge([bad], gate=Gate.IMPROVEMENT).exit_code == 1

    def test_significant_but_trivial_is_not_a_regression(self) -> None:
        tiny = result(estimate=-0.01, se=0.003, ci=(-0.015, -0.005))
        assessment = judge([tiny], min_effect=0.02)
        assert assessment.verdicts[0].outcome is Outcome.NO_DETECTABLE_DIFFERENCE
        assert "smaller" in assessment.verdicts[0].reason
        assert assessment.exit_code == 0

    def test_underpowered_when_nothing_can_be_seen(self) -> None:
        vague = result(estimate=0.01, se=0.05, p=0.4, ci=(-0.09, 0.11))
        assessment = judge([vague], min_effect=0.02)
        verdict = assessment.verdicts[0]
        assert verdict.outcome is Outcome.UNDERPOWERED
        assert verdict.items_needed is not None
        assert verdict.items_needed > 200
        assert assessment.exit_code == 2
        assert judge([vague], gate=Gate.IMPROVEMENT, min_effect=0.02).exit_code == 2

    def test_powered_null_passes_non_regression_but_fails_improvement(self) -> None:
        quiet = result(estimate=0.01, se=0.02, p=0.5, ci=(-0.03, 0.05))
        assessment = judge([quiet], min_effect=0.1)
        assert assessment.verdicts[0].outcome is Outcome.NO_DETECTABLE_DIFFERENCE
        assert assessment.exit_code == 0
        assert judge([quiet], gate=Gate.IMPROVEMENT, min_effect=0.1).exit_code == 1

    def test_worst_outcome_wins_overall(self) -> None:
        mixed = [result(), result(metric="latency", estimate=-0.05, ci=(-0.07, -0.03))]
        assessment = judge(mixed)
        assert assessment.overall is Outcome.REGRESSION
        assert assessment.exit_code == 1

    def test_multiple_metrics_share_one_alpha_budget(self) -> None:
        borderline = result(metric="a", p=0.04, estimate=0.05, ci=(0.01, 0.09))
        noise = result(metric="b", p=0.9, estimate=0.0, ci=(-0.02, 0.02))
        assessment = judge([borderline, noise])
        verdict = assessment.verdicts[0]
        assert verdict.q_value > 0.05
        assert verdict.outcome is not Outcome.IMPROVEMENT

    def test_lower_is_better_flips_the_direction(self) -> None:
        worse = result(metric="latency", estimate=0.05, ci=(0.03, 0.07))
        better = result(metric="latency", estimate=-0.05, ci=(-0.07, -0.03))
        assert judge([worse], lower_is_better={"latency"}).verdicts[0].outcome is Outcome.REGRESSION
        assert (
            judge([better], lower_is_better={"latency"}).verdicts[0].outcome is Outcome.IMPROVEMENT
        )

    def test_identical_variants_pass_non_regression(self) -> None:
        flat = result(estimate=0.0, se=0.0, p=1.0, ci=(0.0, 0.0))
        assessment = judge([flat])
        assert assessment.verdicts[0].outcome is Outcome.NO_DETECTABLE_DIFFERENCE
        assert "identical" in assessment.verdicts[0].reason
        assert assessment.exit_code == 0

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            judge([])

    def test_duplicate_metrics_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            judge([result(), result()])

    def test_assessment_carries_configuration(self) -> None:
        assessment = judge([result()], alpha=0.01, min_effect=0.001, power=0.9)
        assert isinstance(assessment, Assessment)
        assert (assessment.alpha, assessment.min_effect, assessment.power) == (
            0.01,
            0.001,
            0.9,
        )
        assert assessment.gate is Gate.NON_REGRESSION
