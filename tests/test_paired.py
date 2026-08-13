from __future__ import annotations

import math

import pytest

from noisecheck import ComparisonResult, DataError, PairedData, compare_paired


def paired(
    baseline: list[float],
    candidate: list[float],
    metric: str = "accuracy",
) -> PairedData:
    n = len(baseline)
    return PairedData(
        metric=metric,
        example_ids=tuple(f"ex{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=(None,) * n,
        baseline=tuple(baseline),
        candidate=tuple(candidate),
        discarded_baseline=0,
        discarded_candidate=0,
    )


def noisy(n: int, shift: float) -> PairedData:
    baseline = [0.4 + 0.31 * ((i * 7919) % 13) / 13 for i in range(n)]
    candidate = [b + shift + 0.17 * ((i * 104729) % 11 - 5) / 11 for i, b in enumerate(baseline)]
    return paired(baseline, candidate)


class TestComparePaired:
    def test_reports_the_mean_improvement(self) -> None:
        result = compare_paired(paired([0.0, 0.2, 0.4, 0.6], [0.3, 0.5, 0.5, 0.9]))
        assert math.isclose(result.estimate, 0.25)
        assert result.n == 4
        assert 0.0 < result.p_value <= 1.0

    def test_result_is_deterministic(self) -> None:
        data = noisy(40, 0.1)
        assert compare_paired(data, b=500, seed=9) == compare_paired(data, b=500, seed=9)

    def test_needs_at_least_two_pairs(self) -> None:
        with pytest.raises(DataError, match="at least 2"):
            compare_paired(paired([1.0], [0.5]))

    def test_identical_variants_report_zero_and_p_one(self) -> None:
        values = [0.1, 0.5, 0.9, 0.3]
        result = compare_paired(paired(values, list(values)))
        assert result.estimate == 0.0
        assert (result.ci_low, result.ci_high) == (0.0, 0.0)
        assert result.p_value == 1.0
        assert any("identical" in w for w in result.warnings)

    def test_constant_shift_reports_degenerate_interval(self) -> None:
        baseline = [0.0, 0.5, 1.0, 0.25]
        candidate = [b + 0.25 for b in baseline]
        result = compare_paired(paired(baseline, candidate))
        assert (result.ci_low, result.ci_high) == (0.25, 0.25)
        assert result.se == 0.0
        assert any("zero variance" in w for w in result.warnings)
        assert 0.0 < result.p_value <= 1.0

    def test_binary_metric_gets_mcnemar(self) -> None:
        baseline = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        candidate = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        result = compare_paired(paired(baseline, candidate))
        assert result.mcnemar_p is not None
        assert math.isclose(result.mcnemar_p, 0.625)

    def test_continuous_metric_has_no_mcnemar(self) -> None:
        result = compare_paired(noisy(20, 0.05))
        assert result.mcnemar_p is None

    def test_identical_binary_variants_have_mcnemar_one(self) -> None:
        values = [0.0, 1.0, 0.0, 1.0]
        result = compare_paired(paired(values, list(values)))
        assert result.mcnemar_p == 1.0
        assert result.p_value == 1.0

    def test_small_sample_warning_lists_pair_count(self) -> None:
        result = compare_paired(noisy(10, 0.1))
        assert any("10 pairs" in w for w in result.warnings)

    def test_no_warnings_for_large_clean_data(self) -> None:
        result = compare_paired(noisy(100, 0.1))
        assert result.warnings == ()

    def test_carries_configuration(self) -> None:
        result = compare_paired(noisy(40, 0.1), b=777, seed=3, level=0.9)
        assert (result.b, result.seed, result.level) == (777, 3, 0.9)

    def test_higher_level_gives_wider_interval(self) -> None:
        data = noisy(60, 0.1)
        narrow = compare_paired(data, b=2000, seed=5, level=0.9)
        wide = compare_paired(data, b=2000, seed=5, level=0.99)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_result_type_is_exported(self) -> None:
        assert isinstance(compare_paired(noisy(5, 0.0)), ComparisonResult)
