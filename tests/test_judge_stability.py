from __future__ import annotations

import math

import pytest

from noisecheck import Dataset, Record, StabilityResult, judge_stability


def run_records(runs: dict[str, dict[str, float]], metric: str = "pass") -> Dataset:
    return Dataset(
        [
            Record(
                example_id=example,
                variant="judge",
                metric=metric,
                value=value,
                run_id=run,
            )
            for run, values in runs.items()
            for example, value in values.items()
        ]
    )


class TestJudgeStability:
    def test_reports_the_overall_flip_rate(self) -> None:
        ds = run_records(
            {
                "r1": {"a": 1.0, "b": 1.0, "c": 1.0},
                "r2": {"a": 1.0, "b": 0.0, "c": 1.0},
                "r3": {"a": 0.0, "b": 1.0, "c": 1.0},
                "r4": {"a": 1.0, "b": 0.0, "c": 1.0},
            }
        )
        result = judge_stability(ds, "judge", "pass")
        assert isinstance(result, StabilityResult)
        assert result.n_runs == 4
        assert result.n_examples == 3
        assert result.flip_rate is not None
        assert math.isclose(result.flip_rate, 0.25)
        assert result.flip_rates == (("b", 0.5), ("a", 0.25))
        assert any("only 4 runs" in w for w in result.warnings)

    def test_continuous_scores_report_spread_instead_of_flips(self) -> None:
        ds = run_records(
            {
                "r1": {"a": 0.4, "b": 0.6},
                "r2": {"a": 0.5, "b": 0.7},
                "r3": {"a": 0.45, "b": 0.65},
                "r4": {"a": 0.42, "b": 0.61},
                "r5": {"a": 0.5, "b": 0.66},
            },
            metric="score",
        )
        result = judge_stability(ds, "judge", "score")
        assert result.flip_rate is None
        assert result.flip_rates is None
        assert result.run_sd > 0.0
        assert result.floor > 0.0
        assert result.warnings == ()

    def test_perfectly_stable_judge(self) -> None:
        ds = run_records(
            {
                "r1": {"a": 1.0, "b": 0.0},
                "r2": {"a": 1.0, "b": 0.0},
                "r3": {"a": 1.0, "b": 0.0},
                "r4": {"a": 1.0, "b": 0.0},
                "r5": {"a": 1.0, "b": 0.0},
            }
        )
        result = judge_stability(ds, "judge", "pass")
        assert result.flip_rate == 0.0
        assert result.floor == 0.0

    def test_needs_runs(self) -> None:
        ds = run_records({"r1": {"a": 1.0, "b": 0.0}})
        with pytest.raises(Exception, match="at least 2 runs"):
            judge_stability(ds, "judge", "pass")
