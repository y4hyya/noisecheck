from __future__ import annotations

import math

import pytest

from noisecheck import DataError, PairedData, PositionResult, judge_position


def swapped_runs(original: list[float], swapped: list[float]) -> PairedData:
    n = len(original)
    return PairedData(
        metric="candidate_preferred",
        example_ids=tuple(f"pair{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=(None,) * n,
        baseline=tuple(original),
        candidate=tuple(swapped),
        discarded_baseline=0,
        discarded_candidate=0,
    )


class TestJudgePosition:
    def test_measures_a_known_position_bias(self) -> None:
        original = [1.0] * 8 + [0.0] * 12
        swapped = [1.0] * 8 + [1.0] * 6 + [0.0] * 6
        result = judge_position(swapped_runs(original, swapped), b=500, seed=7)
        assert isinstance(result, PositionResult)
        assert math.isclose(result.bias, 0.3)
        assert math.isclose(result.flip_rate, 0.3)
        assert result.mcnemar_p is not None
        assert math.isclose(result.mcnemar_p, 0.03125)
        assert 0.0 < result.p_value <= 1.0

    def test_a_fair_judge_shows_no_bias(self) -> None:
        verdicts = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        result = judge_position(swapped_runs(verdicts, list(verdicts)), b=200)
        assert result.bias == 0.0
        assert result.flip_rate == 0.0
        assert result.p_value == 1.0
        assert result.mcnemar_p == 1.0

    def test_is_deterministic(self) -> None:
        data = swapped_runs([1.0, 0.0, 1.0, 0.0] * 5, [1.0, 1.0, 1.0, 0.0] * 5)
        assert judge_position(data, b=300, seed=5) == judge_position(data, b=300, seed=5)

    def test_non_binary_verdicts_rejected(self) -> None:
        with pytest.raises(DataError, match="binary"):
            judge_position(swapped_runs([0.5, 1.0], [1.0, 0.0]))
