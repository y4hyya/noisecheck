from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from noisecheck import DataError, UnpairedResult, compare_unpaired


def sample(n: int, mean: float, sd: float, seed: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(mean, sd, n)


class TestCompareUnpaired:
    def test_detects_a_clear_separation(self) -> None:
        result = compare_unpaired(sample(40, 0.3, 0.05, 1), sample(35, 0.8, 0.05, 2))
        assert result.p_value < 1e-6
        assert math.isclose(result.estimate, 0.5, abs_tol=0.05)
        assert result.ci_low < 0.5 < result.ci_high

    def test_identical_samples_give_p_one(self) -> None:
        values = sample(30, 0.5, 0.1, 3)
        result = compare_unpaired(values, values.copy())
        assert result.estimate == 0.0
        assert result.p_value == 1.0

    def test_reports_the_mean_difference(self) -> None:
        result = compare_unpaired([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
        assert math.isclose(result.estimate, 1.0)
        assert result.ci_low <= result.ci_high
        assert result.n_baseline == 4
        assert result.n_candidate == 4

    def test_needs_two_values_per_side(self) -> None:
        with pytest.raises(DataError, match="at least 2"):
            compare_unpaired([1.0], [1.0, 2.0])

    def test_two_constant_sides_rejected(self) -> None:
        with pytest.raises(DataError, match="zero variance"):
            compare_unpaired([0.5, 0.5, 0.5], [0.7, 0.7])

    def test_one_constant_side_is_fine_but_noted(self) -> None:
        result = compare_unpaired([0.5, 0.5, 0.5], [0.4, 0.7, 0.9, 0.6])
        assert 0.0 < result.p_value <= 1.0
        assert any("precision" in w for w in result.warnings)

    def test_always_warns_that_pairing_is_better(self) -> None:
        result = compare_unpaired([1.0, 2.0, 3.0], [2.0, 4.0, 5.0])
        assert any("unpaired" in w for w in result.warnings)

    def test_carries_level_and_type(self) -> None:
        result = compare_unpaired([1.0, 2.0, 3.0], [2.0, 4.0, 5.0], level=0.9)
        assert isinstance(result, UnpairedResult)
        assert result.level == 0.9
