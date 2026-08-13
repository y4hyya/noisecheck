from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from noisecheck.stats.resample import bootstrap_ci, sign_flip_p


def normal_deltas(n: int, mean: float, sd: float, seed: int = 7) -> NDArray[np.float64]:
    return np.random.default_rng(seed).normal(mean, sd, n)


class TestBootstrapCi:
    def test_is_deterministic_for_a_seed(self) -> None:
        values = normal_deltas(60, 0.2, 1.0)
        assert bootstrap_ci(values, b=500, seed=11) == bootstrap_ci(values, b=500, seed=11)

    def test_different_seeds_give_different_intervals(self) -> None:
        values = normal_deltas(60, 0.2, 1.0)
        assert bootstrap_ci(values, b=500, seed=1) != bootstrap_ci(values, b=500, seed=2)

    def test_matches_t_interval_on_normal_data(self) -> None:
        values = normal_deltas(200, 0.5, 1.0)
        lo, hi = bootstrap_ci(values, b=4000, seed=11)
        sem = float(values.std(ddof=1) / np.sqrt(values.size))
        t_lo, t_hi = scipy_stats.t.interval(
            0.95, df=values.size - 1, loc=float(values.mean()), scale=sem
        )
        width = t_hi - t_lo
        assert abs(lo - t_lo) < 0.25 * width
        assert abs(hi - t_hi) < 0.25 * width

    def test_interval_bounds_are_ordered(self) -> None:
        values = normal_deltas(30, 0.0, 2.0)
        lo, hi = bootstrap_ci(values, b=800, seed=3)
        assert lo <= hi

    def test_rejects_constant_deltas(self) -> None:
        with pytest.raises(ValueError, match="variance"):
            bootstrap_ci(np.full(10, 0.3))

    def test_rejects_a_single_delta(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci(np.array([1.0]))


class TestSignFlipP:
    def test_is_deterministic_for_a_seed(self) -> None:
        values = normal_deltas(40, 0.1, 1.0)
        assert sign_flip_p(values, b=999, seed=5) == sign_flip_p(values, b=999, seed=5)

    def test_detects_a_strong_shift(self) -> None:
        values = normal_deltas(30, 1.0, 0.1)
        assert sign_flip_p(values, b=999, seed=5) <= 2 / 1000

    def test_is_one_for_all_zero_deltas(self) -> None:
        assert sign_flip_p(np.zeros(20), b=999, seed=5) == 1.0

    def test_never_goes_below_the_add_one_floor(self) -> None:
        values = normal_deltas(10, 0.0, 1.0)
        p = sign_flip_p(values, b=99, seed=5)
        assert 1 / 100 <= p <= 1.0

    def test_rejects_a_single_delta(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            sign_flip_p(np.array([1.0]))


class TestChunking:
    def test_large_inputs_run_in_chunks_and_stay_deterministic(self) -> None:
        values = normal_deltas(50_000, 0.1, 1.0)
        first = bootstrap_ci(values, b=64, seed=9)
        second = bootstrap_ci(values, b=64, seed=9)
        assert first == second
        assert np.isfinite(first).all()
        p_first = sign_flip_p(values, b=64, seed=9)
        p_second = sign_flip_p(values, b=64, seed=9)
        assert p_first == p_second
        assert 0.0 < p_first <= 1.0
