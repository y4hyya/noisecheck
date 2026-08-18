from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy import stats as scipy_stats

from noisecheck import PairedData, compare_paired
from noisecheck.stats.resample import sign_flip_p

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
pair_rows = st.lists(st.tuples(finite, finite), min_size=2, max_size=40)


def build(rows: list[tuple[float, float]]) -> PairedData:
    n = len(rows)
    return PairedData(
        metric="score",
        example_ids=tuple(f"ex{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=(None,) * n,
        baseline=tuple(row[0] for row in rows),
        candidate=tuple(row[1] for row in rows),
        discarded_baseline=0,
        discarded_candidate=0,
    )


class TestProperties:
    @given(pair_rows)
    @settings(max_examples=60, deadline=None)
    def test_swapping_variants_flips_the_sign(self, rows: list[tuple[float, float]]) -> None:
        forward = compare_paired(build(rows), b=200, seed=7)
        backward = compare_paired(build([(c, b) for b, c in rows]), b=200, seed=7)
        assert math.isclose(forward.estimate, -backward.estimate, rel_tol=1e-9, abs_tol=1e-9)
        assert forward.p_value == backward.p_value
        assert math.isclose(forward.ci_low, -backward.ci_high, rel_tol=1e-7, abs_tol=1e-7)
        assert math.isclose(forward.ci_high, -backward.ci_low, rel_tol=1e-7, abs_tol=1e-7)

    @given(pair_rows, st.floats(min_value=0.05, max_value=1000.0))
    @settings(max_examples=60, deadline=None)
    @example(rows=[(0.0, 0.0), (0.0, -1.0), (5.960464477539063e-08, -1.0)], k=0.05)
    @example(rows=[(0.0, 0.0), (0.0, 1.0), (5.960464477539063e-08, 1.0)], k=0.05)
    def test_scaling_scales_estimate_and_interval(
        self, rows: list[tuple[float, float]], k: float
    ) -> None:
        plain = compare_paired(build(rows), b=200, seed=7)
        scaled = compare_paired(build([(b * k, c * k) for b, c in rows]), b=200, seed=7)
        magnitude = abs(scaled.estimate) + scaled.se + abs(scaled.ci_low) + abs(scaled.ci_high)
        tolerance = 1e-9 + 1e-8 * magnitude
        assert abs(scaled.estimate - plain.estimate * k) <= tolerance
        assert abs(scaled.ci_low - plain.ci_low * k) <= tolerance
        assert abs(scaled.ci_high - plain.ci_high * k) <= tolerance
        assert abs(scaled.p_value - plain.p_value) <= 2 / 201 + 1e-12

    @given(pair_rows)
    @settings(max_examples=60, deadline=None)
    def test_p_value_is_a_valid_probability_and_interval_is_ordered(
        self, rows: list[tuple[float, float]]
    ) -> None:
        result = compare_paired(build(rows), b=200, seed=7)
        assert 1 / 201 <= result.p_value <= 1.0
        assert result.ci_low <= result.ci_high

    @given(pair_rows)
    @settings(max_examples=30, deadline=None)
    def test_same_input_same_output(self, rows: list[tuple[float, float]]) -> None:
        assert compare_paired(build(rows), b=100, seed=3) == compare_paired(
            build(rows), b=100, seed=3
        )


@pytest.mark.slow
def test_sign_flip_p_is_uniform_under_the_null() -> None:
    rng = np.random.default_rng(123)
    seeds = rng.integers(0, 2**31, 150)
    ps = [sign_flip_p(rng.normal(0.0, 1.0, 40), b=999, seed=int(s)) for s in seeds]
    outcome = scipy_stats.kstest(ps, "uniform")
    assert outcome.pvalue > 0.001
