from __future__ import annotations

import math
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noisecheck.stats.multiplicity import benjamini_hochberg

probabilities = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=20
)


class TestBenjaminiHochberg:
    def test_matches_a_hand_computed_example(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        expected = [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20]
        for got, want in zip(adjusted, expected, strict=True):
            assert math.isclose(got, want, rel_tol=1e-12)

    def test_single_p_value_is_unchanged(self) -> None:
        assert benjamini_hochberg([0.037]) == [0.037]

    def test_empty_input_gives_empty_output(self) -> None:
        assert benjamini_hochberg([]) == []

    def test_equal_p_values_stay_equal(self) -> None:
        assert benjamini_hochberg([0.05, 0.05]) == [0.05, 0.05]

    def test_rejects_values_outside_zero_one(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            benjamini_hochberg([0.5, 1.5])

    @given(probabilities)
    @settings(max_examples=80, deadline=None)
    def test_adjusted_values_never_shrink_and_stay_probabilities(self, values: list[float]) -> None:
        adjusted = benjamini_hochberg(values)
        for raw, q in zip(values, adjusted, strict=True):
            assert q >= raw
            assert q <= 1.0

    @given(probabilities)
    @settings(max_examples=80, deadline=None)
    def test_order_is_preserved(self, values: list[float]) -> None:
        adjusted = benjamini_hochberg(values)
        ranked = sorted(zip(values, adjusted, strict=True))
        for (_, q_small), (_, q_big) in pairwise(ranked):
            assert q_small <= q_big + 1e-15
