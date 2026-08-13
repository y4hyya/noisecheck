from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noisecheck.stats.power import items_needed, mde


class TestMde:
    def test_matches_the_normal_approximation(self) -> None:
        assert math.isclose(mde(0.1), 0.28016, rel_tol=1e-3)

    def test_stricter_alpha_needs_a_bigger_effect(self) -> None:
        assert mde(0.1, alpha=0.01) > mde(0.1, alpha=0.05)

    def test_more_power_needs_a_bigger_effect(self) -> None:
        assert mde(0.1, power=0.9) > mde(0.1, power=0.8)

    def test_scales_linearly_with_the_standard_error(self) -> None:
        assert math.isclose(mde(0.2), 2 * mde(0.1), rel_tol=1e-12)

    def test_rejects_nonpositive_se(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            mde(0.0)


class TestItemsNeeded:
    def test_half_the_target_needs_four_times_the_items(self) -> None:
        current = mde(0.1)
        assert items_needed(0.1, 100, current / 2) == 400

    def test_target_already_reachable_needs_no_more(self) -> None:
        assert items_needed(0.1, 100, mde(0.1) * 2) <= 100

    def test_rejects_nonpositive_target(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            items_needed(0.1, 100, 0.0)

    def test_rejects_nonpositive_se(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            items_needed(0.0, 100, 0.05)

    @given(
        st.floats(min_value=0.01, max_value=10.0),
        st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(max_examples=50, deadline=None)
    def test_smaller_targets_never_need_fewer_items(self, se: float, target: float) -> None:
        assert items_needed(se, 50, target / 2) >= items_needed(se, 50, target)
