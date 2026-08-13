from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from noisecheck import DataError, PairedData, compare_paired
from noisecheck.stats.power import design_effect, effective_sample_size, icc
from noisecheck.stats.resample import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    cluster_sign_flip_p,
    sign_flip_p,
)


def clustered_deltas(
    n_clusters: int,
    cluster_size: int,
    between_sd: float,
    within_sd: float,
    seed: int = 11,
) -> tuple[NDArray[np.float64], list[str]]:
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, between_sd, n_clusters)
    values = np.repeat(effects, cluster_size) + rng.normal(
        0.0, within_sd, n_clusters * cluster_size
    )
    labels = [f"c{i // cluster_size}" for i in range(n_clusters * cluster_size)]
    return values, labels


def build(
    baseline: list[float],
    candidate: list[float],
    cluster_ids: tuple[str | None, ...] | None = None,
    example_ids: tuple[str, ...] | None = None,
) -> PairedData:
    n = len(baseline)
    return PairedData(
        metric="score",
        example_ids=example_ids or tuple(f"ex{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=cluster_ids or (None,) * n,
        baseline=tuple(baseline),
        candidate=tuple(candidate),
        discarded_baseline=0,
        discarded_candidate=0,
    )


class TestClusterEngines:
    def test_cluster_bootstrap_is_deterministic(self) -> None:
        values, labels = clustered_deltas(20, 5, 1.0, 0.3)
        first = cluster_bootstrap_ci(values, labels, b=400, seed=3)
        second = cluster_bootstrap_ci(values, labels, b=400, seed=3)
        assert first == second

    def test_clustered_interval_is_wider_when_clusters_dominate(self) -> None:
        values, labels = clustered_deltas(25, 8, 1.0, 0.05)
        lo_c, hi_c = cluster_bootstrap_ci(values, labels, b=800, seed=5)
        lo_i, hi_i = bootstrap_ci(values, b=800, seed=5)
        assert (hi_c - lo_c) > 1.8 * (hi_i - lo_i)

    def test_singleton_clusters_match_the_iid_engine(self) -> None:
        values = np.random.default_rng(7).normal(0.2, 1.0, 30)
        labels = [f"row{i}" for i in range(30)]
        lo_c, hi_c = cluster_bootstrap_ci(values, labels, b=300, seed=9)
        lo_i, hi_i = bootstrap_ci(values, b=300, seed=9)
        assert math.isclose(lo_c, lo_i, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(hi_c, hi_i, rel_tol=1e-9, abs_tol=1e-12)
        p_c = cluster_sign_flip_p(values, labels, b=299, seed=9)
        p_i = sign_flip_p(values, b=299, seed=9)
        assert p_c == p_i

    def test_cluster_sign_flip_detects_a_cluster_level_shift(self) -> None:
        values, labels = clustered_deltas(30, 4, 0.05, 0.02, seed=3)
        shifted = values + 1.0
        assert cluster_sign_flip_p(shifted, labels, b=999, seed=5) <= 2 / 1000

    def test_cluster_sign_flip_respects_the_add_one_floor(self) -> None:
        values, labels = clustered_deltas(10, 3, 1.0, 1.0)
        p = cluster_sign_flip_p(values, labels, b=99, seed=5)
        assert 1 / 100 <= p <= 1.0

    def test_engines_need_at_least_two_clusters(self) -> None:
        values = np.array([0.1, 0.4, 0.3])
        with pytest.raises(ValueError, match="at least 2 clusters"):
            cluster_bootstrap_ci(values, ["a", "a", "a"], b=100)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            cluster_sign_flip_p(values, ["a", "a", "a"], b=100)


class TestClusterDiagnostics:
    def test_icc_matches_a_hand_computed_example(self) -> None:
        values = np.array([1.0, 2.0, 5.0, 6.0])
        labels = ["a", "a", "b", "b"]
        assert math.isclose(icc(values, labels), 15.5 / 16.5, rel_tol=1e-12)

    def test_icc_is_near_zero_for_pure_noise(self) -> None:
        values, labels = clustered_deltas(30, 5, 0.0, 1.0, seed=2)
        assert abs(icc(values, labels)) < 0.2

    def test_icc_is_near_one_when_clusters_dominate(self) -> None:
        values, labels = clustered_deltas(25, 8, 1.0, 0.05)
        assert icc(values, labels) > 0.9

    def test_icc_is_zero_for_all_singletons(self) -> None:
        values = np.array([0.3, 0.9, 0.1])
        assert icc(values, ["a", "b", "c"]) == 0.0

    def test_icc_is_zero_for_constant_values(self) -> None:
        values = np.array([0.5, 0.5, 0.5, 0.5])
        assert icc(values, ["a", "a", "b", "b"]) == 0.0

    def test_negative_raw_icc_is_clipped_to_zero(self) -> None:
        values = np.array([0.0, 10.0, 0.1, 9.9])
        assert icc(values, ["a", "a", "b", "b"]) == 0.0

    def test_design_effect_and_effective_sample_size(self) -> None:
        deff = design_effect(icc_value=15.5 / 16.5, average_cluster_size=2.0)
        assert math.isclose(deff, 1.0 + 15.5 / 16.5)
        assert math.isclose(effective_sample_size(4, deff), 4.0 / (1.0 + 15.5 / 16.5))


class TestClusteredCompare:
    def test_clustered_data_reports_cluster_fields(self) -> None:
        values, labels = clustered_deltas(25, 8, 1.0, 0.05)
        data = build([0.0] * values.size, list(values), cluster_ids=tuple(labels))
        result = compare_paired(data, b=400, seed=5)
        assert result.n_clusters == 25
        assert result.icc is not None and result.icc > 0.9
        assert result.design_effect is not None and result.design_effect > 2.0
        assert result.effective_n is not None and result.effective_n < values.size / 2

    def test_iid_data_has_no_cluster_fields(self) -> None:
        result = compare_paired(build([0.1, 0.5, 0.9], [0.2, 0.7, 0.8]), b=100)
        assert result.n_clusters is None
        assert result.icc is None
        assert result.design_effect is None
        assert result.effective_n is None

    def test_ignoring_clusters_understates_the_interval(self) -> None:
        values, labels = clustered_deltas(25, 8, 1.0, 0.05)
        with_clusters = compare_paired(
            build([0.0] * values.size, list(values), cluster_ids=tuple(labels)), b=800, seed=5
        )
        without = compare_paired(build([0.0] * values.size, list(values)), b=800, seed=5)
        assert (with_clusters.ci_high - with_clusters.ci_low) > 1.8 * (
            without.ci_high - without.ci_low
        )

    def test_repeated_examples_without_clusters_warn(self) -> None:
        data = build(
            [0.1, 0.4, 0.2, 0.8],
            [0.3, 0.5, 0.4, 0.9],
            example_ids=("a", "a", "b", "b"),
        )
        result = compare_paired(data, b=100)
        assert any("independent" in w for w in result.warnings)

    def test_rows_without_cluster_id_become_their_own_cluster(self) -> None:
        values, labels = clustered_deltas(12, 3, 0.6, 0.2)
        ids: tuple[str | None, ...] = (None, *labels[1:])
        data = build([0.0] * values.size, list(values), cluster_ids=ids)
        result = compare_paired(data, b=200, seed=5)
        assert any("own cluster" in w for w in result.warnings)
        assert result.n_clusters == 13

    def test_few_clusters_warn(self) -> None:
        values, labels = clustered_deltas(5, 6, 0.5, 0.2)
        data = build([0.0] * values.size, list(values), cluster_ids=tuple(labels))
        result = compare_paired(data, b=200, seed=5)
        assert any("only 5 clusters" in w for w in result.warnings)

    def test_single_cluster_is_rejected(self) -> None:
        data = build([0.1, 0.2, 0.3], [0.4, 0.5, 0.9], cluster_ids=("c1", "c1", "c1"))
        with pytest.raises(DataError, match="single cluster"):
            compare_paired(data, b=100)

    def test_clustered_binary_data_flags_mcnemar(self) -> None:
        data = build(
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            cluster_ids=("c1", "c1", "c2", "c2", "c3", "c3"),
        )
        result = compare_paired(data, b=100)
        assert result.mcnemar_p is not None
        assert any("mcnemar" in w for w in result.warnings)

    def test_swapping_variants_mirrors_with_clusters(self) -> None:
        values, labels = clustered_deltas(15, 4, 0.8, 0.3)
        forward = compare_paired(
            build([0.0] * values.size, list(values), cluster_ids=tuple(labels)), b=300, seed=7
        )
        backward = compare_paired(
            build(list(values), [0.0] * values.size, cluster_ids=tuple(labels)), b=300, seed=7
        )
        assert math.isclose(forward.estimate, -backward.estimate, rel_tol=1e-9, abs_tol=1e-12)
        assert forward.p_value == backward.p_value
        assert math.isclose(forward.ci_low, -backward.ci_high, rel_tol=1e-7, abs_tol=1e-9)


finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
clustered_rows = st.lists(
    st.tuples(finite, finite, st.integers(min_value=0, max_value=3)),
    min_size=2,
    max_size=30,
)


class TestClusterProperties:
    @given(clustered_rows, st.permutations([0, 1, 2, 3]))
    @settings(max_examples=40, deadline=None)
    def test_relabeling_clusters_changes_nothing(
        self, rows: list[tuple[float, float, int]], relabel: list[int]
    ) -> None:
        labels = [f"c{j}" for _, _, j in rows]
        assume(len(set(labels)) >= 2)
        renamed = [f"r{relabel[j]}" for _, _, j in rows]
        baseline = [b for b, _, _ in rows]
        candidate = [c for _, c, _ in rows]
        original = compare_paired(
            build(baseline, candidate, cluster_ids=tuple(labels)), b=150, seed=5
        )
        rebuilt = compare_paired(
            build(baseline, candidate, cluster_ids=tuple(renamed)), b=150, seed=5
        )
        assert original == rebuilt
