from __future__ import annotations

import pytest

from noisecheck import Dataset, PairingError, Record, pair


def rows(
    variant: str,
    example_ids: list[str],
    value: float,
    run_id: str | None = None,
    cluster: dict[str, str] | None = None,
) -> list[Record]:
    return [
        Record(
            example_id=ex,
            variant=variant,
            metric="accuracy",
            value=value,
            run_id=run_id,
            cluster_id=(cluster or {}).get(ex),
        )
        for ex in example_ids
    ]


def ids(n: int, prefix: str = "ex") -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


class TestPairing:
    def test_aligns_shared_examples_in_stable_order(self) -> None:
        ds = Dataset(rows("baseline", ["b", "a"], 0.0) + rows("candidate", ["a", "b"], 1.0))
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.example_ids == ("a", "b")
        assert paired.baseline == (0.0, 0.0)
        assert paired.candidate == (1.0, 1.0)
        assert paired.n == 2

    def test_counts_discarded_examples_on_both_sides(self) -> None:
        shared = ids(18)
        ds = Dataset(
            rows("baseline", [*shared, "onlyb1", "onlyb2"], 0.5)
            + rows("candidate", [*shared, "onlyc1"], 0.6)
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.n == 18
        assert paired.discarded_baseline == 2
        assert paired.discarded_candidate == 1

    def test_rejects_overlap_below_90_percent(self) -> None:
        shared = ids(89)
        ds = Dataset(
            rows("baseline", shared + ids(11, "extra"), 0.5) + rows("candidate", shared, 0.6)
        )
        with pytest.raises(PairingError, match="90"):
            pair(ds, "baseline", "candidate", "accuracy")

    def test_allows_overlap_at_exactly_90_percent(self) -> None:
        shared = ids(90)
        ds = Dataset(
            rows("baseline", shared + ids(10, "extra"), 0.5) + rows("candidate", shared, 0.6)
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.n == 90

    def test_allows_overlap_at_91_percent(self) -> None:
        shared = ids(91)
        ds = Dataset(
            rows("baseline", shared + ids(9, "extra"), 0.5) + rows("candidate", shared, 0.6)
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.n == 91

    def test_unknown_variant_raises(self) -> None:
        ds = Dataset(rows("baseline", ["a"], 1.0))
        with pytest.raises(PairingError, match="candidate"):
            pair(ds, "baseline", "candidate", "accuracy")

    def test_unknown_metric_raises(self) -> None:
        ds = Dataset(rows("baseline", ["a"], 1.0) + rows("candidate", ["a"], 1.0))
        with pytest.raises(PairingError, match="latency"):
            pair(ds, "baseline", "candidate", "latency")

    def test_variant_without_rows_for_metric_raises(self) -> None:
        ds = Dataset(
            [
                *rows("baseline", ["a"], 1.0),
                Record(example_id="a", variant="candidate", metric="latency", value=1.0),
            ]
        )
        with pytest.raises(PairingError, match="no rows"):
            pair(ds, "baseline", "candidate", "accuracy")

    def test_pairs_within_matching_runs(self) -> None:
        ds = Dataset(
            rows("baseline", ["a", "b"], 0.0, run_id="r1")
            + rows("baseline", ["a", "b"], 0.0, run_id="r2")
            + rows("candidate", ["a", "b"], 1.0, run_id="r1")
            + rows("candidate", ["a", "b"], 1.0, run_id="r2")
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.n == 4
        assert paired.run_ids == ("r1", "r2", "r1", "r2")
        assert paired.example_ids == ("a", "a", "b", "b")

    def test_carries_cluster_ids_through(self) -> None:
        cluster = {"a": "c1", "b": "c2"}
        ds = Dataset(
            rows("baseline", ["a", "b"], 0.0, cluster=cluster)
            + rows("candidate", ["a", "b"], 1.0, cluster=cluster)
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.cluster_ids == ("c1", "c2")

    def test_conflicting_cluster_ids_raise(self) -> None:
        ds = Dataset(
            rows("baseline", ["a"], 0.0, cluster={"a": "c1"})
            + rows("candidate", ["a"], 1.0, cluster={"a": "c2"})
        )
        with pytest.raises(PairingError, match="cluster"):
            pair(ds, "baseline", "candidate", "accuracy")

    def test_selects_only_the_requested_metric(self) -> None:
        ds = Dataset(
            rows("baseline", ["a"], 0.5)
            + rows("candidate", ["a"], 0.9)
            + [
                Record(example_id="a", variant="baseline", metric="latency", value=3.0),
                Record(example_id="a", variant="candidate", metric="latency", value=4.0),
            ]
        )
        paired = pair(ds, "baseline", "candidate", "accuracy")
        assert paired.baseline == (0.5,)
        assert paired.candidate == (0.9,)
