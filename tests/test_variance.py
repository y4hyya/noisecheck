from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from noisecheck import DataError, Dataset, NoiseFloor, Record, noise_floor


def run_records(
    variant: str,
    metric: str,
    runs: dict[str, dict[str, float]],
) -> list[Record]:
    return [
        Record(
            example_id=example,
            variant=variant,
            metric=metric,
            value=value,
            run_id=run,
        )
        for run, values in runs.items()
        for example, value in values.items()
    ]


class TestNoiseFloor:
    def test_reports_run_means_and_the_floor(self) -> None:
        ds = Dataset(
            run_records(
                "baseline",
                "score",
                {
                    "r1": {"a": 0.4, "b": 0.6},
                    "r2": {"a": 0.5, "b": 0.7},
                    "r3": {"a": 0.45, "b": 0.65},
                },
            )
        )
        floor = noise_floor(ds, "baseline", "score")
        assert isinstance(floor, NoiseFloor)
        assert floor.n_runs == 3
        assert floor.run_means == (0.5, 0.6, 0.55)
        expected_sd = float(np.std([0.5, 0.6, 0.55], ddof=1))
        assert math.isclose(floor.run_sd, expected_sd, rel_tol=1e-12)
        expected_floor = float(scipy_stats.t.ppf(0.975, 2)) * expected_sd
        assert math.isclose(floor.floor, expected_floor, rel_tol=1e-12)
        assert floor.flip_rates is None

    def test_identical_runs_have_a_zero_floor(self) -> None:
        ds = Dataset(
            run_records(
                "baseline",
                "score",
                {"r1": {"a": 0.4, "b": 0.6}, "r2": {"a": 0.4, "b": 0.6}},
            )
        )
        floor = noise_floor(ds, "baseline", "score")
        assert floor.run_sd == 0.0
        assert floor.floor == 0.0

    def test_binary_metrics_report_the_top_flippers(self) -> None:
        ds = Dataset(
            run_records(
                "baseline",
                "pass",
                {
                    "r1": {"a": 1.0, "b": 1.0, "c": 1.0},
                    "r2": {"a": 1.0, "b": 0.0, "c": 1.0},
                    "r3": {"a": 0.0, "b": 1.0, "c": 1.0},
                    "r4": {"a": 1.0, "b": 0.0, "c": 1.0},
                },
            )
        )
        floor = noise_floor(ds, "baseline", "pass")
        assert floor.flip_rates == (("b", 0.5), ("a", 0.25))

    def test_needs_at_least_two_runs(self) -> None:
        ds = Dataset(run_records("baseline", "score", {"r1": {"a": 0.4}}))
        with pytest.raises(DataError, match="at least 2 runs"):
            noise_floor(ds, "baseline", "score")

    def test_rows_without_run_ids_are_rejected(self) -> None:
        ds = Dataset(
            [
                Record(example_id="a", variant="baseline", metric="score", value=0.4),
                Record(
                    example_id="a",
                    variant="baseline",
                    metric="score",
                    value=0.5,
                    run_id="r1",
                ),
            ]
        )
        with pytest.raises(DataError, match="no run id"):
            noise_floor(ds, "baseline", "score")

    def test_runs_covering_different_examples_are_rejected(self) -> None:
        ds = Dataset(
            run_records(
                "baseline",
                "score",
                {"r1": {"a": 0.4, "b": 0.6}, "r2": {"a": 0.5, "c": 0.7}},
            )
        )
        with pytest.raises(DataError, match="different examples"):
            noise_floor(ds, "baseline", "score")

    def test_unknown_variant_or_metric_rejected(self) -> None:
        ds = Dataset(run_records("baseline", "score", {"r1": {"a": 0.4}, "r2": {"a": 0.5}}))
        with pytest.raises(DataError, match="candidate"):
            noise_floor(ds, "candidate", "score")
        with pytest.raises(DataError, match="latency"):
            noise_floor(ds, "baseline", "latency")
