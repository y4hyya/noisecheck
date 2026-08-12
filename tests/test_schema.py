from __future__ import annotations

import pytest
from pydantic import ValidationError

from noisecheck import DataError, Dataset, Record


def record(**overrides: object) -> Record:
    payload: dict[str, object] = {
        "example_id": "ex1",
        "variant": "baseline",
        "metric": "accuracy",
        "value": 1.0,
    }
    payload.update(overrides)
    return Record.model_validate(payload)


class TestRecord:
    def test_holds_one_metric_value_for_one_example(self) -> None:
        r = Record(
            example_id="q17",
            variant="baseline",
            metric="task_success",
            value=1.0,
            cluster_id="conv4",
            run_id="r1",
            meta={"model": "m"},
        )
        assert r.example_id == "q17"
        assert r.value == 1.0
        assert r.cluster_id == "conv4"
        assert r.run_id == "r1"

    def test_optional_fields_default_to_none(self) -> None:
        r = record()
        assert r.cluster_id is None
        assert r.run_id is None
        assert r.meta is None

    def test_rejects_blank_example_id(self) -> None:
        with pytest.raises(ValidationError):
            record(example_id="  ")

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            record(surprise=1)

    def test_accepts_boolean_value_as_number(self) -> None:
        assert record(value=True).value == 1.0
        assert record(value=False).value == 0.0


class TestDataset:
    def test_lists_variants_and_metrics(self) -> None:
        ds = Dataset(
            [
                record(),
                record(variant="candidate"),
                record(metric="latency", value=0.4),
            ]
        )
        assert ds.variants() == {"baseline", "candidate"}
        assert ds.metrics() == {"accuracy", "latency"}

    def test_rejects_duplicate_rows(self) -> None:
        with pytest.raises(DataError, match="duplicate"):
            Dataset([record(), record()])

    def test_same_example_in_different_runs_is_not_a_duplicate(self) -> None:
        ds = Dataset([record(run_id="r1"), record(run_id="r2")])
        assert len(ds) == 2

    def test_binary_metric_detected(self) -> None:
        ds = Dataset([record(value=0.0), record(example_id="ex2", value=1.0)])
        assert ds.is_binary("accuracy") is True

    def test_score_metric_is_not_binary(self) -> None:
        ds = Dataset([record(value=0.7)])
        assert ds.is_binary("accuracy") is False

    def test_unknown_metric_raises(self) -> None:
        ds = Dataset([record()])
        with pytest.raises(DataError, match="unknown metric"):
            ds.is_binary("latency")
