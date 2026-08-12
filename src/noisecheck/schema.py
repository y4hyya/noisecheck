"""Eval data model: records, datasets, and safe pairing of two variants."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from noisecheck.errors import DataError, PairingError

_MIN_OVERLAP_PERCENT = 90

_PairKey = tuple[str, str | None]


class Record(BaseModel):
    """One metric value for one example under one variant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    variant: str
    metric: str
    value: float
    cluster_id: str | None = None
    run_id: str | None = None
    meta: dict[str, Any] | None = None

    @field_validator("example_id", "variant", "metric")
    @classmethod
    def _reject_blank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("must not be blank")
        return text

    @field_validator("value", mode="before")
    @classmethod
    def _bool_to_number(cls, value: object) -> object:
        if isinstance(value, bool):
            return float(value)
        return value


class Dataset:
    """A validated, immutable collection of eval records."""

    def __init__(self, records: Iterable[Record]) -> None:
        self._records = tuple(records)
        seen: set[tuple[str, str, str, str | None]] = set()
        for r in self._records:
            key = (r.example_id, r.variant, r.metric, r.run_id)
            if key in seen:
                raise DataError(
                    f"duplicate record for example {r.example_id!r}, variant {r.variant!r}, "
                    f"metric {r.metric!r}, run {r.run_id!r}"
                )
            seen.add(key)

    @property
    def records(self) -> tuple[Record, ...]:
        """All records in insertion order."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def variants(self) -> frozenset[str]:
        """Names of every variant present in the data."""
        return frozenset(r.variant for r in self._records)

    def metrics(self) -> frozenset[str]:
        """Names of every metric present in the data."""
        return frozenset(r.metric for r in self._records)

    def is_binary(self, metric: str) -> bool:
        """Whether every value of this metric is 0 or 1."""
        values = [r.value for r in self._records if r.metric == metric]
        if not values:
            raise DataError(f"unknown metric {metric!r}")
        return all(v in (0.0, 1.0) for v in values)


@dataclass(frozen=True)
class PairedData:
    """Two variants aligned on shared examples and runs for one metric."""

    metric: str
    example_ids: tuple[str, ...]
    run_ids: tuple[str | None, ...]
    cluster_ids: tuple[str | None, ...]
    baseline: tuple[float, ...]
    candidate: tuple[float, ...]
    discarded_baseline: int
    discarded_candidate: int

    @property
    def n(self) -> int:
        """Number of paired observations."""
        return len(self.example_ids)


def pair(dataset: Dataset, baseline: str, candidate: str, metric: str) -> PairedData:
    """Align two variants on shared examples and runs for one metric.

    Refuses to pair when either side would lose more than 10 percent of its rows,
    because silent partial pairing biases the comparison.
    """
    known_variants = dataset.variants()
    for name in (baseline, candidate):
        if name not in known_variants:
            raise PairingError(f"variant {name!r} is not present in the data")
    if metric not in dataset.metrics():
        raise PairingError(f"metric {metric!r} is not present in the data")

    base_rows = _rows_by_key(dataset, baseline, metric)
    cand_rows = _rows_by_key(dataset, candidate, metric)
    for name, variant_rows in ((baseline, base_rows), (candidate, cand_rows)):
        if not variant_rows:
            raise PairingError(f"variant {name!r} has no rows for metric {metric!r}")

    shared = sorted(base_rows.keys() & cand_rows.keys(), key=_sort_key)
    for name, variant_rows in ((baseline, base_rows), (candidate, cand_rows)):
        if not _has_sufficient_overlap(len(shared), len(variant_rows)):
            raise PairingError(
                f"only {len(shared)} of {len(variant_rows)} rows for variant {name!r} have a "
                f"partner; pairing requires at least {_MIN_OVERLAP_PERCENT} percent overlap "
                f"on both sides"
            )

    cluster_ids: list[str | None] = []
    for key in shared:
        base_cluster = base_rows[key].cluster_id
        cand_cluster = cand_rows[key].cluster_id
        if base_cluster != cand_cluster:
            raise PairingError(
                f"example {key[0]!r} has cluster {base_cluster!r} under {baseline!r} "
                f"but cluster {cand_cluster!r} under {candidate!r}"
            )
        cluster_ids.append(base_cluster)

    return PairedData(
        metric=metric,
        example_ids=tuple(key[0] for key in shared),
        run_ids=tuple(key[1] for key in shared),
        cluster_ids=tuple(cluster_ids),
        baseline=tuple(base_rows[key].value for key in shared),
        candidate=tuple(cand_rows[key].value for key in shared),
        discarded_baseline=len(base_rows) - len(shared),
        discarded_candidate=len(cand_rows) - len(shared),
    )


def _rows_by_key(dataset: Dataset, variant: str, metric: str) -> dict[_PairKey, Record]:
    return {
        (r.example_id, r.run_id): r
        for r in dataset.records
        if r.variant == variant and r.metric == metric
    }


def _sort_key(key: _PairKey) -> tuple[str, str]:
    return (key[0], key[1] or "")


def _has_sufficient_overlap(shared: int, total: int) -> bool:
    return shared * 100 >= total * _MIN_OVERLAP_PERCENT
