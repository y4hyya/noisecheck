"""Read promptfoo eval output files into records."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TypeGuard, cast

from noisecheck.errors import LoadError
from noisecheck.io._common import into_dataset
from noisecheck.schema import Dataset, Record


def read_promptfoo(path: str | Path) -> Dataset:
    """Read a promptfoo eval output file, one record per test, provider, and metric."""
    location = Path(path)
    try:
        raw: object = json.loads(location.read_text(encoding="utf-8"))
    except OSError as error:
        raise LoadError(f"{location}: {error}") from error
    except json.JSONDecodeError as error:
        raise LoadError(f"{location}: invalid json: {error.msg}") from error

    items = _result_items(raw, location)
    parsed = [_parse_item(item, index, location) for index, item in enumerate(items)]
    pair_counts = Counter((variant, example) for variant, example, _ in parsed)
    with_runs = any(count > 1 for count in pair_counts.values())
    occurrence: Counter[tuple[str, str]] = Counter()
    records: list[Record] = []
    for variant, example, metrics in parsed:
        run_id: str | None = None
        if with_runs:
            run_id = f"r{occurrence[(variant, example)]}"
            occurrence[(variant, example)] += 1
        for metric, value in metrics:
            records.append(
                Record(
                    example_id=example,
                    variant=variant,
                    metric=metric,
                    value=value,
                    run_id=run_id,
                )
            )
    return into_dataset(records, location)


def _result_items(raw: object, location: Path) -> list[object]:
    if isinstance(raw, dict):
        payload = cast("Mapping[str, object]", raw)
        results = payload.get("results")
        if isinstance(results, list):
            return cast("list[object]", results)
        if isinstance(results, dict):
            nested = cast("Mapping[str, object]", results).get("results")
            if isinstance(nested, list):
                return cast("list[object]", nested)
    raise LoadError(f"{location}: does not look like promptfoo output, expected a results list")


def _parse_item(
    raw_item: object, index: int, location: Path
) -> tuple[str, str, list[tuple[str, float]]]:
    if not isinstance(raw_item, dict):
        raise LoadError(f"{location} results[{index}]: expected an object")
    item = cast("Mapping[str, object]", raw_item)
    variant = _variant(item.get("provider"), index, location)
    test_idx = item.get("testIdx")
    if isinstance(test_idx, bool) or not isinstance(test_idx, int):
        raise LoadError(f"{location} results[{index}]: missing testIdx")

    metrics: list[tuple[str, float]] = []
    success = item.get("success")
    if isinstance(success, bool):
        metrics.append(("pass", 1.0 if success else 0.0))
    score = item.get("score")
    if _is_number(score):
        metrics.append(("score", float(score)))
    named = item.get("namedScores")
    if isinstance(named, dict):
        scores = cast("Mapping[str, object]", named)
        for name in sorted(scores):
            if not name.strip():
                raise LoadError(f"{location} results[{index}]: blank metric name")
            value = scores[name]
            if _is_number(value):
                metrics.append((name, float(value)))
    latency = item.get("latencyMs")
    if _is_number(latency):
        metrics.append(("latency_ms", float(latency)))
    if not metrics:
        raise LoadError(f"{location} results[{index}]: no metrics found")
    return variant, f"t{test_idx}", metrics


def _variant(provider: object, index: int, location: Path) -> str:
    if isinstance(provider, str) and provider.strip():
        return provider
    if isinstance(provider, dict):
        payload = cast("Mapping[str, object]", provider)
        label = payload.get("label")
        if isinstance(label, str) and label.strip():
            return label
        identifier = payload.get("id")
        if isinstance(identifier, str) and identifier.strip():
            return identifier
    raise LoadError(f"{location} results[{index}]: missing provider")


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
