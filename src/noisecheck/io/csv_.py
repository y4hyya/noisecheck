"""Read eval records from csv files."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from noisecheck.errors import LoadError
from noisecheck.io._common import first_validation_issue, into_dataset
from noisecheck.schema import Dataset, Record

_REQUIRED_COLUMNS = ("example_id", "variant", "metric", "value")
_OPTIONAL_COLUMNS = ("cluster_id", "run_id")
_EXTRA_VALUES_KEY = "_extra"


def read_csv(path: str | Path) -> Dataset:
    """Read eval records from a csv file with a header row."""
    location = Path(path)
    try:
        with location.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, restkey=_EXTRA_VALUES_KEY)
            header = reader.fieldnames
            if header is None:
                raise LoadError(f"{location}: file is empty")
            _check_header(set(header), location)
            records = [_row_to_record(row, reader.line_num, location) for row in reader]
    except OSError as error:
        raise LoadError(f"{location}: {error}") from error
    return into_dataset(records, location)


def _check_header(columns: set[str], location: Path) -> None:
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise LoadError(f"{location}: missing required columns: {', '.join(missing)}")
    unexpected = sorted(columns - set(_REQUIRED_COLUMNS) - set(_OPTIONAL_COLUMNS))
    if unexpected:
        raise LoadError(
            f"{location}: unexpected columns: {', '.join(unexpected)}; expected "
            f"{', '.join(_REQUIRED_COLUMNS)} and optional {', '.join(_OPTIONAL_COLUMNS)}"
        )


def _row_to_record(row: Mapping[str, object], lineno: int, location: Path) -> Record:
    if _EXTRA_VALUES_KEY in row:
        raise LoadError(
            f"{location} line {lineno}: row has more values than the header has columns"
        )
    payload: dict[str, object] = {
        "example_id": row.get("example_id"),
        "variant": row.get("variant"),
        "metric": row.get("metric"),
        "value": _parse_number(row.get("value"), lineno, location),
    }
    for column in _OPTIONAL_COLUMNS:
        text = row.get(column)
        if isinstance(text, str) and text.strip():
            payload[column] = text
    try:
        return Record.model_validate(payload)
    except ValidationError as error:
        raise LoadError(f"{location} line {lineno}: {first_validation_issue(error)}") from error


def _parse_number(text: object, lineno: int, location: Path) -> float:
    if not isinstance(text, str) or not text.strip():
        raise LoadError(f"{location} line {lineno}: value is missing")
    try:
        return float(text)
    except ValueError as error:
        raise LoadError(f"{location} line {lineno}: value {text!r} is not a number") from error
