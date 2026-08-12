from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from noisecheck.errors import DataError, LoadError
from noisecheck.schema import Dataset, Record


def first_validation_issue(error: ValidationError) -> str:
    issue = error.errors()[0]
    field = ".".join(str(part) for part in issue["loc"]) or "record"
    return f"{field}: {issue['msg']}"


def into_dataset(records: list[Record], location: Path) -> Dataset:
    if not records:
        raise LoadError(f"{location}: no records found")
    try:
        return Dataset(records)
    except DataError as error:
        raise LoadError(f"{location}: {error}") from error
