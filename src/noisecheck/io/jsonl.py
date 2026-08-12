"""Read eval records from jsonl files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from noisecheck.errors import LoadError
from noisecheck.io._common import first_validation_issue, into_dataset
from noisecheck.schema import Dataset, Record


def read_jsonl(path: str | Path) -> Dataset:
    """Read eval records from a jsonl file with one json object per line."""
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise LoadError(f"{location}: {error}") from error

    records: list[Record] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise LoadError(f"{location} line {lineno}: invalid json: {error.msg}") from error
        if not isinstance(raw, dict):
            raise LoadError(f"{location} line {lineno}: expected a json object")
        try:
            records.append(Record.model_validate(raw))
        except ValidationError as error:
            raise LoadError(f"{location} line {lineno}: {first_validation_issue(error)}") from error

    return into_dataset(records, location)
