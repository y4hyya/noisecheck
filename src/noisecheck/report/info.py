"""Provenance carried by every report."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunInfo:
    """What was compared, with which inputs, seed, and package version."""

    baseline_path: str
    candidate_path: str
    baseline_sha256: str
    candidate_sha256: str
    package_version: str
    seed: int
    b: int
    level: float


def file_sha256(path: str | Path) -> str:
    """Content hash of an input file, so reports are traceable to exact data."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
