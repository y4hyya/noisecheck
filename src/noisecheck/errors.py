"""Errors raised by noisecheck."""

from __future__ import annotations


class NoisecheckError(Exception):
    """Base class for all noisecheck errors."""


class DataError(NoisecheckError):
    """Raised when records are invalid or inconsistent."""


class PairingError(NoisecheckError):
    """Raised when two variants cannot be paired safely."""


class LoadError(NoisecheckError):
    """Raised when a file cannot be read into records."""
