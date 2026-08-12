"""Statistical verdicts for AI evals."""

from importlib.metadata import PackageNotFoundError, version

from noisecheck.errors import DataError, LoadError, NoisecheckError, PairingError
from noisecheck.schema import Dataset, PairedData, Record, pair

try:
    __version__ = version("noisecheck")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DataError",
    "Dataset",
    "LoadError",
    "NoisecheckError",
    "PairedData",
    "PairingError",
    "Record",
    "pair",
]
