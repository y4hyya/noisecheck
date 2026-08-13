"""Statistical verdicts for AI evals."""

from importlib.metadata import PackageNotFoundError, version

from noisecheck.errors import DataError, LoadError, NoisecheckError, PairingError
from noisecheck.io.csv_ import read_csv
from noisecheck.io.jsonl import read_jsonl
from noisecheck.io.promptfoo import read_promptfoo
from noisecheck.schema import Dataset, PairedData, Record, pair
from noisecheck.stats.multiplicity import benjamini_hochberg
from noisecheck.stats.paired import ComparisonResult, compare_paired
from noisecheck.stats.power import items_needed, mde
from noisecheck.stats.unpaired import UnpairedResult, compare_unpaired
from noisecheck.stats.variance import NoiseFloor, noise_floor
from noisecheck.verdict import Assessment, Gate, MetricVerdict, Outcome, judge

try:
    __version__ = version("noisecheck")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Assessment",
    "ComparisonResult",
    "DataError",
    "Dataset",
    "Gate",
    "LoadError",
    "MetricVerdict",
    "NoiseFloor",
    "NoisecheckError",
    "Outcome",
    "PairedData",
    "PairingError",
    "Record",
    "UnpairedResult",
    "benjamini_hochberg",
    "compare_paired",
    "compare_unpaired",
    "items_needed",
    "judge",
    "mde",
    "noise_floor",
    "pair",
    "read_csv",
    "read_jsonl",
    "read_promptfoo",
]
