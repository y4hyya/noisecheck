from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from noisecheck import PairedData, compare_paired, compare_unpaired
from noisecheck.stats.resample import bootstrap_ci, sign_flip_p

FIXTURE = json.loads((Path(__file__).parent / "golden" / "stats.json").read_text())


def binary_paired(case: dict[str, Any]) -> PairedData:
    baseline: list[float] = []
    candidate: list[float] = []
    for count, base_value, cand_value in [
        (case["zero_to_one"], 0.0, 1.0),
        (case["one_to_zero"], 1.0, 0.0),
        (case["both_pass"], 1.0, 1.0),
        (case["both_fail"], 0.0, 0.0),
    ]:
        baseline.extend([base_value] * count)
        candidate.extend([cand_value] * count)
    n = len(baseline)
    return PairedData(
        metric="pass",
        example_ids=tuple(f"ex{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=(None,) * n,
        baseline=tuple(baseline),
        candidate=tuple(candidate),
        discarded_baseline=0,
        discarded_candidate=0,
    )


class TestAgainstReferences:
    def test_mcnemar_matches_statsmodels_exact(self) -> None:
        for case in FIXTURE["mcnemar"]:
            result = compare_paired(binary_paired(case), b=50)
            assert result.mcnemar_p is not None
            assert math.isclose(result.mcnemar_p, case["expected_p"], rel_tol=1e-12)

    def test_welch_matches_statsmodels(self) -> None:
        spec = FIXTURE["welch"]
        result = compare_unpaired(spec["baseline"], spec["candidate"])
        assert math.isclose(result.p_value, spec["expected_p"], rel_tol=1e-9)
        assert math.isclose(result.ci_low, spec["expected_ci_low"], rel_tol=1e-9)
        assert math.isclose(result.ci_high, spec["expected_ci_high"], rel_tol=1e-9)

    def test_paired_mean_and_se_match_numpy(self) -> None:
        spec = FIXTURE["paired_se"]
        deltas = spec["deltas"]
        data = PairedData(
            metric="score",
            example_ids=tuple(f"ex{i:03d}" for i in range(len(deltas))),
            run_ids=(None,) * len(deltas),
            cluster_ids=(None,) * len(deltas),
            baseline=(0.0,) * len(deltas),
            candidate=tuple(deltas),
            discarded_baseline=0,
            discarded_candidate=0,
        )
        result = compare_paired(data, b=50)
        assert math.isclose(result.estimate, spec["expected_mean"], rel_tol=1e-12)
        assert math.isclose(result.se, spec["expected_se"], rel_tol=1e-12)


class TestRegressionLocks:
    def test_bootstrap_interval_is_locked(self) -> None:
        locks = FIXTURE["locks"]
        low, high = bootstrap_ci(FIXTURE["paired_se"]["deltas"], b=2000, seed=42)
        assert math.isclose(low, locks["bootstrap_ci_low"], rel_tol=1e-12)
        assert math.isclose(high, locks["bootstrap_ci_high"], rel_tol=1e-12)

    def test_sign_flip_p_is_locked(self) -> None:
        p = sign_flip_p(FIXTURE["paired_se"]["deltas"], b=999, seed=42)
        assert math.isclose(p, FIXTURE["locks"]["sign_flip_p"], rel_tol=1e-12)
