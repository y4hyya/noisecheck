from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from noisecheck import AgreementResult, DataError, PairedData, judge_agreement

FIXTURE = json.loads((Path(__file__).parent / "golden" / "stats.json").read_text())


def paired(
    human: list[float],
    judge: list[float],
    cluster_ids: tuple[str | None, ...] | None = None,
) -> PairedData:
    n = len(human)
    return PairedData(
        metric="winner",
        example_ids=tuple(f"ex{i:03d}" for i in range(n)),
        run_ids=(None,) * n,
        cluster_ids=cluster_ids or (None,) * n,
        baseline=tuple(human),
        candidate=tuple(judge),
        discarded_baseline=0,
        discarded_candidate=0,
    )


class TestJudgeAgreement:
    def test_matches_sklearn_goldens(self) -> None:
        for case in FIXTURE["kappa"]:
            result = judge_agreement(
                paired([float(v) for v in case["human"]], [float(v) for v in case["judge"]]),
                b=50,
                weights=case["weights"],
            )
            assert math.isclose(result.kappa, case["expected_kappa"], rel_tol=1e-12)

    def test_perfect_agreement_is_one(self) -> None:
        values = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        result = judge_agreement(paired(values, list(values)), b=50)
        assert result.kappa == 1.0
        assert result.observed_agreement == 1.0

    def test_chance_level_agreement_is_zero(self) -> None:
        result = judge_agreement(paired([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]), b=50)
        assert math.isclose(result.kappa, 0.0, abs_tol=1e-12)

    def test_one_label_everywhere_reports_zero_with_a_warning(self) -> None:
        result = judge_agreement(paired([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), b=50)
        assert result.kappa == 0.0
        assert (result.ci_low, result.ci_high) == (0.0, 0.0)
        assert any("label" in w for w in result.warnings)

    def test_confusion_counts_are_reported(self) -> None:
        result = judge_agreement(paired([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]), b=50)
        assert ("0", "0", 1) in result.confusion
        assert ("0", "1", 1) in result.confusion
        assert ("1", "1", 2) in result.confusion

    def test_worst_disagreements_name_examples(self) -> None:
        result = judge_agreement(paired([0.0, 2.0, 1.0], [0.0, 0.0, 2.0], cluster_ids=None), b=50)
        assert result.worst_disagreements[0] == "ex001"
        assert set(result.worst_disagreements) == {"ex001", "ex002"}

    def test_is_deterministic(self) -> None:
        data = paired([0.0, 1.0, 1.0, 0.0, 1.0, 0.0] * 5, [0.0, 1.0, 0.0, 0.0, 1.0, 1.0] * 5)
        assert judge_agreement(data, b=200, seed=9) == judge_agreement(data, b=200, seed=9)

    def test_clustered_interval_is_ordered(self) -> None:
        human = [0.0, 1.0, 1.0, 0.0] * 6
        judge = [0.0, 1.0, 0.0, 0.0] * 6
        clusters = tuple(f"c{i // 2}" for i in range(24))
        result = judge_agreement(paired(human, judge, cluster_ids=clusters), b=300, seed=3)
        assert result.ci_low <= result.ci_high
        assert isinstance(result, AgreementResult)

    def test_small_samples_warn(self) -> None:
        result = judge_agreement(paired([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]), b=50)
        assert any("4 pairs" in w for w in result.warnings)

    def test_unknown_weights_rejected(self) -> None:
        with pytest.raises(DataError, match="weights"):
            judge_agreement(paired([0.0, 1.0], [0.0, 1.0]), weights="cubic")
