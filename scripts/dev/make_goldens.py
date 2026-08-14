"""Generates golden fixtures from independent reference implementations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from noisecheck.stats.resample import bootstrap_ci, sign_flip_p


def main() -> None:
    rng = np.random.default_rng(20260813)
    deltas = [round(float(x), 6) for x in rng.normal(0.05, 0.3, 80)]
    welch_baseline = [round(float(x), 6) for x in rng.normal(0.5, 0.12, 25)]
    welch_candidate = [round(float(x), 6) for x in rng.normal(0.62, 0.2, 31)]

    mcnemar_cases = []
    for zero_to_one, one_to_zero, both_pass, both_fail in [
        (3, 1, 2, 2),
        (9, 3, 5, 4),
        (0, 5, 3, 3),
        (2, 17, 1, 2),
        (5, 5, 4, 4),
    ]:
        table = np.array([[both_pass, one_to_zero], [zero_to_one, both_fail]])
        expected = float(mcnemar(table, exact=True).pvalue)
        mcnemar_cases.append(
            {
                "zero_to_one": zero_to_one,
                "one_to_zero": one_to_zero,
                "both_pass": both_pass,
                "both_fail": both_fail,
                "expected_p": expected,
            }
        )

    means = CompareMeans(
        DescrStatsW(np.array(welch_candidate)), DescrStatsW(np.array(welch_baseline))
    )
    _, welch_p, _ = means.ttest_ind(usevar="unequal")
    welch_low, welch_high = means.tconfint_diff(alpha=0.05, usevar="unequal")

    bh_cases = []
    for p_values in [
        [0.01, 0.04, 0.03, 0.2],
        [0.5],
        [0.001, 0.001, 0.9],
        [0.04, 0.9, 0.05, 0.049],
    ]:
        _, adjusted, _, _ = multipletests(p_values, method="fdr_bh")
        bh_cases.append({"p_values": p_values, "expected_q": [float(q) for q in adjusted]})

    kappa_cases = []
    kappa_human = [0, 1, 2, 2, 1, 0, 2, 1, 0, 2, 1, 1]
    kappa_judge = [0, 1, 2, 1, 1, 0, 2, 2, 0, 2, 0, 1]
    binary_human = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    binary_judge = [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    for human, judged, weights in [
        (binary_human, binary_judge, None),
        (kappa_human, kappa_judge, None),
        (kappa_human, kappa_judge, "linear"),
        (kappa_human, kappa_judge, "quadratic"),
    ]:
        expected = float(cohen_kappa_score(human, judged, weights=weights))
        kappa_cases.append(
            {
                "human": human,
                "judge": judged,
                "weights": weights or "none",
                "expected_kappa": expected,
            }
        )

    delta_array = np.asarray(deltas)
    lock_low, lock_high = bootstrap_ci(delta_array, b=2000, seed=42)
    payload = {
        "benjamini_hochberg": bh_cases,
        "kappa": kappa_cases,
        "mcnemar": mcnemar_cases,
        "welch": {
            "baseline": welch_baseline,
            "candidate": welch_candidate,
            "expected_p": float(welch_p),
            "expected_ci_low": float(welch_low),
            "expected_ci_high": float(welch_high),
        },
        "paired_se": {
            "deltas": deltas,
            "expected_mean": float(delta_array.mean()),
            "expected_se": float(delta_array.std(ddof=1) / np.sqrt(delta_array.size)),
        },
        "locks": {
            "bootstrap_ci_low": lock_low,
            "bootstrap_ci_high": lock_high,
            "sign_flip_p": sign_flip_p(delta_array, b=999, seed=42),
        },
    }
    target = Path(__file__).parents[2] / "tests" / "golden" / "stats.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
