from __future__ import annotations

import numpy as np
import pytest

from noisecheck.stats.resample import bootstrap_ci, cluster_bootstrap_ci


def coverage(clustered: bool, icc: float, sims: int = 300, n: int = 100) -> float:
    rng = np.random.default_rng(77)
    labels = [f"c{i // 5}" for i in range(n)]
    hits = 0
    for sim in range(sims):
        effects = np.repeat(rng.normal(0.0, np.sqrt(icc), n // 5), 5) if icc > 0.0 else np.zeros(n)
        values = 0.3 + effects + rng.normal(0.0, np.sqrt(1.0 - icc), n)
        if clustered:
            low, high = cluster_bootstrap_ci(values, labels, b=500, seed=1000 + sim)
        else:
            low, high = bootstrap_ci(values, b=500, seed=1000 + sim)
        hits += int(low <= 0.3 <= high)
    return hits / sims


@pytest.mark.slow
def test_clustered_interval_holds_where_the_naive_one_fails() -> None:
    assert coverage(clustered=True, icc=0.5) >= 0.90
    assert coverage(clustered=False, icc=0.5) <= 0.85


@pytest.mark.slow
def test_iid_interval_covers_on_independent_data() -> None:
    assert 0.91 <= coverage(clustered=False, icc=0.0) <= 0.985
