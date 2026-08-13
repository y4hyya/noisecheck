"""Benjamini Hochberg control of the false discovery rate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Adjusted q values that stay honest when many metrics are tested at once."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        return []
    if not bool(((values >= 0.0) & (values <= 1.0)).all()):
        raise ValueError("p values must be between 0 and 1")
    m = values.size
    order = np.argsort(values, kind="stable")
    ranked = values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    adjusted = np.maximum(adjusted, values[order])
    result = np.empty(m)
    result[order] = adjusted
    return [float(q) for q in result]
