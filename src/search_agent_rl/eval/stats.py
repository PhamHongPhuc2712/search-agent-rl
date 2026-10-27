from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling questions."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_resamples, len(data)))
    means = data[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
