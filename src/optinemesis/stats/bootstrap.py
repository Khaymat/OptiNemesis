"""Percentile bootstrap for paired or independent samples."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class BootstrapCI(NamedTuple):
    estimate: float
    low: float
    high: float
    se: float
    level: float
    resamples: int


def _check_inputs(n: int, resamples: int, level: float) -> None:
    if n < 1:
        raise ValueError("need at least one observation")
    if resamples < 100:
        raise ValueError("resamples must be >= 100")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    resamples: int = 10_000,
    level: float = 0.95,
    root_seed: int = 0,
) -> BootstrapCI:
    """Percentile CI for ``statistic(a, b)`` resampling instance indices jointly.

    Joint index resampling preserves pairing between the two configurations.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired bootstrap requires equal-length 1-D samples")
    _check_inputs(x.size, resamples, level)

    rng = np.random.default_rng(root_seed)
    estimate = float(statistic(x, y))
    indices = rng.integers(0, x.size, size=(resamples, x.size))
    replicates = np.array([statistic(x[idx], y[idx]) for idx in indices])
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(replicates, [100 * alpha, 100 * (1.0 - alpha)])
    se = float(np.std(replicates, ddof=1)) if resamples > 1 else 0.0
    return BootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        se=se,
        level=level,
        resamples=resamples,
    )


def independent_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    resamples: int = 10_000,
    level: float = 0.95,
    root_seed: int = 0,
) -> BootstrapCI:
    """Percentile CI resampling each configuration's sample independently."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    _check_inputs(max(x.size, y.size), resamples, level)
    rng = np.random.default_rng(root_seed)
    estimate = float(statistic(x, y))
    idx_x = rng.integers(0, x.size, size=(resamples, x.size))
    idx_y = rng.integers(0, y.size, size=(resamples, y.size))
    replicates = np.array([statistic(x[ix], y[iy]) for ix, iy in zip(idx_x, idx_y, strict=True)])
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(replicates, [100 * alpha, 100 * (1.0 - alpha)])
    se = float(np.std(replicates, ddof=1)) if resamples > 1 else 0.0
    return BootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        se=se,
        level=level,
        resamples=resamples,
    )
