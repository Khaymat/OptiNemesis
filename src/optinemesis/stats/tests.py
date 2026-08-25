"""Hypothesis-test wrappers and multiple-comparison correction.

Sensitive algorithms are delegated to SciPy; this module only standardizes
interfaces and adds the Holm step-down correction. All p-values returned here
are raw unless a function name says otherwise.
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import mannwhitneyu, permutation_test, rankdata, wilcoxon


def _permutation_rng_kwarg() -> str:
    params = inspect.signature(permutation_test).parameters
    return "rng" if "rng" in params else "random"


def _run_permutation_test(**kwargs: object) -> Any:
    kwname = _permutation_rng_kwarg()
    kwargs[kwname] = kwargs.pop("root_rng")
    return permutation_test(**kwargs)


class TestOutcome(NamedTuple):
    test_name: str
    statistic: float
    p_value: float
    n_used: int


def _as_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("expected non-empty 1-D sample")
    return arr


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> TestOutcome:
    """Two-sided Wilcoxon signed-rank test on matched pairs.

    Zero differences are dropped (SciPy ``zero_method="wilcox"``); an
    all-zero difference sample yields p = 1 by convention.
    """
    x = _as_1d(a)
    y = _as_1d(b)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal length")
    diff = x - y
    n_nonzero = int(np.sum(diff != 0.0))
    if n_nonzero == 0:
        return TestOutcome("wilcoxon_signed_rank", 0.0, 1.0, x.size)
    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return TestOutcome("wilcoxon_signed_rank", float(stat), float(p), x.size)


def paired_permutation_test_median_gap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    root_seed: int = 0,
) -> TestOutcome:
    """Sign-flip permutation test for median(a) - median(b) under H0: gap <=? symmetric.

    Uses within-pair sign flips (exact exchangeability under a symmetric null),
    two-sided.
    """
    x = _as_1d(a)
    y = _as_1d(b)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal length")

    def statistic(x_: np.ndarray, y_: np.ndarray, axis: int = -1) -> float:
        return float(np.median(x_, axis=axis) - np.median(y_, axis=axis))

    result = _run_permutation_test(
        data=(x, y),
        statistic=statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="two-sided",
        root_rng=np.random.default_rng(root_seed),
        vectorized=False,
    )
    return TestOutcome(
        "paired_signflip_median",
        float(result.statistic),
        float(result.pvalue),
        x.size,
    )


def independent_mann_whitney(a: np.ndarray, b: np.ndarray) -> TestOutcome:
    """Two-sided Mann-Whitney U test for independent samples."""
    x = _as_1d(a)
    y = _as_1d(b)
    result = mannwhitneyu(x, y, alternative="two-sided")
    total_n = x.size + y.size
    return TestOutcome(
        "mannwhitney_u", float(result.statistic), float(result.pvalue), int(total_n)
    )


def independent_permutation_test_median_gap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    root_seed: int = 0,
) -> TestOutcome:
    """Randomization test permuting group labels for the median gap."""
    x = _as_1d(a)
    y = _as_1d(b)

    def statistic(x_: np.ndarray, y_: np.ndarray, axis: int = -1) -> float:
        return float(np.median(x_, axis=axis) - np.median(y_, axis=axis))

    result = _run_permutation_test(
        data=(x, y),
        statistic=statistic,
        permutation_type="independent",
        n_resamples=n_resamples,
        alternative="two-sided",
        root_rng=np.random.default_rng(root_seed),
        vectorized=False,
    )
    return TestOutcome(
        "independent_permutation_median",
        float(result.statistic),
        float(result.pvalue),
        x.size + y.size,
    )


def holm_bonferroni(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control).

    Adjusted p_(i) = max over j<=i of min(1, (m - j + 1) * p_(j)), enforcing
    monotonicity across ordered p-values.
    """
    raw = np.asarray(p_values, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("need a non-empty 1-D array of p-values")
    if np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    m = raw.size
    order = np.argsort(raw, kind="stable")
    sorted_p = raw[order]
    adjusted_sorted = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    running_max = np.maximum.accumulate(adjusted_sorted)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = running_max
    return adjusted


def rank_biserial_from_wilcoxon_outcome(outcome: TestOutcome, n_pairs: int) -> float:
    """Recover the signed rank-biserial from a one-sided-style W statistic.

    Provided for cross-checking; production code computes effects directly via
    ``stats.effects.paired_rank_biserial``.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    total_mass = n_pairs * (n_pairs + 1) / 2.0
    return float(2.0 * outcome.statistic / total_mass - 1.0)


def ranks_average(values: np.ndarray) -> np.ndarray:
    """Expose SciPy average ranking for reference tests."""
    return rankdata(np.asarray(values, dtype=float))
