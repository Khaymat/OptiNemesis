"""Statistics: effect sizes, bootstrap CIs, tests, corrections."""

from optinemesis.stats.bootstrap import BootstrapCI, independent_bootstrap_ci, paired_bootstrap_ci
from optinemesis.stats.effects import (
    cliffs_delta_from_samples,
    iqr,
    median,
    median_gap,
    paired_rank_biserial,
    probability_superiority,
    summarize_regrets,
)
from optinemesis.stats.tests import (
    TestOutcome,
    holm_bonferroni,
    independent_mann_whitney,
    independent_permutation_test_median_gap,
    paired_permutation_test_median_gap,
    paired_wilcoxon,
    ranks_average,
)

__all__ = [
    "BootstrapCI",
    "TestOutcome",
    "cliffs_delta_from_samples",
    "holm_bonferroni",
    "independent_bootstrap_ci",
    "independent_mann_whitney",
    "independent_permutation_test_median_gap",
    "iqr",
    "median",
    "median_gap",
    "paired_bootstrap_ci",
    "paired_permutation_test_median_gap",
    "paired_rank_biserial",
    "paired_wilcoxon",
    "probability_superiority",
    "ranks_average",
    "summarize_regrets",
]
