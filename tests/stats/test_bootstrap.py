import numpy as np
import pytest

from optinemesis.stats import (
    BootstrapCI,
    independent_bootstrap_ci,
    median_gap,
    paired_bootstrap_ci,
    paired_rank_biserial,
)


class TestPairedBootstrap:
    def test_recovers_true_positive_gap(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 1.0, size=200)
        a = base + 1.0
        b = base
        ci = paired_bootstrap_ci(
            a,
            b,
            median_gap,
            resamples=2000,
            root_seed=0,
        )
        assert isinstance(ci, BootstrapCI)
        assert ci.low > 0.5
        assert ci.high < 1.5
        assert ci.estimate == pytest.approx(1.0)
        assert ci.level == 0.95

    def test_zero_gap_ci_contains_zero(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, size=150)
        b = rng.normal(0.0, 1.0, size=150)
        ci = paired_bootstrap_ci(a, b, median_gap, resamples=1000, root_seed=5)
        assert ci.low <= 0.0 <= ci.high

    def test_deterministic_given_root_seed(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.normal(size=80) + 0.2
        b = rng.normal(size=80)
        c1 = paired_bootstrap_ci(a, b, paired_rank_biserial, resamples=500, root_seed=9)
        c2 = paired_bootstrap_ci(a, b, paired_rank_biserial, resamples=500, root_seed=9)
        assert (c1.low, c1.high, c1.se) == (c2.low, c2.high, c2.se)

    def test_pairing_preserved_under_joint_resampling(self) -> None:
        # identical pairs => statistic is constant across replicates
        a = np.array([5.0, -2.0, 7.0])
        b = a.copy()
        ci = paired_bootstrap_ci(
            a,
            b,
            lambda x, y: float(np.mean(x - y)),
            resamples=300,
            root_seed=1,
        )
        assert ci.se == pytest.approx(0.0)
        assert ci.low == ci.high == pytest.approx(0.0)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.ones(3), np.ones(4), median_gap)
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.ones(3), np.ones(3), median_gap, resamples=10)


class TestIndependentBootstrap:
    def test_two_sample_median_ci(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(2.0, 0.5, size=120)
        b = rng.normal(0.0, 0.5, size=120)
        ci = independent_bootstrap_ci(a, b, median_gap, resamples=1500, root_seed=2)
        assert ci.low > 1.5
        assert ci.high < 2.5

    def test_level_and_metadata(self) -> None:
        rng = np.random.default_rng(4)
        ci = independent_bootstrap_ci(
            rng.normal(size=40), rng.normal(size=40), median_gap,
            resamples=400, level=0.90, root_seed=6,
        )
        assert ci.resamples == 400
        assert ci.level == 0.90
