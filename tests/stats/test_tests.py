import numpy as np
import pytest
from scipy.stats import rankdata, wilcoxon

from optinemesis.stats import (
    holm_bonferroni,
    independent_mann_whitney,
    independent_permutation_test_median_gap,
    paired_permutation_test_median_gap,
    paired_wilcoxon,
    ranks_average,
)


class TestWilcoxonWrapper:
    def test_matches_scipy_reference(self) -> None:
        rng = np.random.default_rng(19)
        x = rng.normal(0.3, 1.0, size=60)
        y = rng.normal(0.0, 1.0, size=60)
        outcome = paired_wilcoxon(x, y)
        stat_ref, p_ref = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        assert outcome.p_value == pytest.approx(float(p_ref))
        assert outcome.statistic == pytest.approx(float(stat_ref))
        assert outcome.test_name == "wilcoxon_signed_rank"

    def test_all_ties_yield_p_one(self) -> None:
        outcome = paired_wilcoxon(np.ones(5), np.ones(5))
        assert outcome.p_value == 1.0
        assert outcome.n_used == 5

    def test_perfect_separation_is_decisive(self) -> None:
        outcome_n4 = paired_wilcoxon(
            np.array([10.0, 20.0, 30.0, 40.0]), np.array([1.0, 2.0, 3.0, 4.0])
        )
        assert outcome_n4.p_value == pytest.approx(0.125)
        rng = np.random.default_rng(101)
        base = rng.normal(size=12)
        outcome_n12 = paired_wilcoxon(base + 1.0, base)
        assert outcome_n12.p_value < 0.01

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            paired_wilcoxon(np.ones(3), np.ones(5))


class TestPermutationTests:
    def test_paired_signflip_detects_shift(self) -> None:
        rng = np.random.default_rng(23)
        base = rng.normal(size=50)
        outcome = paired_permutation_test_median_gap(base + 0.8, base, n_resamples=2000)
        assert outcome.p_value < 0.01

    def test_paired_signflip_null_is_uniformish(self) -> None:
        rng = np.random.default_rng(29)
        p_values = []
        for trial in range(20):
            base = rng.normal(size=40)
            shift = 0.0
            outcome = paired_permutation_test_median_gap(
                base + shift, base + 0.0, n_resamples=500, root_seed=trial
            )
            p_values.append(outcome.p_value)
        fraction_significant = float(np.mean(np.asarray(p_values) < 0.05))
        assert fraction_significant <= 0.25

    def test_independent_variant_runs(self) -> None:
        rng = np.random.default_rng(31)
        outcome_a = independent_permutation_test_median_gap(
            rng.normal(1.0, 1.0, size=40), rng.normal(0.0, 1.0, size=40), n_resamples=800
        )
        assert outcome_a.p_value < 0.05


def test_mann_whitney_reference() -> None:
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(37)
    a = rng.normal(1.0, 1.0, size=45)
    b = rng.normal(0.0, 1.0, size=55)
    outcome = independent_mann_whitney(a, b)
    ref = mannwhitneyu(a, b, alternative="two-sided")
    assert outcome.p_value == pytest.approx(float(ref.pvalue))


class TestHolmCorrection:
    def test_hand_computed_reference(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_bonferroni(raw)
        # sorted [0.01, 0.03, 0.04] x multipliers [3, 2, 1] -> running max [0.03, 0.06, 0.06]
        expected = [0.03, 0.06, 0.06]
        np.testing.assert_allclose(adjusted, expected)

    def test_monotone_and_capped_at_one(self) -> None:
        raw = np.array([0.001, 0.002, 0.02, 0.2, 0.9])
        adjusted = holm_bonferroni(raw)
        assert np.all(np.diff(adjusted[raw.argsort(kind="stable")]) >= -1e-12)
        assert np.all(adjusted >= raw)
        assert np.all(adjusted <= 1.0)

    def test_single_value(self) -> None:
        assert holm_bonferroni([0.03]) == pytest.approx(0.03)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            holm_bonferroni([0.5, 1.5])
        with pytest.raises(ValueError):
            holm_bonferroni([])

    def test_family_wise_control_simulation(self) -> None:
        rng = np.random.default_rng(41)
        rejections = 0
        trials = 400
        for _ in range(trials):
            ps = list(rng.uniform(0.0, 1.0, size=8))
            adjusted = holm_bonferroni(ps)
            if float(np.min(adjusted)) < 0.05:
                rejections += 1
        rate = rejections / trials
        assert rate <= 0.10


def test_ranks_average_exposes_scipy_semantics() -> None:
    np.testing.assert_array_equal(ranks_average([10.0, 20.0, 30.0]), rankdata([10.0, 20.0, 30.0]))
