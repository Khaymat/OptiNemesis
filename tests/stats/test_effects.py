import numpy as np
import pytest

from optinemesis.stats import (
    cliffs_delta_from_samples,
    iqr,
    median,
    median_gap,
    paired_rank_biserial,
    probability_superiority,
    summarize_regrets,
)


class TestLocationStatistics:
    def test_median_iqr_reference(self) -> None:
        values = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0])
        assert median(values) == 3.0
        assert iqr(values) == pytest.approx(2.75)

    def test_median_gap_sign(self) -> None:
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([1.0, 1.0, 1.0])
        assert median_gap(a, b) == pytest.approx(2.0)
        assert median_gap(b, a) == pytest.approx(-2.0)

    def test_summarize_keys(self) -> None:
        s = summarize_regrets(np.arange(10, dtype=float))
        assert set(s) == {"median", "iqr", "q25", "q75"}

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            median([])
        with pytest.raises(ValueError):
            summarize_regrets([])


class TestPairedRankBiserial:
    def test_perfect_separation_positive(self) -> None:
        a = np.array([10.0, 20.0, 30.0, 40.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert paired_rank_biserial(a, b) == pytest.approx(1.0)

    def test_perfect_separation_negative(self) -> None:
        assert paired_rank_biserial(
            np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
        ) == pytest.approx(-1.0)

    def test_ties_and_symmetry_give_zero(self) -> None:
        assert paired_rank_biserial(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
        a = np.array([1.0, -1.0, 3.0, -3.0])
        b = np.zeros(4)
        assert paired_rank_biserial(a, b) == pytest.approx(0.0)

    def test_hand_computed_reference(self) -> None:
        # d = [1, 2, -3, 8]; |d| ranks: 1->1, 2->2, 3->3, 8->4
        # signed rank sum = +1 +2 -3 +4 = 4; total = 10 => delta = 0.4
        a = np.array([2.0, 3.0, 1.0, 9.0])
        b = np.array([1.0, 1.0, 4.0, 1.0])
        assert paired_rank_biserial(a, b) == pytest.approx(0.4)

    def test_mismatched_shapes_rejected(self) -> None:
        with pytest.raises(ValueError):
            paired_rank_biserial(np.ones(3), np.ones(4))
        with pytest.raises(ValueError):
            paired_rank_biserial(np.array([]), np.array([]))


class TestProbabilityOfSuperiority:
    def test_paired_with_half_tie(self) -> None:
        a = np.array([3.0, 2.0, 1.0])
        b = np.array([1.0, 2.0, 2.0])
        ps = probability_superiority(a, b, paired=True)
        assert ps == pytest.approx(0.5)

    def test_independent_cliff_consistency(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(1.0, 1.0, size=50)
        b = rng.normal(0.0, 1.0, size=60)
        ps = probability_superiority(a, b, paired=False)
        cliff = cliffs_delta_from_samples(a, b)
        assert cliff == pytest.approx(2 * ps - 1)

    def test_bounds(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.uniform(size=20)
        b = rng.uniform(size=20)
        for ps in (
            probability_superiority(a, b),
            probability_superiority(np.ones(5), np.zeros(5)),
            probability_superiority(np.zeros(5), np.ones(5), paired=True),
        ):
            assert 0.0 <= ps <= 1.0
