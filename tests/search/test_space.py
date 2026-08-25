import numpy as np
import pytest

from optinemesis.core import FamilyRef, ParamRange
from optinemesis.search import sample_thetas_lhs, sample_thetas_random

BOX = FamilyRef(
    name="X",
    version="1",
    parameter_box=(
        ParamRange("alpha", 10.0, 1000.0),
        ParamRange("beta", 0.0, 1.0),
        ParamRange("gamma", -2.0, 2.0),
    ),
)


class TestRandomSampler:
    def test_deterministic(self) -> None:
        a = sample_thetas_random(BOX, 10, root_seed=7)
        b = sample_thetas_random(BOX, 10, root_seed=7)
        assert a == b

    def test_within_box_and_length(self) -> None:
        thetas = sample_thetas_random(BOX, 50, root_seed=1)
        assert len(thetas) == 50
        for theta in thetas:
            assert len(theta) == 3
            assert 10.0 <= theta[0] <= 1000.0
            assert 0.0 <= theta[1] <= 1.0
            assert -2.0 <= theta[2] <= 2.0


class TestLHSSampler:
    def test_stratification_per_dimension(self) -> None:
        n = 24
        thetas = sample_thetas_lhs(BOX, n, root_seed=42)
        arr = np.asarray(thetas)
        for j, (low, high) in enumerate(((10.0, 1000.0), (0.0, 1.0), (-2.0, 2.0))):
            scaled = (arr[:, j] - low) / (high - low)
            strata = np.floor(scaled * n).astype(int)
            assert len(set(strata.tolist())) == n, f"dimension {j} not stratified"
            assert np.all((scaled >= 0.0) & (scaled <= 1.0))

    def test_deterministic(self) -> None:
        a = sample_thetas_lhs(BOX, 16, root_seed=3)
        b = sample_thetas_lhs(BOX, 16, root_seed=3)
        assert a == b
        c = sample_thetas_lhs(BOX, 16, root_seed=4)
        assert a != c

    def test_minimum_two_candidates(self) -> None:
        with pytest.raises(ValueError):
            sample_thetas_lhs(BOX, 1, root_seed=0)

    def test_better_marginal_spread_than_random(self) -> None:
        n = 40
        lhs_arr = np.asarray(sample_thetas_lhs(BOX, n, root_seed=11))
        rnd_arr = np.asarray(sample_thetas_random(BOX, n, root_seed=11))
        lhs_gaps = np.diff(np.sort(lhs_arr[:, 1]))
        rnd_gaps = np.diff(np.sort(rnd_arr[:, 1]))
        assert np.max(lhs_gaps) <= np.max(rnd_gaps) + 1.0 / n
