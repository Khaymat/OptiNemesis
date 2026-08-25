import numpy as np
import pytest

from optinemesis.families.transforms import (
    corner_upper_bound,
    givens_rotation,
    random_unit_vector,
    shifted_center,
)


class TestGivensRotation:
    def test_zero_strength_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        q = givens_rotation(6, 0.0, rng)
        np.testing.assert_array_equal(q, np.eye(6))

    def test_full_strength_is_orthogonal(self) -> None:
        for seed in range(5):
            q = givens_rotation(7, 1.0, np.random.default_rng(seed))
            np.testing.assert_allclose(q.T @ q, np.eye(7), atol=1e-12)
            assert abs(abs(np.linalg.det(q)) - 1.0) < 1e-12

    @pytest.mark.parametrize("dimension", [1, 2, 3, 10, 40])
    def test_orthogonality_across_dimensions(self, dimension: int) -> None:
        for strength in (0.25, 0.5, 0.75):
            q = givens_rotation(dimension, strength, np.random.default_rng(dimension))
            np.testing.assert_allclose(q.T @ q, np.eye(dimension), atol=1e-12)

    def test_deterministic_per_seed(self) -> None:
        q1 = givens_rotation(9, 0.8, np.random.default_rng(123))
        q2 = givens_rotation(9, 0.8, np.random.default_rng(123))
        np.testing.assert_array_equal(q1, q2)

    def test_positive_strength_changes_the_matrix(self) -> None:
        for seed in range(5):
            for strength in (0.25, 0.5, 1.0):
                q = givens_rotation(8, strength, np.random.default_rng(1000 + seed))
                assert not np.array_equal(q, np.eye(8))

    def test_interpretation_is_fraction_of_planes_rotated(self) -> None:
        q_half = givens_rotation(10, 0.5, np.random.default_rng(0))
        changed_columns = int(np.sum(np.any(q_half != np.eye(10), axis=0)))
        assert changed_columns >= 3


class TestHelpers:
    def test_unit_vector_normalized_and_deterministic(self) -> None:
        v1 = random_unit_vector(15, np.random.default_rng(3))
        v2 = random_unit_vector(15, np.random.default_rng(3))
        assert abs(np.linalg.norm(v1) - 1.0) < 1e-12
        np.testing.assert_array_equal(v1, v2)

    def test_shifted_center_inside_shrunk_box(self) -> None:
        lo = np.full(4, -5.0)
        hi = np.full(4, 5.0)
        c = shifted_center(lo, hi, 0.3, np.random.default_rng(8))
        assert np.all(c >= lo + 0.15 * (hi - lo) - 1e-12)
        assert np.all(c <= hi - 0.15 * (hi - lo) + 1e-12)

    def test_shifted_center_valid_for_fraction_above_half(self) -> None:
        lo = np.full(3, -1.0)
        hi = np.full(3, 1.0)
        c = shifted_center(lo, hi, 0.9, np.random.default_rng(8))
        assert np.all(c > lo) and np.all(c < hi)

    def test_shifted_center_rejects_bad_radius(self) -> None:
        with pytest.raises(ValueError):
            shifted_center(np.zeros(2), np.ones(2), 1.0, np.random.default_rng(0))

    def test_corner_upper_bound_is_exact_at_corners(self) -> None:
        lo = np.array([-2.0, -3.0])
        hi = np.array([1.0, 4.0])
        center = np.array([0.5, 0.0])
        bound = corner_upper_bound(center, lo, hi)
        corners = np.array([[a, b] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])])
        worst = np.max(np.sum((corners - center) ** 2, axis=1))
        assert bound == pytest.approx(worst)

    def test_invalid_strength_rejected(self) -> None:
        with pytest.raises(ValueError):
            givens_rotation(4, 1.5, np.random.default_rng(0))
