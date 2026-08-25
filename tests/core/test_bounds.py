import numpy as np
import pytest

from optinemesis.core import Bounds, ContractError


def test_valid_bounds() -> None:
    b = Bounds(lower=(-1.0, 0.0), upper=(1.0, 2.0))
    assert b.dimension == 2
    lo, hi = b.as_arrays()
    np.testing.assert_array_equal(lo, [-1.0, 0.0])
    np.testing.assert_array_equal(hi, [1.0, 2.0])
    assert b.contains(np.array([0.0, 1.0]))
    assert not b.contains(np.array([1.5, 1.0]))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((), ()),
        ((1.0,), (-1.0,)),
        ((np.inf,), (2.0,)),
        ((-1.0,), (np.nan,)),
        ((-1.0,), (1.0, 2.0)),
    ],
)
def test_invalid_bounds_rejected(lower: tuple[float, ...], upper: tuple[float, ...]) -> None:
    with pytest.raises(ContractError):
        Bounds(lower=lower, upper=upper)


def test_bounds_spec_roundtrip() -> None:
    b = Bounds(lower=(-5.5, 0.25), upper=(4.5, 3.75))
    assert Bounds.from_spec(b.to_spec()) == b


def test_touching_bounds_rejected() -> None:
    with pytest.raises(ContractError):
        Bounds(lower=(0.0,), upper=(0.0,))
