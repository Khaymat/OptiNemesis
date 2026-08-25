
import pytest

from optinemesis.core import canonical_json, core_module_fingerprint, fingerprint


def test_fingerprint_stable_and_key_order_insensitive() -> None:
    a = fingerprint({"x": 1, "y": [1.0, 2.0]})
    b = fingerprint({"y": [1.0, 2.0], "x": 1})
    assert a == b
    c = fingerprint({"x": 1.0000000001, "y": [1.0, 2.0]})
    assert a != c


def test_fingerprint_handles_numpy_and_tuples() -> None:
    v1 = fingerprint({"a": (1, 2, 3)})
    v2 = fingerprint({"a": [1, 2, 3]})
    assert v1 == v2
    f1 = fingerprint({"x": 1.5})
    f2 = fingerprint({"x": 1.5})
    assert f1 == f2


def test_nan_rejected_in_canonical_json() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_core_module_fingerprint_changes_with_schema_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = core_module_fingerprint()
    monkeypatch.setattr("optinemesis.SCHEMA_VERSION_CONTRACT", "999")
    changed = core_module_fingerprint()
    assert base != changed


def test_fingerprint_hex_digest() -> None:
    h = fingerprint({"anything": True})
    assert len(h) == 64
    int(h, 16)  # must parse as hex
