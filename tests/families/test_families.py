import numpy as np
import pytest

from optinemesis.core import Bounds, RegistryError
from optinemesis.families import (
    DeceptiveDoubleWellFamily,
    EllipsoidalFamily,
    MultimodalBlendFamily,
    get_family,
    list_families,
)

BOUNDS = Bounds(lower=tuple(-5.0 for _ in range(8)), upper=tuple(5.0 for _ in range(8)))
D = 8


def probes(generator: np.random.Generator, n: int = 20) -> list[np.ndarray]:
    return [generator.uniform(-5, 5, size=D) for _ in range(n)]


FAMILIES = [
    (EllipsoidalFamily(), (100.0, 0.7, 0.4)),
    (MultimodalBlendFamily(), (2.0, 0.6, 0.35)),
    (DeceptiveDoubleWellFamily(), (6.0, 8.0, 0.5)),
]


@pytest.mark.parametrize("family,theta", FAMILIES, ids=lambda f: getattr(f, "name", ""))
class TestDeterministicValues:
    def test_same_seed_same_values(self, family: object, theta: tuple[float, ...]) -> None:
        inst_a = family.sample(theta, D, instance_seed=777, bounds=BOUNDS)  # type: ignore[attr-defined]
        inst_b = family.sample(theta, D, instance_seed=777, bounds=BOUNDS)  # type: ignore[attr-defined]
        rng = np.random.default_rng(2024)
        for x in probes(rng):
            assert inst_a.evaluate(x) == inst_b.evaluate(x)

    def test_different_seed_different_structure(
        self, family: object, theta: tuple[float, ...]
    ) -> None:
        inst_a = family.sample(theta, D, instance_seed=1, bounds=BOUNDS)  # type: ignore[attr-defined]
        inst_b = family.sample(theta, D, instance_seed=2, bounds=BOUNDS)  # type: ignore[attr-defined]
        rng = np.random.default_rng(555)
        values_a = [inst_a.evaluate(x) for x in probes(rng)]
        values_b = [inst_b.evaluate(x) for x in probes(rng)]
        assert any(a != b for a, b in zip(values_a, values_b, strict=True))

    def test_known_optimum_attained(self, family: object, theta: tuple[float, ...]) -> None:
        inst = family.sample(theta, D, instance_seed=42, bounds=BOUNDS)  # type: ignore[attr-defined]
        assert inst.optimum_x is not None
        optimum_x = np.asarray(inst.optimum_x, dtype=float)
        value = inst.evaluate(optimum_x)
        assert value == pytest.approx(inst.optimum_value, abs=1e-9)
        assert inst.bounds.contains(optimum_x)

    def test_regret_scale_positive_finite(
        self, family: object, theta: tuple[float, ...]
    ) -> None:
        inst = family.sample(theta, D, instance_seed=9, bounds=BOUNDS)  # type: ignore[attr-defined]
        scale = float(inst.landscape_tags["regret_scale"])
        assert np.isfinite(scale) and scale > 0

    def test_spec_carries_reconstruction_fields(
        self, family: object, theta: tuple[float, ...]
    ) -> None:
        inst = family.sample(theta, D, instance_seed=13, bounds=BOUNDS)  # type: ignore[attr-defined]
        spec = inst.to_spec()
        assert spec["family_name"] == inst.family_name
        assert spec["family_version"] == "1"
        assert spec["spec"]["instance_seed"] == 13
        assert spec["theta"] == list(theta)


def test_multimodal_blend_nonnegative_everywhere_probed() -> None:
    fam = MultimodalBlendFamily()
    for theta in ((1.0, 0.0, 0.5), (3.0, 1.0, 0.2)):
        inst = fam.sample(theta, D, instance_seed=31, bounds=BOUNDS)
        rng = np.random.default_rng(808)
        for x in probes(rng, 50):
            v = inst.evaluate(x)
            assert v >= -1e-12, f"negative value {v} at {x}"


def test_multimodal_separable_vs_rotated_differ() -> None:
    fam = MultimodalBlendFamily()
    sep_inst = fam.sample((1.5, 0.0, 0.5), D, instance_seed=17, bounds=BOUNDS)
    rot_inst = fam.sample((1.5, 1.0, 0.5), D, instance_seed=17, bounds=BOUNDS)
    rng = np.random.default_rng(64)
    diffs = [
        abs(sep_inst.evaluate(x) - rot_inst.evaluate(x)) for x in probes(rng, 30)
    ]
    assert max(diffs) > 1e-6


def test_doublewell_wide_basin_is_local_not_global() -> None:
    fam = DeceptiveDoubleWellFamily()
    inst = fam.sample((10.0, 10.0, 0.5), D, instance_seed=21, bounds=BOUNDS)
    c_wide = np.asarray(inst.spec["c_wide"], dtype=float)
    c_deep = np.asarray(inst.spec["c_deep"], dtype=float)
    wide_value = inst.evaluate(c_wide)
    deep_value = inst.evaluate(c_deep)
    assert deep_value == pytest.approx(inst.optimum_value, rel=1e-12)
    assert wide_value > deep_value
    assert wide_value == pytest.approx(-float(inst.spec["delta_wide"]), rel=1e-9)


def test_ellipsoidal_conditioning_increases_anisotropy() -> None:
    fam = EllipsoidalFamily()
    weak = fam.sample((10.0, 0.9, 0.5), D, instance_seed=5, bounds=BOUNDS)
    strong = fam.sample((1000.0, 0.9, 0.5), D, instance_seed=5, bounds=BOUNDS)
    rng = np.random.default_rng(303)
    ratios = []
    for x in probes(rng, 20):
        r = strong.evaluate(x) / max(weak.evaluate(x), 1e-300)
        ratios.append(r)
    assert max(ratios) > 10.0


def test_dimension_mismatch_rejected() -> None:
    fam = EllipsoidalFamily()
    with pytest.raises(ValueError):
        fam.sample((100.0, 0.5, 0.5), 3, 1, BOUNDS)


def test_out_of_box_theta_rejected() -> None:
    from optinemesis.core import ContractError

    fam = EllipsoidalFamily()
    with pytest.raises(ContractError):
        fam.sample((5000.0, 0.5, 0.5), D, 1, BOUNDS)


def test_registry_roundtrip() -> None:
    names = list_families()
    assert "Ellipsoidal@1" in names
    assert get_family("Ellipsoidal").name == "Ellipsoidal"
    with pytest.raises(RegistryError):
        get_family("NoSuchFamily")
