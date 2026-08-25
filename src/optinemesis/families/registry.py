"""Family registry. Registration is idempotent per (name, version)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optinemesis.core.errors import RegistryError

if TYPE_CHECKING:
    from optinemesis.families.base import ProblemFamily

_REGISTRY: dict[str, ProblemFamily] = {}


def _as_instance(family_or_class: ProblemFamily | type[ProblemFamily]) -> ProblemFamily:
    if isinstance(family_or_class, type):
        return family_or_class()
    return family_or_class


def register_family(
    family_or_class: ProblemFamily | type[ProblemFamily],
) -> ProblemFamily | type[ProblemFamily]:
    """Register a family instance or stateless family class; usable as a decorator."""
    family = _as_instance(family_or_class)
    key = family.registry_key()
    existing = _REGISTRY.get(key)
    if existing is not None:
        if type(existing) is type(family):
            return family_or_class
        raise RegistryError(f"duplicate family registration for {key!r}")
    _REGISTRY[key] = family
    return family_or_class


def get_family(name: str, version: str = "1") -> ProblemFamily:
    key = f"{name}@{version}"
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise RegistryError(f"unknown family {key!r}; registered: {known}") from exc


def list_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def clear_registry() -> None:
    """Test-only reset."""
    _REGISTRY.clear()
