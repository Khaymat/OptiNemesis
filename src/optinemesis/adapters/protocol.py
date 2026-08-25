"""Optimizer protocol and implementation registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from optinemesis.core.errors import AdapterError

if TYPE_CHECKING:
    from optinemesis.core.objective import CountingObjective
    from optinemesis.core.results import RunResult


@runtime_checkable
class Optimizer(Protocol):
    """Minimal optimizer protocol.

    Implementations receive the harness-owned :class:`CountingObjective` and a
    dedicated integer seed. Bounds, dimension and budget are read from the
    objective itself, so adapters cannot disagree with the contract.
    """

    def run(self, objective: CountingObjective, seed: int) -> RunResult: ...


_FACTORY_REGISTRY: dict[str, Callable[[dict[str, Any]], Optimizer]] = {}


def register_optimizer_factory(
    implementation: str,
) -> Callable[[Callable[[dict[str, Any]], Optimizer]], Callable[[dict[str, Any]], Optimizer]]:
    """Decorator registering a factory ``(config dict) -> Optimizer``."""

    def decorator(
        factory: Callable[[dict[str, Any]], Optimizer],
    ) -> Callable[[dict[str, Any]], Optimizer]:
        if implementation in _FACTORY_REGISTRY:
            raise AdapterError(f"optimizer implementation {implementation!r} already registered")
        _FACTORY_REGISTRY[implementation] = factory
        return factory

    return decorator


def list_implementations() -> tuple[str, ...]:
    return tuple(sorted(_FACTORY_REGISTRY))


def build_optimizer(spec_implementation: str, config: dict[str, Any] | None = None) -> Optimizer:
    """Instantiate an optimizer from its registered implementation name."""
    key = spec_implementation
    if key not in _FACTORY_REGISTRY:
        _load_entry_points()
        if key not in _FACTORY_REGISTRY:
            known = ", ".join(list_implementations()) or "<none>"
            raise AdapterError(
                f"unknown optimizer implementation {key!r}; registered: {known}"
            )
    return _FACTORY_REGISTRY[key](dict(config or {}))


_ENTRY_POINTS_LOADED = False


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        group = entry_points(group="optinemesis.optimizers")
    except Exception:
        return
    for ep in group:
        name = f"{ep.name}"
        if name in _FACTORY_REGISTRY:
            continue
        try:
            factory = ep.load()
            _FACTORY_REGISTRY[name] = factory
        except Exception:
            continue
