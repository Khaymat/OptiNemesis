"""Box bounds for problem instances and contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optinemesis.core.errors import ContractError


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned box bounds. Elementwise strict ``lower < upper`` required."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        if lo.ndim != 1 or hi.ndim != 1:
            raise ContractError("bounds must be one-dimensional")
        if lo.size == 0:
            raise ContractError("bounds must not be empty")
        if lo.size != hi.size:
            raise ContractError(
                f"bounds length mismatch: lower has {lo.size}, upper has {hi.size}"
            )
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ContractError("bounds must be finite")
        if not np.all(lo < hi):
            raise ContractError("every lower bound must be strictly below its upper bound")

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.lower, dtype=float), np.asarray(self.upper, dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        lo, hi = self.as_arrays()
        return bool(np.all(x >= lo) and np.all(x <= hi))

    def to_spec(self) -> dict[str, list[float]]:
        return {"lower": list(self.lower), "upper": list(self.upper)}

    @classmethod
    def from_spec(cls, spec: dict[str, list[float]]) -> Bounds:
        return cls(lower=tuple(spec["lower"]), upper=tuple(spec["upper"]))
