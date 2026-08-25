"""Shared recording wrapper used by all adapters.

The recorder sits between the harness-owned CountingObjective and the backend.
It observes every completed evaluation, maintains best-so-far state and an
anytime history, and lets BudgetExhausted propagate upward. Because recorder
state survives the propagated exception, adapters can honor the budget exactly
even when a backend attempts an evaluation past the cap.
"""

from __future__ import annotations

import numpy as np

from optinemesis.core.errors import BudgetExhausted
from optinemesis.core.objective import CountingObjective


class RecordingObjective:
    def __init__(self, objective: CountingObjective) -> None:
        self._objective = objective
        self.best_x: np.ndarray | None = None
        self.best_f: float | None = None
        self.history: list[tuple[int, float]] = []
        self.overshoot_events = 0

    @property
    def inner(self) -> CountingObjective:
        return self._objective

    @property
    def terminated_by_budget(self) -> bool:
        return self._objective.remaining <= 0

    def __call__(self, x: np.ndarray) -> float:
        try:
            value = self._objective(x)
        except BudgetExhausted:
            self.overshoot_events += 1
            raise
        arr = np.asarray(x, dtype=float).reshape(-1)
        if self.best_f is None or value < self.best_f:
            self.best_f = value
            self.best_x = arr.copy()
            self.history.append((self._objective.consumed, value))
        return value

    def evaluate_without_recording(self, x: np.ndarray) -> float:
        return self._objective(x)
