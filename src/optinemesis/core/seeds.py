"""Hierarchical seed management.

All randomness in OptiNemesis flows through :class:`SeedTree`. There is no
global RNG state. The root entropy spawns five disjoint subtrees; the
validation subtree further spawns three disjoint sub-subtrees. Sharing seeds
between the search and validation stages is impossible by construction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

SUBTREE_LABELS = ("meta", "problem_search", "opt_a", "opt_b", "validation")
VALIDATION_LABELS = ("problem_val", "opt_a_val", "opt_b_val")

DERIVATION_RECIPE = (
    "SeedSequence(root).spawn(5) -> "
    "(meta, problem_search, opt_a, opt_b, validation); "
    "validation.spawn(3) -> (problem_val, opt_a_val, opt_b_val); "
    "leaf seeds = SeedSequence.generate_state(1, dtype=uint32)"
)


def _leaf_seed(sequence: np.random.SeedSequence) -> int:
    # 32-bit leaves keep every backend happy (scipy legacy RandomState
    # rejects seeds >= 2**32); collision risk is acceptable for v0.1 scale.
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class SeedSubtree:
    """A named subtree of the seed hierarchy able to emit deterministic leaf seeds."""

    label: str
    sequence: np.random.SeedSequence = dataclasses.field(repr=False, compare=False)

    def seeds(self, n: int) -> tuple[int, ...]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return tuple(_leaf_seed(child) for child in self.sequence.spawn(n))

    def generator(self, index: int) -> np.random.Generator:
        child = self.sequence.spawn(index + 1)[index]
        return np.random.default_rng(_leaf_seed(child))


@dataclass(frozen=True)
class SeedTree:
    """Root of the five-subtree hierarchy defined by ``DERIVATION_RECIPE``."""

    root_entropy: str

    def __post_init__(self) -> None:
        try:
            int(self.root_entropy)
        except ValueError as exc:
            raise ValueError("root_entropy must be a decimal integer string") from exc

    @classmethod
    def create(cls, root_entropy: int | None = None) -> SeedTree:
        if root_entropy is None:
            root_entropy = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
        return cls(root_entropy=str(int(root_entropy)))

    def _root(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(int(self.root_entropy))

    def _subtree(self, label: str) -> SeedSubtree:
        index = SUBTREE_LABELS.index(label)
        return SeedSubtree(label=label, sequence=self._root().spawn(len(SUBTREE_LABELS))[index])

    @property
    def meta(self) -> SeedSubtree:
        return self._subtree("meta")

    @property
    def problem_search(self) -> SeedSubtree:
        return self._subtree("problem_search")

    @property
    def opt_a(self) -> SeedSubtree:
        return self._subtree("opt_a")

    @property
    def opt_b(self) -> SeedSubtree:
        return self._subtree("opt_b")

    @property
    def validation(self) -> SeedSubtree:
        return self._subtree("validation")

    def validation_subtrees(self) -> dict[str, SeedSubtree]:
        children = self.validation.sequence.spawn(len(VALIDATION_LABELS))
        return {
            label: SeedSubtree(label=label, sequence=child)
            for label, child in zip(VALIDATION_LABELS, children, strict=True)
        }

    def to_dict(self) -> dict[str, str | tuple[str, ...]]:
        return {"root_entropy": self.root_entropy, "subtree_labels": SUBTREE_LABELS}
