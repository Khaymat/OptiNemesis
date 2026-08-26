"""Diagnostics-lite: parameter sensitivity from the search cloud."""

from __future__ import annotations

from optinemesis.core.contract import StudySpec
from optinemesis.core.results import SearchResult


def sensitivity_marginals(
    search_result: SearchResult, study: StudySpec
) -> dict[str, list[tuple[float, float]]]:
    """Per-parameter list of ``(theta_j value, effect)`` pairs.

    The caller may render these as scatter plots or compute rank
    correlations. No causal claim is made; the mapping is purely
    correlational (docs/CLAIMS.md).
    """
    names = [p.name for p in study.family.parameter_box]
    out: dict[str, list[tuple[float, float]]] = {name: [] for name in names}
    for evaluation in search_result.evaluations:
        for j, name in enumerate(names):
            out[name].append((evaluation.theta[j], evaluation.effect))
    return out
