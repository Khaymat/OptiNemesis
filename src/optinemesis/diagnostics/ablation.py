"""Transformation-ablation diagnostics.

For a nominated candidate theta, each tagged transformation is neutralized in
turn while holding the other parameters at their candidate values. The
resulting effect shift is reported. Neutral values are the parameter-box lower
bounds (identity-like) except for a small curated override map where the lower
bound would be degenerate.
"""

from __future__ import annotations

from optinemesis.core.contract import StudySpec
from optinemesis.core.seeds import SeedTree
from optinemesis.search.engine import evaluate_theta

NEUTRAL_OVERRIDES: dict[tuple[str, str], float] = {}


def neutral_theta_for_family(study: StudySpec, param_index: int) -> float:
    param = study.family.parameter_box[param_index]
    key = (study.family.name, param.name)
    if key in NEUTRAL_OVERRIDES:
        return NEUTRAL_OVERRIDES[key]
    return param.lower


def ablate_candidate(
    study: StudySpec,
    candidate_theta: tuple[float, ...],
    *,
    bootstrap_resamples: int = 300,
) -> dict[str, dict[str, float]]:
    """Re-evaluate *candidate_theta* with each parameter neutralized.

    Returns ``{param_name: {effect, delta_vs_candidate}}``. Each ablation
    reuses the search-stage instance seeds (deterministic) so deltas are
    attributable to the parameter change alone.
    """
    tree = SeedTree.create(int(study.seeds.root_entropy))
    n = study.seeds.n_search_seeds
    instance_seeds = tree.problem_search.seeds(n)
    seeds_a = tree.opt_a.seeds(n)
    seeds_b = tree.opt_b.seeds(n)

    baseline = evaluate_theta(
        study,
        candidate_theta,
        instance_seeds=instance_seeds,
        seeds_a=seeds_a,
        seeds_b=seeds_b,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_root=int(tree.meta.seeds(1)[0]),
    )

    results: dict[str, dict[str, float]] = {
        "baseline": {"effect": baseline.effect, "delta": 0.0}
    }
    names = [p.name for p in study.family.parameter_box]
    for j, name in enumerate(names):
        ablated = list(candidate_theta)
        ablated[j] = neutral_theta_for_family(study, j)
        evaluation = evaluate_theta(
            study,
            tuple(ablated),
            instance_seeds=instance_seeds,
            seeds_a=seeds_a,
            seeds_b=seeds_b,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_root=int(tree.meta.seeds(1)[0]) + j + 1,
        )
        results[name] = {
            "effect": evaluation.effect,
            "delta": evaluation.effect - baseline.effect,
        }
    return results
