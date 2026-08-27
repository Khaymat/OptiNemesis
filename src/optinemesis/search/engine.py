"""Meta-search engine: exploratory evaluation of theta candidates.

Everything produced here is exploratory. Candidates are nominated, never
confirmed; see optinemesis.validate for the mandatory fresh-seed stage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as normal

from optinemesis.core.contract import StudySpec
from optinemesis.core.results import (
    CandidateEvaluation,
    ReferenceDirection,
    SearchResult,
)
from optinemesis.core.seeds import SeedTree
from optinemesis.runners.execution import execute_run
from optinemesis.search.space import sample_thetas_lhs, sample_thetas_random


def _theta_samples(study: StudySpec, tree: SeedTree) -> list[tuple[float, ...]]:
    method = study.seeds.meta_search_method
    n = study.seeds.n_theta_candidates
    root_seed = tree.meta.seeds(1)[0]
    if method == "lhs":
        return sample_thetas_lhs(study.family, n, root_seed=root_seed)
    return sample_thetas_random(study.family, n, root_seed=root_seed)


def evaluate_theta(
    study: StudySpec,
    theta: tuple[float, ...],
    *,
    instance_seeds: tuple[int, ...],
    seeds_a: tuple[int, ...],
    seeds_b: tuple[int, ...],
    bootstrap_resamples: int = 500,
    bootstrap_root: int = 0,
) -> CandidateEvaluation:
    """Paired evaluation of one theta on the given instance seeds (exploratory)."""
    from optinemesis.stats import (
        paired_bootstrap_ci,
        paired_rank_biserial,
        probability_superiority,
    )

    contract = study.bind(theta)
    spec_a, spec_b = contract.optimizers
    regrets_a = np.empty(len(instance_seeds), dtype=float)
    regrets_b = np.empty_like(regrets_a)
    for i, s in enumerate(instance_seeds):
        run_a = execute_run(contract, spec_a, seeds_a[i], s)
        run_b = execute_run(contract, spec_b, seeds_b[i], s)
        assert run_a.regret_normalized is not None
        assert run_b.regret_normalized is not None
        regrets_a[i] = run_a.regret_normalized
        regrets_b[i] = run_b.regret_normalized

    effect = paired_rank_biserial(regrets_a, regrets_b)
    ci = paired_bootstrap_ci(
        regrets_a,
        regrets_b,
        paired_rank_biserial,
        resamples=bootstrap_resamples,
        level=1.0 - study.thresholds.alpha_search,
        root_seed=bootstrap_root,
    )
    z_value = float(normal.ppf(1.0 - study.thresholds.alpha_search))
    # Symmetric conservative bound towards zero; direction-aware.
    # Degenerate bootstrap SE=0 at perfect separation (all differences same
    # sign) yields SE=0 and LCB=effect=±1, overstating confidence for small n.
    # In that case use exact Clopper-Pearson lower bound for PS mapped to
    # rank-biserial: L = 2*alpha^{1/n} -1 (k=n successes). Gives finite-sample
    # guaranteed coverage without arbitrary epsilon. See DESIGN.md §2.
    if ci.se < 1e-12 and abs(abs(effect) - 1.0) < 1e-9:
        n = regrets_a.size
        alpha = study.thresholds.alpha_search
        # alpha in (0,1); for n≥1, L in [-1,1)
        l_abs = 2.0 * (alpha ** (1.0 / float(n))) - 1.0
        # signed conservative bound towards zero
        if effect > 0:
            lcb = float(l_abs)
        elif effect < 0:
            lcb = float(-l_abs)
        else:
            lcb = 0.0
    else:
        if effect >= 0:
            lcb = effect - z_value * ci.se
        else:
            lcb = effect + z_value * ci.se
    return CandidateEvaluation(
        theta=theta,
        effect=effect,
        lcb=lcb,
        se=ci.se,
        median_regret_a=float(np.median(regrets_a)),
        median_regret_b=float(np.median(regrets_b)),
        probability_superiority=probability_superiority(regrets_a, regrets_b),
        nominated=False,
    )


def run_search(study: StudySpec) -> SearchResult:
    """Explore the parameter box and nominate candidates by the LCB criterion."""
    tree = SeedTree.create(int(study.seeds.root_entropy))
    n_seeds = study.seeds.n_search_seeds
    instance_seeds = tree.problem_search.seeds(n_seeds)
    seeds_a = tree.opt_a.seeds(n_seeds)
    seeds_b = tree.opt_b.seeds(n_seeds)

    thetas = _theta_samples(study, tree)
    evaluations: list[CandidateEvaluation] = []
    all_gaps: list[float] = []
    for index, theta in enumerate(thetas):
        evaluation = evaluate_theta(
            study,
            theta,
            instance_seeds=instance_seeds,
            seeds_a=seeds_a,
            seeds_b=seeds_b,
            bootstrap_resamples=min(500, study.thresholds.bootstrap_resamples),
            bootstrap_root=(tree.meta.seeds(2)[1] + index) % (2**63),
        )
        evaluations.append(evaluation)
        all_gaps.append(evaluation.median_regret_a - evaluation.median_regret_b)

    reference_median_gap = float(np.median(all_gaps))
    reference_sign = 0
    if reference_median_gap > 0.0:
        reference_sign = 1
    elif reference_median_gap < 0.0:
        reference_sign = -1
    reference_direction = ReferenceDirection(
        sample_size=len(thetas),
        median_gap=reference_median_gap,
        sign=reference_sign,
    )

    epsilon_min = study.thresholds.epsilon_min
    # Symmetric, reference-aware nomination.
    # Reference defines the neutral ranking; a reversal is an effect whose
    # sign opposes the reference. Nomination therefore considers only
    # candidates whose sign is opposite the reference (or any sign if
    # reference is 0, but then validation will reject via reference gate).
    # This makes swapping A↔B (which flips both effect and reference) preserve
    # the nominated theta set, satisfying label-swap symmetry.
    # lcb is signed conservative bound towards zero; abs(lcb) is magnitude.
    if reference_sign == 0:
        # No reference direction — nothing can be a reversal by definition
        nominated_indices: set[int] = set()
    else:
        need_sign = -reference_sign  # 1 => need -1, -1 => need 1
        # Filter to reversal candidates first, then rank by magnitude
        def _sign(v: float) -> int:
            return 1 if v > 0 else -1 if v < 0 else 0

        reversal_indices = [
            i for i, e in enumerate(evaluations) if _sign(e.effect) == need_sign
        ]
        ranked = sorted(
            reversal_indices, key=lambda i: abs(evaluations[i].lcb), reverse=True
        )
        nominated_indices = {
            i for i in ranked[:5] if abs(evaluations[i].lcb) >= epsilon_min
        }
    flagged = [
        CandidateEvaluation(
            theta=e.theta,
            effect=e.effect,
            lcb=e.lcb,
            se=e.se,
            median_regret_a=e.median_regret_a,
            median_regret_b=e.median_regret_b,
            probability_superiority=e.probability_superiority,
            nominated=(i in nominated_indices),
        )
        for i, e in enumerate(evaluations)
    ]
    return SearchResult(evaluations=tuple(flagged), reference_direction=reference_direction)
