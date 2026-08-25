"""Mandatory fresh-seed validation of nominated candidates.

This stage is the load-bearing safeguard against the winner's curse: every
statistic here is computed exclusively on the disjoint validation subtrees of
the seed hierarchy, never on search-stage seeds.
"""

from __future__ import annotations

import numpy as np

from optinemesis.core.contract import StudySpec
from optinemesis.core.results import (
    CandidateOutcome,
    CriteriaFlags,
    SearchResult,
    ValidationResult,
)
from optinemesis.core.seeds import SeedTree
from optinemesis.runners.execution import execute_run
from optinemesis.stats import (
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_rank_biserial,
    probability_superiority,
)
from optinemesis.stats.effects import median_gap

MAX_VALIDATED_CANDIDATES = 5


def _validation_regrets(
    study: StudySpec,
    theta: tuple[float, ...],
    tree: SeedTree,
) -> tuple[np.ndarray, np.ndarray]:
    contract = study.bind(theta)
    spec_a, spec_b = contract.optimizers
    n = study.seeds.n_validation_seeds
    instance_seeds = tree.validation_subtrees()["problem_val"].seeds(n)
    seeds_a = tree.validation_subtrees()["opt_a_val"].seeds(n)
    seeds_b = tree.validation_subtrees()["opt_b_val"].seeds(n)
    regrets_a = np.empty(n, dtype=float)
    regrets_b = np.empty(n, dtype=float)
    for i in range(n):
        run_a = execute_run(contract, spec_a, seeds_a[i], instance_seeds[i])
        run_b = execute_run(contract, spec_b, seeds_b[i], instance_seeds[i])
        assert run_a.regret_normalized is not None and run_b.regret_normalized is not None
        regrets_a[i] = run_a.regret_normalized
        regrets_b[i] = run_b.regret_normalized
    return regrets_a, regrets_b


def validate_candidates(search_result: SearchResult, study: StudySpec) -> ValidationResult:
    """Confirm or reject nominated candidates on a completely fresh seed subtree.

    A candidate becomes a validated counterexample only if all four criteria
    hold: Holm-corrected significance, effect size above ``epsilon_min``,
    median-gap bootstrap CI excluding the practical-equivalence margin in the
    claimed direction, and opposition to the neutral reference direction.
    """
    from optinemesis.stats.tests import paired_wilcoxon

    tree = SeedTree.create(int(study.seeds.root_entropy))
    nominated = [e for e in search_result.nominated][:MAX_VALIDATED_CANDIDATES]
    if not nominated:
        return ValidationResult(
            confirmed=(), failed=(), holm_family_alpha=study.thresholds.alpha_validation
        )

    raw_p: list[float] = []
    effects: list[float] = []
    effect_cis: list[tuple[float, float]] = []
    gap_cis: list[tuple[float, float]] = []
    medians: list[tuple[float, float]] = []
    pss: list[float] = []

    for evaluation in nominated:
        regrets_a, regrets_b = _validation_regrets(study, evaluation.theta, tree)
        wilcoxon_result = paired_wilcoxon(regrets_a, regrets_b)
        effect_ci = paired_bootstrap_ci(
            regrets_a,
            regrets_b,
            paired_rank_biserial,
            resamples=study.thresholds.bootstrap_resamples,
            level=1.0 - study.thresholds.alpha_validation,
            root_seed=int(tree.meta.seeds(1)[0]),
        )
        gap_ci = paired_bootstrap_ci(
            regrets_a,
            regrets_b,
            median_gap,
            resamples=study.thresholds.bootstrap_resamples,
            level=1.0 - study.thresholds.alpha_validation,
            root_seed=int(tree.meta.seeds(1)[0]) + 1,
        )
        raw_p.append(wilcoxon_result.p_value)
        effects.append(paired_rank_biserial(regrets_a, regrets_b))
        effect_cis.append((effect_ci.low, effect_ci.high))
        gap_cis.append((gap_ci.low, gap_ci.high))
        medians.append((float(np.median(regrets_a)), float(np.median(regrets_b))))
        pss.append(probability_superiority(regrets_a, regrets_b))

    adjusted = holm_bonferroni(raw_p)
    epsilon_min = study.thresholds.epsilon_min
    epsilon_zero = study.thresholds.epsilon_zero
    reference_sign = search_result.reference_direction.sign

    confirmed: list[CandidateOutcome] = []
    failed: list[CandidateOutcome] = []
    for i, evaluation in enumerate(nominated):
        gap_low, gap_high = gap_cis[i]
        flags = CriteriaFlags(
            holm_significant=bool(adjusted[i] < study.thresholds.alpha_validation),
            effect_threshold_met=bool(effects[i] >= epsilon_min),
            ci_excludes_margin=bool(gap_low > epsilon_zero),
            reference_direction_opposed=bool(
                reference_sign < 0 and effects[i] > 0
            ),
        )
        outcome = CandidateOutcome(
            theta=evaluation.theta,
            dimension=study.dimension,
            effect=effects[i],
            effect_ci_low=effect_cis[i][0],
            effect_ci_high=effect_cis[i][1],
            p_value_raw=raw_p[i],
            p_value_holm=float(adjusted[i]),
            test_name="wilcoxon_signed_rank",
            n_validation_seeds=study.seeds.n_validation_seeds,
            median_regret_a=medians[i][0],
            median_regret_b=medians[i][1],
            median_gap=medians[i][0] - medians[i][1],
            median_gap_ci_low=gap_low,
            median_gap_ci_high=gap_high,
            probability_superiority=pss[i],
            criteria=flags,
        )
        (confirmed if outcome.confirmed else failed).append(outcome)

    return ValidationResult(
        confirmed=tuple(confirmed),
        failed=tuple(failed),
        holm_family_alpha=study.thresholds.alpha_validation,
    )
