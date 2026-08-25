"""Conservative Markdown reporting.

The renderer requires a :class:`ValidationResult` by signature: an unvalidated
search result can never be rendered as a confirmed finding. All claim sentences
follow docs/CLAIMS.md.
"""

from __future__ import annotations

from optinemesis.core.results import CandidateOutcome, SearchResult, ValidationResult
from optinemesis.reporting.artifact import build_artifact


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _candidate_table_row(outcome: CandidateOutcome, status: str) -> str:
    theta = "(" + ", ".join(_fmt(t) for t in outcome.theta) + ")"
    return (
        f"| {status} | {theta} | {_fmt(outcome.effect)} "
        f"| [{_fmt(outcome.effect_ci_low)}, {_fmt(outcome.effect_ci_high)}] "
        f"| {_fmt(outcome.p_value_holm)} | {outcome.n_validation_seeds} "
        f"| {_fmt(outcome.median_regret_a)} / {_fmt(outcome.median_regret_b)} |"
    )


def render_markdown(
    study: object,
    search_result: SearchResult,
    validation_result: ValidationResult,
) -> str:
    """Render the full report. ``study`` must be a StudySpec (typed loosely to avoid a cycle)."""
    from optinemesis.core.contract import StudySpec

    if not isinstance(study, StudySpec):
        raise TypeError("render_markdown requires a StudySpec as first argument")

    lines: list[str] = []
    lines.append("# OptiNemesis counterexample-discovery report")
    lines.append("")
    lines.append(
        "> Scientific positioning: *a framework for validated counterexample "
        "discovery in optimization algorithm comparisons*. \"Nemesis\" is "
        "branding only; all claims below are contract-scoped (docs/CLAIMS.md)."
    )
    lines.append("")

    lines.append("## Experimental contract")
    lines.append("")
    lines.append(f"- Family: `{study.family.name}@{study.family.version}`")
    lines.append(f"- Dimension: {study.dimension}")
    lines.append(f"- Bounds: {study.bounds.to_spec()}")
    lines.append(f"- Evaluation budget (cap): {study.budget}")
    lines.append(f"- Initialization policy: {study.initialization_policy}")
    lines.append(
        f"- Seeds: search={study.seeds.n_search_seeds}, "
        f"validation={study.seeds.n_validation_seeds} "
        f"(disjoint subtrees of root {study.seeds.root_entropy})"
    )
    lines.append(
        "- Optimizer configurations: "
        + "; ".join(
            f"`{o.name}` = {o.implementation} config={o.config}" for o in study.optimizers
        )
    )
    lines.append(
        f"- Thresholds: epsilon_min={study.thresholds.epsilon_min}, "
        f"epsilon_zero={study.thresholds.epsilon_zero}, "
        f"alpha_validation={study.thresholds.alpha_validation}"
    )
    lines.append("")

    ref = search_result.reference_direction
    lines.append("## Neutral reference direction (exploratory context)")
    lines.append("")
    lines.append(
        f"Over a neutral sample of {ref.sample_size} parameter vectors, the median "
        f"regret gap (`{study.optimizers[0].name}` minus `{study.optimizers[1].name}`) "
        f"was {_fmt(ref.median_gap)} (sign {ref.sign:+d}). This is exploratory "
        "context, not confirmatory evidence."
    )
    lines.append("")

    lines.append("## Search stage (exploratory - not confirmatory evidence)")
    lines.append("")
    lines.append("| nominated | theta | effect | LCB | SE | median regret A / B | PS |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in sorted(search_result.evaluations, key=lambda x: x.lcb, reverse=True):
        theta = "(" + ", ".join(_fmt(t) for t in e.theta) + ")"
        lines.append(
            f"| {'yes' if e.nominated else 'no'} | {theta} | {_fmt(e.effect)} "
            f"| {_fmt(e.lcb)} | {_fmt(e.se)} "
            f"| {_fmt(e.median_regret_a)} / {_fmt(e.median_regret_b)} "
            f"| {_fmt(e.probability_superiority)} |"
        )
    lines.append("")

    name_a, name_b = study.optimizers[0].name, study.optimizers[1].name
    lines.append("## Validation stage (fresh seeds, confirmatory)")
    lines.append("")
    lines.append(
        f"Holm-corrected at family-wise alpha = {validation_result.holm_family_alpha}. "
        "Positive effect means configuration A underperformed configuration B."
    )
    lines.append("")
    if validation_result.candidates:
        lines.append(
            "| status | theta | effect | 95% CI | Holm p | n seeds | median regret A / B |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c in validation_result.confirmed:
            lines.append(_candidate_table_row(c, "confirmed"))
        for c in validation_result.failed:
            lines.append(_candidate_table_row(c, "failed"))
        lines.append("")
    else:
        lines.append(
            "No candidate met the nomination threshold; this does not establish "
            "that no reversal exists for these configurations."
        )
        lines.append("")

    if validation_result.confirmed:
        lines.append("### Validated findings")
        lines.append("")
        for c in validation_result.confirmed:
            theta = "(" + ", ".join(_fmt(t) for t in c.theta) + ")"
            lines.append(
                f"- Under this serialized experimental contract and on this validated "
                f"region of problem family `{study.family.name}`, optimizer "
                f"configuration `{name_a}` underperformed configuration `{name_b}` at "
                f"theta = {theta}: median normalized regret "
                f"{_fmt(c.median_regret_a)} vs {_fmt(c.median_regret_b)}, rank-biserial "
                f"effect size {_fmt(c.effect)} (95% bootstrap CI "
                f"[{_fmt(c.effect_ci_low)}, {_fmt(c.effect_ci_high)}]), Holm-corrected "
                f"validation p = {_fmt(c.p_value_holm)}, n = {c.n_validation_seeds} fresh "
                f"seeds disjoint from all search-stage seeds."
            )
        lines.append("")

    if validation_result.failed:
        lines.append("### Failed validations (retained; must not be read as findings)")
        lines.append("")
        for c in validation_result.failed:
            theta = "(" + ", ".join(_fmt(t) for t in c.theta) + ")"
            reasons = "; ".join(c.failure_reasons)
            lines.append(f"- theta = {theta}: failed validation ({reasons}).")
        lines.append("")

    artifact_stub = build_artifact(study, search_result, validation_result)
    fingerprints = artifact_stub["fingerprints"]
    lines.append("## Reproduction fingerprint")
    lines.append("")
    lines.append(f"- Contract id: `{fingerprints['contract_id'][:16]}...`")
    lines.append(f"- Core module: `{fingerprints['core_module'][:16]}...`")
    lines.append(f"- Registry: `{fingerprints['registry'][:16]}...`")
    lines.append("- Seed derivation: see artifact `seed_tree.derivation_recipe`.")
    lines.append("")
    lines.append(
        "*This software emits conditional statements about optimizer configurations "
        "under specific contracts; it does not rank algorithms in general.*"
    )
    return "\n".join(lines)
