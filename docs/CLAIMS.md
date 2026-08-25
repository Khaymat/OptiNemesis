# OptiNemesis — Claim Boundaries (NORMATIVE)

This document is normative. If any code, report template, README sentence, or
artifact field conflicts with this document, this document wins.

The word **"configuration"** below always means "optimizer configuration":
an optimizer implementation together with its frozen hyperparameter values,
as serialized in the contract.

---

## 1. Allowed claims (templates used verbatim by the report renderer)

- "Under this serialized experimental contract and on this validated region of
  problem family `{family}`, optimizer configuration `{A}` underperformed
  configuration `{B}`: median normalized regret `{x}` vs `{y}`
  (rank-biserial effect size `{δ}`, 95% bootstrap CI `[{lo}, {hi}]`,
  Holm-corrected validation p = `{p}`, n = `{n}` fresh seeds)."

- "The search identified a parameter region of `{family}` where, under this
  contract, the ranking observed over a neutral reference sample of parameters
  reversed; this reversal was reproduced on `{n}` fresh problem instances and
  fresh optimizer seeds that were disjoint from all search-stage seeds."

- "Candidate θ = {θ} did not survive fresh-seed validation and must not be
  interpreted as evidence of a performance difference."

- "Under this contract, no candidate met the nomination threshold; this does not
  establish that no reversal exists for these configurations."

- "Runtime measurements are informational only; this study makes no runtime-fairness
  claim."

## 2. Forbidden claims

The software must never emit, imply, or template the following:

| Forbidden | Why |
|---|---|
| "Optimizer B is better than optimizer A." | Unscoped superiority claim; contradicts bounded empirical evidence. |
| "This proves optimizer A is fundamentally weak." | Proof language; single-family, finite-budget evidence. |
| "Algorithm-independent finding." | Discovered regions may reflect family expressiveness and search design. |
| Any search-stage statistic presented as confirmation. | Search is exploratory; winner's curse. |
| "No free lunch is violated/circumvented." | NFL misuse in either direction. |
| "OptiNemesis shows which optimizer to use on your problem." | Solver recommendation is out of scope. |
| Causal landscape explanations ("rotation *causes* the gap"). | Diagnostics are correlational; hedged phrasing required ("is associated with", "consistent with"). |
| Runtime-based rankings. | Wall-clock is metadata only in v0.1. |
| Generalization across dimensions/budgets/families beyond those tested. | Each contract dimension is separate evidence. |
| "Benchmark performance" numbers for OptiNemesis itself presented as scientific results. | Internal benchmarks are engineering data. |

## 3. Statistical reporting rules

1. Every confirmatory claim reports: effect size + bootstrap CI + Holm-corrected
   p-value + seed count + the declared practical thresholds (`ε_min`, `ε₀`).
2. Search-stage statistics may appear in reports only inside sections explicitly
   labeled "Search stage (exploratory — not confirmatory evidence)".
3. Failed validations appear with the same prominence as confirmed ones.
4. Single-seed results are structurally unrepresentable: every reported statistic
   aggregates ≥ n_search_seeds / n_validation_seeds runs.
5. p-values without effect sizes and CIs are never rendered.

## 4. No-Free-Lunch policy

Permitted: citing NFL as motivation for expecting reversals between configurations
on different problem distributions.

Forbidden: claiming OptiNemesis constructs "the" counterexample distribution,
or that absence of discovered reversals within a family supports any general claim.

## 5. Prior-art acknowledgment (must ship in README and docs)

OptiNemesis does **not** claim to invent:
optimization benchmarking platforms (COCO/BBOB; IOHprofiler), parameterized
benchmark generation (MA-BBOB; W-model), adversarial/hard instance generation
(Smith-Miles & Bowly; HIRO), Instance Space Analysis / algorithm footprints
(MATILDA), pairwise win-rate comparison (Nevergrad fight plots), or exploratory
landscape analysis (flacco/pflacco).

Its claimed contribution is narrowly: an integrated, reproducible pipeline that
actively searches interpretable problem-family parameter spaces for ranking
reversals between two optimizer configurations and certifies them on disjoint
fresh seeds under a serialized fairness contract.
