# OptiNemesis — Design Specification (Phase 0 Freeze)

**Scientific positioning:** *A framework for validated counterexample discovery in
optimization algorithm comparisons.*

"Nemesis" is branding only. Every scientific claim, artifact field, report line, and
documentation sentence must use conservative contract-scoped language
(see `CLAIMS.md` — it is normative and overrides any other document if they conflict).

- Schema versions frozen by this document: `contract.schema.json` version `1`,
  `artifact.schema.json` version `1`.
- Status of this document: normative for v0.1. Changes require a schema-version bump
  or an explicit `DESIGN-CHANGES.md` entry.

---

## 1. Terminology

| Term | Meaning |
|---|---|
| **Problem family** | A named, parameterized generator of optimization problem instances built from interpretable transformations of classical functions. |
| **Parameter vector (θ)** | A point in the family's bounded, interpretable parameter box Θ ⊂ R^k. |
| **Instance seed (`s`)** | Seed controlling stochastic parts of instance generation (rotations, shifts). Same `(family, θ, d, s)` ⇒ bit-identical instance on a fixed NumPy version. |
| **Optimizer configuration** | A frozen, serializable description of an optimizer *and its hyperparameters*. Comparisons are always between configurations, never reified "algorithms". |
| **Run** | One optimizer configuration executing once on one problem instance under one budget with one optimizer seed. |
| **Contract** | The immutable set of experimental conditions (§4). Serialized into every artifact. |
| **Search stage** | Exploratory evaluation of many θ candidates under a fixed contract. Search statistics are **never citable evidence**; they only nominate candidates. |
| **Validation stage** | Confirmatory re-evaluation of nominated candidates on a completely disjoint seed hierarchy. Only validation statistics may appear in confirmatory claims. |
| **Candidate** | A θ that passed the search-stage optimism-penalized nomination threshold. A candidate is *never* a finding. |
| **Validated counterexample** | A candidate whose fresh-seed validation satisfied significance, effect-size, and practical-margin criteria. This is the strongest object OptiNemesis can emit. |
| **Failed validation** | A candidate that did not survive holdout. First-class output; retained in artifacts and reports. |
| **Reversal reference direction** | The aggregate sign of the performance gap over a neutral reference sample of Θ. A "reversal" is only meaningful relative to this direction. |

## 2. Mathematical formulation

Family `F`, parameters `θ ∈ Θ`, dimension `d`, instance seed `s`. Instance:

```
f = F.sample(θ, d, s)          # pure function, box bounds [lo, hi]^d, tracked optimum value f★
```

Normalized clipped regret for optimizer configuration `i` run `r` at budget `B`:

```
ρ_i(θ, s, r) = clip( ( ŷ_i − f★ ) / scale(θ, d) , 0.0, ρ_MAX )
```

- `ŷ_i` = final best objective value of the run.
- `scale(θ, d)` = family-declared positive normalizer (e.g., objective range over the
  box or a documented reference quantity). Purpose: make gaps comparable across θ and d.
- `ρ_MAX = 1e6` (constant): prevents pathological blow-up from dominating statistics.

Paired per-instance difference (both configurations see identical instance seeds):

```
Δ(θ, s) = median_{r∈S_B} ρ_B(θ,s,r) − median_{r∈S_A} ρ_A(θ,s,r)
```

Primary statistic (frozen for v0.1): **rank-biserial effect size of paired regrets**
`δ ∈ [−1, 1]`, sign convention: `δ > 0` means configuration A has larger regret than B
(A underperformed) on paired comparisons.

Secondary statistics (reported, exploratory): median regret per configuration,
median gap `Δ̄`, probability of superiority `PS = P(ρ_A > ρ_B)`.

**Nomination criterion (search stage):** maximize the optimism-penalized estimate

```
LCB_towards_zero(θ) = δ̂(θ) − z_{1−α_s}·SE_boot  if δ̂≥0
                     = δ̂(θ) + z_{1−α_s}·SE_boot  if δ̂<0   (bound towards zero)
```

with bootstrap standard error `SE_boot = std(bootstrap replicates)` at level
`1−α_s` (`z` the standard normal quantile, default 95%).  The absolute
conservative magnitude `|LCB_towards_zero|` ranks candidates symmetrically so
swapping labels `A↔B` (which negates `δ`) preserves the nominated set up to
sign. Candidates advance iff `|LCB_towards_zero| ≥ ε_min`.

*Finite-sample degeneracy:* when all paired differences have the same sign,
`δ̂=±1` and every bootstrap replicate is `±1` ⇒ `SE_boot=0` and naive
`LCB=±1` overstates confidence for small `n`. In this perfect-separation case
OptiNemesis replaces the normal approximation with the exact Clopper-Pearson
lower bound for `p = P(ρ_A > ρ_B)` (binomial `k=n` successes): `L_p = α_s^{1/n}`,
mapped to rank-biserial as `L_δ = 2·L_p −1`. The signed conservative bound is
`sign(δ̂)·L_δ`. This is distribution-free, guarantees coverage, and penalizes
small-`n` perfect separation (e.g. `n=4, α=0.05 → L_δ≈−0.05` not nominated;
`n=10 → L_δ≈0.48` nominated). Otherwise the bootstrap normal bound is used.

*Search bootstrap cap:* `bootstrap_resamples` is 10 000 by default but the
search stage caps it at `min(500, bootstrap_resamples)` `src/optinemesis/search/engine.py:102`
for speed; validation always uses the full count. Thresholds are otherwise
identical.

**Confirmation criteria (validation stage), all mandatory and symmetric:**
1. Wilcoxon signed-rank test on matched pairs, Holm-corrected across all
   validated candidates at family-wise `α_v` (default 0.05);
2. validation rank-biserial `|δ̂_val| ≥ ε_min` (absolute, so swapping labels preserves the test);
3. validation bootstrap CI of the secondary gap excludes the practical-equivalence
   margin `ε₀` in the claimed direction: `CI_low > ε₀` if `δ̂_val>0`,
   `CI_high < −ε₀` if `δ̂_val<0`;
4. reversal: `sign(δ̂_val) ≠ 0`, `sign(reference_gap) ≠ 0`, and `sign(δ̂_val) ≠ sign(reference_gap)`.
   The reference direction is the sign of the median gap over the neutral
   `Θ` sample; a candidate is a reversal only when its effect sign opposes
   that reference. The absolute-value formulation makes swapping `A↔B`
   (which negates both `δ̂_val` and the reference gap) preserve confirmability.

A reversal claim additionally requires the reference direction condition (4).

## 3. Evaluation-budget semantics

- **Currency:** exact function evaluations. Nothing else.
- **Budget is a cap, not a quota** (COCO convention). Backends that genuinely
  converge before exhausting the budget may stop early *only* when
  `termination_reason` records backend convergence; v0.1 adapters minimize early
  stopping via zero tolerances wherever the backend permits. Comparisons under an
  equal cap remain fair because every run may use up to `B` evaluations.
- The harness owns the objective. Adapters receive a `CountingObjective` which
  increments a counter on every evaluation and raises `BudgetExhausted` when the
  budget would be exceeded.
- Overshoot policy: when the backend attempts an evaluation beyond budget, the
  exception propagates into the adapter, which must catch it, retain partial state,
  and return normally. `n_evals_consumed ≤ budget` always; attempted overshoot is
  recorded in `overshoot_events`. **Silent overshoot is forbidden**: any run where
  more than zero evaluations were attempted past budget carries a compliance flag.
- Wall-clock time is recorded as metadata and is never a fairness currency in v0.1.
  No claim in v0.1 may be runtime-based.
- Stopping criterion is budget exhaustion only. Optimizer-native early stopping must
  be disabled or recorded as a termination reason; v0.1 adapters disable it.

## 4. Fairness contract

The immutable `Contract` records, before execution:

- artifact/contract schema versions;
- family name + family parameter spec + θ bounds;
- dimension(s) `d` (each d is a separate experiment key);
- box bounds `[lo, hi]` (identical for all optimizers);
- evaluation budget `B` (integer, identical);
- initialization policy: `"independent"` (only supported value in v0.1; each
  configuration uses its own seeded initialization from disjoint `SeedTree`
  subtrees). `"paired"` was removed in the correctness rework because
  heterogeneous adapters (population-based `differential_evolution` vs
  single-point `L-BFGS-B`/`Nelder-Mead`/`Powell` vs `builtin.random_search`)
  have fundamentally different initialization mechanisms; faking identical
  initial conditions would be scientifically dishonest. Future paired support
  would require per-adapter injection contracts.
- seed counts: `n_search_seeds`, `n_validation_seeds`, meta-search sample count;
- primary metric name (frozen to `rank_biserial` for v0.1; `MetricPlan` validates but
  rejects other values until `epsilon_min` can be recalibrated per metric) and exploratory metrics list;
- practical thresholds: `ε_min` (effect), `ε₀` (equivalence margin), α levels;
- optimizer configuration specs (serialized verbatim).

Identical across optimizers (mandatory): instance seeds, θ, d, bounds, budget,
metric definitions, seed counts.
Not standardized (they are part of the compared configuration): internal population
sizes, learning rates, surrogate models, initialization mechanics.

**No post-hoc metric switching:** the artifact stores `primary_metric` separately from
exploratory metrics; reports render the primary first and mark others exploratory.

**Hyperparameter tuning inside the loop is forbidden in v0.1.** Configurations are
frozen inputs; tuning both sides equally is a future, separate experiment type.

## 5. Seed hierarchy

One user-supplied root entropy (int or None→OS entropy). Derived via
`numpy.random.SeedSequence(root).spawn()` into five disjoint subtrees:

```
root
├── meta            # meta-search sampling (θ candidates)
├── problem_search  # instance seeds used during search
├── opt_a           # optimizer-A seeds (search)
├── opt_b           # optimizer-B seeds (search)
└── validation      # spawns further: problem_val, opt_a_val, opt_b_val, holm ordering
```

Rules:
- No global RNG state anywhere. All randomness enters through explicit integer seeds
  derived from these trees.
- Validation subtrees are disjoint from all search subtrees **by construction**;
  sharing a seed across stages is impossible through the public API.
- Determinism guarantee: same root entropy + same package versions ⇒ identical
  instances, runs, and statistics, bit-for-bit modulo documented float caveats.
- Determinism limitation: SciPy/NumPy may change floating-point behavior across
  versions; artifacts therefore record exact package versions and the fingerprint of
  the installed `optinemesis` core module. Cross-version bit-reproduction is a
  non-goal; statistical reproduction (same qualitative result) is the goal.

## 6. Search vs validation distinction (architectural)

- `SearchResult` and `ValidationResult` are distinct types. A `SearchResult` cannot be
  rendered as a confirmed report: `render_report()` requires a `ValidationResult`.
- The public API offers no code path that converts a candidate into a validated
  counterexample without executing fresh-seed runs.
- Failed validations are retained in `ValidationResult.failed[]` and rendered in reports.

## 7. Practical-effect threshold handling

- `ε_min` (minimum effect size to care about) and `ε₀` (practical equivalence margin
  for the CI exclusion test) are declared in the Contract before execution.
- Defaults: `ε_min = 0.33` ("medium" rank-biserial), `ε₀ = 0.0` (CI must exclude zero
  gap) with documented guidance that domain-specific values should replace defaults.
- Thresholds appear in every claim sentence ("under the declared threshold …").

## 8. Problem-family principles

1. Families are compositions of **interpretable, tagged transformations** of
   established classical functions. Each transformation documents the landscape
   property it manipulates.
2. Banned in v0.1: arbitrary symbolic composition, neural/surrogate-generated
   landscapes, additive noise/jitter wrappers of any kind, constraint wrappers,
   boundary-penalty transforms, unbounded objective rescaling.
3. Optima: tracked analytically where available (shifted ellipsoid, shifted Rastrigin
   blend); by construction (double-well deep basin center); otherwise via a cached,
   seeded high-budget reference solve recorded in the artifact (v0.1 families all have
   analytic or constructive optima).
4. Parameter ranges are fixed per family, finite, and interpretable; the registry
   rejects θ outside the declared box.
5. Rotation uses seeded random Givens rotations; the rotation-strength parameter
   is interpreted as the fraction of principal coordinate planes rotated
   (identity at 0, dense rotation at 1). Distance from identity grows in
   expectation but is not monotone per seed; this is documented behavior.

## 9. Reproducibility guarantees and limitations

Guaranteed:
- Bit-exact instance reconstruction from `(family_name, family_version, θ, d, instance_seed)`
  on identical numpy/optinemesis versions.
- Byte-identical JSON artifacts (excluding wall-clock fields) for repeated executions
  with equal root entropy.
- Structural impossibility of single-seed findings (all reported statistics are
  distributions over ≥ `n_search_seeds` / `n_validation_seeds` runs).

Limitations (documented, not fixed):
- Floating-point reproducibility across numpy/scipy versions is not guaranteed.
- Wall-clock varies; runtime fields are informational.
- Optimizers whose backends ignore seeds are marked `seeded=false` in metadata; their
  runs are still budget-controlled but not bit-deterministic. v0.1 ships none.
- Parallel execution merges deterministically by sort key; scheduling order never
  affects results.

## 10. Statistical machinery

Use SciPy implementations for sensitive algorithms:
`scipy.stats.wilcoxon`, `scipy.stats.permutation_test` (or an exact
Mann–Whitney-based permutation scheme), `scipy.stats.rankdata` for effect sizes.
Bootstrap: percentile method implemented locally (simple, auditable) with a fixed
resample count (10,000) and spawned RNG. Holm correction implemented locally
(step-down procedure, ~15 lines, property-tested against manual computation).

## 11. Claim boundaries

Normative text lives in `CLAIMS.md`. Summary: all claims are of the form
"Under this serialized contract, on this validated region of this family,
configuration A underperformed configuration B." Nothing stronger is emitted by
the software anywhere (report templates enforce hedged phrasing).
