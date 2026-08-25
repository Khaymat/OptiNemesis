# OptiNemesis — Architecture (Phase 0 Freeze)

## 1. Dependency DAG

Strict layering; arrows may only point downward. Enforced by review and by a
dependency-lint test (`tests/test_architecture.py`) that scans imports.

```
                    ┌─────────────┐
                    │   reporting │  (leaf: artifact JSON, Markdown)
                    └─────────────┘
              ┌──────────┬───────────┬────────────┐
        ┌────▼────┐ ┌───▼────┐ ┌────▼─────┐ ┌────▼────────┐
        │ search  │ │validate│ │diagnostics│ │  adapters   │
        └────┬────┘ └───┬────┘ └────┬─────┘ └────┬────────┘
             │          │           │            │
        ┌────▼──────────▼───────────▼─────┐      │
        │      runners (execution)        │      │
        └────┬──────────┬─────────────────┘      │
        ┌────▼────┐ ┌───▼─────┐                   │
        │ families│ │  stats  │                   │
        └────┬────┘ └───┬─────┘                   │
        ┌────▼──────────▼─────┐                  │
        │        core         │◄─────────────────┘
        └─────────────────────┘   (adapters import core ONLY)
```

Layer responsibilities:

- **core** — frozen dataclasses and pure logic only:
  `Contract`, `Bounds`, `ThetaSpec`, `ProblemInstance`, `RunResult`,
  `SearchResult`, `ValidationResult` (type-level distinction), `Candidate`,
  seed-tree derivation, `CountingObjective`, `BudgetExhausted`,
  fingerprint helpers, registry protocol types.
  Dependencies: numpy, stdlib. Nothing else.
- **families** — registry + the three v0.1 families + shared transform stack.
  Imports core only.
- **runners** — budget enforcement plumbing, run orchestration
  (contract × instance-seeds × optimizer seeds), result aggregation,
  deterministic parallel shard merge. Imports core + families (+ stats).
- **stats** — effect sizes, bootstrap, tests, Holm. Pure functions on arrays;
  imports numpy/scipy + core value types.
- **search** — θ sampling (Random, scrambled LHS), candidate nomination with the
  optimism-penalized criterion. Imports runners + stats + core.
- **validate** — holdout orchestration over disjoint validation subtrees;
  produces `ValidationResult`. Imports runners + stats + core.
- **diagnostics** — parameter sensitivity marginals from search clouds;
  transformation ablation reruns. Optional extras (pflacco bridge is *not* in v0.1).
- **adapters** — optimizer protocol definition lives here (`adapters/protocol.py`);
  `adapters/scipy.py`; `SeededRandomSearch` baseline lives in
  `adapters/builtin.py`. **Import core only.** They receive runner-owned objects
  as arguments; they never import runners upward. Third-party optimizers enter
  exclusively through this layer via entry points.
- **reporting** — schema-versioned JSON artifact writer/loader; conservative
  Markdown renderer requiring a `ValidationResult`. Leaf module.

Forbidden (checked by test): any import of `optinemesis.reporting` from below it;
`families` importing `runners`; circular imports of any kind; adapters importing
anything but `core` (+ their third-party library); `stats` importing runners.

## 2. Public / private boundary

Public (documented, semver-relevant from v0.1):
- `optinemesis.core`: all frozen dataclasses, `SeedTree`, `CountingObjective`,
  `BudgetExhausted`, `ContractError`.
- `optinemesis.families`: registry access `get_family(name)`, the three family
  classes, `ProblemFamily` base.
- `optinemesis.adapters`: `Optimizer` protocol, `RunResult` re-export,
  `register_optimizer` entry-point hook, builtins.
- `optinemesis.search`: `sample_thetas_random`, `sample_thetas_lhs`, `nominate_candidates`.
- `optinemesis.validate`: `validate_candidates`.
- `optinemesis.runners`: `execute_theta_pair`, `aggregate`.
- `optinemesis.stats`: pure statistic functions.
- `optinemesis.diagnostics`: sensitivity/ablation entry points.
- `optinemesis.reporting`: `save_artifact`, `load_artifact`, `render_markdown`.
- Top-level convenience façade `optinemesis.api.run_counterexample_study`
  (thin composition of search→validation; introduced in Phase 6).

Private in v0.1 (may change without notice): parallel executor internals,
checkpoint format details, metric registry extension mechanism (closed enum until
schema stabilizes), anything under `_internal.py` modules.

## 3. Packaging

- src-layout: `src/optinemesis/…`.
- Runtime deps: numpy >=1.24, scipy >=1.10.
- Optional extras: none in v0.1 beyond `[dev]` (pytest, ruff, mypy, hypothesis, build).
- Entry point group reserved: `optinemesis.optimizers` for third-party adapters.

## 4. Data flow (one study)

```
Contract + Optimizer configs + Family spec
        │
        ▼
   SeedTree(root)                      meta subtree
        │                                    │
        ▼                                    ▼
  sample Θ candidates ──────► for each θ: paired runs on search subtrees
                                             │
                                             ▼
                                  per-θ effect estimate + bootstrap LCB
                                             │
                                   nominate candidates (LCB ≥ ε_min)
                                             │
                                             ▼
                          validate_candidates on disjoint validation subtree
                                             │
                             ┌───────────────┴───────────────┐
                             ▼                               ▼
                     ValidationResult.confirmed      ValidationResult.failed
                             └───────────────┬───────────────┘
                                             ▼
                              artifact JSON + Markdown report
```

Every arrow is deterministic given root entropy and package versions.
