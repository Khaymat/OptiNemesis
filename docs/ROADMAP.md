# OptiNemesis — Roadmap (v0.1 only)

This roadmap describes v0.1 exclusively. It deliberately does not promise v0.2,
v0.3, or 1.0 features. Anything not listed here is out of scope until this
document is revised.

## v0.1 scope

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Spec freeze: DESIGN/ARCHITECTURE/CLAIMS/ROADMAP + JSON schemas (v1) | done at commit of these files |
| 1 | Core: frozen types, Contract, SeedTree, CountingObjective, BudgetExhausted, validation of bounds/dim/config | planned |
| 2 | Families: Ellipsoidal, MultimodalBlend, DeceptiveDoubleWell; tagged transforms; optima tests | planned |
| 3 | Execution: Optimizer protocol, SeededRandomSearch baseline, scipy adapter, exact budget enforcement, compliance metadata | planned |
| 4 | Statistics: median/IQR, PS, rank-biserial, percentile bootstrap, Wilcoxon/permutation, Holm | planned |
| 5 | Deterministic keystone ranking-reversal fixture exercising the full pipeline | planned |
| 6 | Random/LHS meta-search, optimism-penalized nomination, disjoint-seed holdout validation; failed validations first-class | planned |
| 7 | Schema-versioned JSON artifact + conservative Markdown report | planned |
| 8 | Diagnostics-lite: parameter sensitivity marginals, transformation ablation | planned |

## v0.1 quality bar

- pytest suite covering the testing standard (see DESIGN and session plan);
- ruff clean; mypy on `core` (best effort elsewhere);
- GitHub Actions CI on supported Python versions;
- package builds (`python -m build`) and imports cleanly.

## Explicitly NOT in v0.1

Multiobjective optimization; discrete/combinatorial domains; constraints or noisy
objectives; GPU/distributed execution; learned, neural, symbolic, or noise-based
problem generation; Bayesian/CMA/evolutionary meta-search; in-loop hyperparameter
tuning; solver recommendation; web dashboards; adapters beyond scipy;
PyPI publication; version tags/releases. Each requires separate approval.
