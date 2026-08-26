# OptiNemesis

> **Find the problems your optimizer doesn't want to meet.**

*Scientific positioning: a framework for validated counterexample discovery in optimization algorithm comparisons.*

"Nemesis" is branding. Every scientific claim produced by this software is strictly **contract-scoped**; see [`docs/CLAIMS.md`](docs/CLAIMS.md) (normative).

OptiNemesis does **not** claim to invent optimization benchmarking (COCO/BBOB, IOHprofiler), parameterized benchmark generation (MA-BBOB, W-model), adversarial/hard instance generation, Instance Space Analysis / algorithm footprints (MATILDA), pairwise win-rate comparison (Nevergrad), or exploratory landscape analysis (flacco/pflacco).

## What is OptiNemesis?

Given two optimizer *configurations* and a parameterized family of optimization problems, OptiNemesis searches the family's parameter space for regions where the ranking between the two configurations **reverses**, then certifies or rejects each candidate on **completely fresh seeds** under a serialized experimental contract.

A *validated counterexample* is a parameter vector where configuration A underperformed configuration B on held-out instances, with a bootstrap confidence interval, a Holm-corrected test, and an effect-size threshold — all recorded in a reproducible JSON artifact. A *failed validation* is retained with equal prominence. Negative results are first-class output.

## What it is NOT

- A new benchmark suite or leaderboard.
- An optimizer library (it adapts `scipy.optimize` and a trivial `builtin.random_search`; other ecosystems are future extras).
- A solver recommender, an auto-tuner, or a claim that any optimizer is "better" in general.
- A runtime-fair comparison tool (budget = function evaluations; wall-clock is metadata only in v0.1).

## Why mandatory holdout validation exists

Searching many parameter vectors and picking the best observed gap suffers from the **winner's curse**: the maximum of many noisy estimates is biased upward. A discovered gap that looks large on its discovery seeds may vanish on fresh seeds. OptiNemesis makes the holdout stage **architecturally mandatory**: `SearchResult` (exploratory, never citable) and `ValidationResult` (confirmatory, Holm-corrected, fresh-seed) are distinct types, and the Markdown renderer requires a `ValidationResult`. Single-seed findings are structurally impossible.

## Minimal example

```python
from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, StudySpec
from optinemesis.families import get_family
from optinemesis.api import run_counterexample_study
from optinemesis.reporting import save_artifact, render_markdown

study = StudySpec(
    family=get_family("Ellipsoidal").family_ref(),
    dimension=6,
    bounds=Bounds(lower=(-5.0,)*6, upper=(5.0,)*6),
    budget=800,
    optimizers=(
        OptimizerSpec(name="de", implementation="scipy.differential_evolution", config={"popsize": 10}),
        OptimizerSpec(name="lbfgs", implementation="scipy.l_bfgs_b"),
    ),
    seeds=SeedPlan(root_entropy="0", n_theta_candidates=32, n_search_seeds=20, n_validation_seeds=80),
)

result = run_counterexample_study(study)

# result.search is exploratory; result.validation is confirmatory
print(f"candidates: {len(result.search.nominated)} -> confirmed: {len(result.validation.confirmed)}")

save_artifact(result.study, result.search, result.validation, "study.nemesis.json")
Path("report.md").write_text(render_markdown(result.study, result.search, result.validation))
```

Every number in the artifact and report is tied to the serialized contract and the hierarchical `SeedSequence` derivation recipe (`seed_tree.derivation_recipe`).

## Current maturity

**Prototype — not production-ready.** The core pipeline (three families, budget accounting, statistics with SciPy-backed tests, LHS/random search, fresh-seed validation, JSON artifact, Markdown report, diagnostics-lite) is implemented and covered by tests, including a deterministic end-to-end reversal fixture. `0.1.0.dev0` on `main`; no PyPI publication and no version tag yet.

## Documentation

- `docs/DESIGN.md` — frozen terminology, formulation, fairness contract, seed hierarchy.
- `docs/ARCHITECTURE.md` — strict dependency DAG and public/private boundaries.
- `docs/CLAIMS.md` — allowed vs forbidden claim templates (normative).
- `docs/ROADMAP.md` — v0.1 scope and explicit non-goals.
- `schemas/` — versioned JSON Schemas for the contract and the artifact.

## Development

```bash
python -m venv .venv && .venv/Scripts/activate  # Windows: .venv\Scripts\python
pip install -e ".[dev]"
ruff check src tests && mypy src/optinemesis && pytest -q
python -m build
```

License: MIT. See [LICENSE](LICENSE).
