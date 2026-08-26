"""OptiNemesis: a framework for validated counterexample discovery in
optimization algorithm comparisons.

"Nemesis" is branding. Scientific claims produced by this software are strictly
contract-scoped; see ``docs/CLAIMS.md`` (normative) and ``docs/DESIGN.md``.

OptiNemesis does not claim to invent optimization benchmarking, adversarial
instance generation, Instance Space Analysis, parameterized benchmark
generation, or pairwise optimizer comparison.
"""

from optinemesis.api import CounterexampleStudyResult, run_counterexample_study

__version__ = "0.1.0.dev0"

SCHEMA_VERSION_CONTRACT = "1"
SCHEMA_VERSION_ARTIFACT = "1"

__all__ = ["CounterexampleStudyResult", "run_counterexample_study"]
