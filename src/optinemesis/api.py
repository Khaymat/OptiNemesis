"""Top-level facade for counterexample-discovery studies."""

from __future__ import annotations

from dataclasses import dataclass

from optinemesis.core.contract import StudySpec
from optinemesis.core.results import SearchResult, ValidationResult
from optinemesis.search import run_search
from optinemesis.validate import validate_candidates


@dataclass(frozen=True)
class CounterexampleStudyResult:
    study: StudySpec
    search: SearchResult
    validation: ValidationResult


def run_counterexample_study(study: StudySpec) -> CounterexampleStudyResult:
    """Run the full pipeline: exploratory search + mandatory fresh-seed validation."""
    search = run_search(study)
    validation = validate_candidates(search, study)
    return CounterexampleStudyResult(study=study, search=search, validation=validation)
