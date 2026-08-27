import json
from pathlib import Path

import numpy as np
import pytest

from optinemesis.core import Bounds, OptimizerSpec, SeedPlan, SeedTree, StudySpec, Thresholds
from optinemesis.core.results import SearchResult, ValidationResult
from optinemesis.families import get_family
from optinemesis.reporting import (
    ArtifactError,
    load_artifact,
    load_study_from_artifact,
    render_markdown,
    replay_instance,
    save_artifact,
)
from optinemesis.search import run_search
from optinemesis.validate import validate_candidates


def make_study(root_entropy: int = 8675309) -> StudySpec:
    family = get_family("Ellipsoidal", "1")
    return StudySpec(
        family=family.family_ref(),
        dimension=3,
        bounds=Bounds(lower=(-5.0, -5.0, -5.0), upper=(5.0, 5.0, 5.0)),
        budget=120,
        optimizers=(
            OptimizerSpec(name="rand", implementation="builtin.random_search"),
            OptimizerSpec(
                name="de",
                implementation="scipy.differential_evolution",
                config={"popsize": 4},
            ),
        ),
        seeds=SeedPlan(
            root_entropy=str(root_entropy),
            n_theta_candidates=3,
            n_search_seeds=2,
            n_validation_seeds=5,
            meta_search_method="lhs",
        ),
        thresholds=Thresholds(bootstrap_resamples=200),
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> tuple[StudySpec, str]:
    study = make_study()
    search = run_search(study)
    validation = validate_candidates(search, study)
    path = tmp_path_factory.mktemp("artifact") / "study.nemesis.json"
    save_artifact(study, search, validation, path)
    return study, str(path)


def test_artifact_roundtrip(completed: tuple[StudySpec, str]) -> None:
    _, path = completed
    data = load_artifact(path)
    assert data["artifact_schema_version"] == "1"
    assert data["contract"]["schema_version"] == "1"
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert set(raw) >= {
        "contract",
        "seed_tree",
        "reference_direction",
        "search_stage",
        "validation_stage",
        "fingerprints",
    }


def test_artifact_contains_no_code(completed: tuple[StudySpec, str]) -> None:
    raw = Path(completed[1]).read_text(encoding="utf-8")
    for forbidden in ("pickle", "b64:", "eval(", "__reduce__", "lambda "):
        assert forbidden not in raw


def test_study_reconstruction_roundtrip(completed: tuple[StudySpec, str]) -> None:
    original, path = completed
    rebuilt = load_study_from_artifact(load_artifact(path))
    assert rebuilt.family.name == original.family.name
    assert rebuilt.dimension == original.dimension
    assert rebuilt.budget == original.budget
    assert [o.name for o in rebuilt.optimizers] == [o.name for o in original.optimizers]
    assert rebuilt.seeds.root_entropy == original.seeds.root_entropy
    assert rebuilt.thresholds.epsilon_min == original.thresholds.epsilon_min
    # Order must be preserved (not alphabetically sorted)
    assert [o.name for o in rebuilt.optimizers] == ["rand", "de"]


def test_replayed_instance_matches_live_generation(
    completed: tuple[StudySpec, str],
) -> None:
    study, path = completed
    artifact = load_artifact(path)
    replayed = replay_instance(artifact, theta_index=0, instance_slot=0)
    theta = tuple(float(t) for t in artifact["search_stage"]["evaluations"][0]["theta"])
    instance_seed = (
        SeedTree.create(int(study.seeds.root_entropy)).problem_search.seeds(1)[0]
    )
    live = get_family(study.family.name).sample(
        theta=theta,
        dimension=study.dimension,
        instance_seed=instance_seed,
        bounds=study.bounds,
    )
    probe = np.array([0.3, -0.7, 1.1])
    assert replayed.evaluate(probe) == live.evaluate(probe)


def test_malformed_artifacts_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_artifact(bad)

    wrong_version = tmp_path / "wrong.json"
    wrong_version.write_text(json.dumps({"artifact_schema_version": "9"}), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_artifact(wrong_version)


def test_markdown_rejects_non_study_and_missing_validation() -> None:
    with pytest.raises(TypeError):
        render_markdown(object(), None, None)  # type: ignore[arg-type]


def test_markdown_conservative_language(completed: tuple[StudySpec, str]) -> None:
    study, _ = completed
    fresh_study = make_study(root_entropy=424242)
    search = run_search(fresh_study)
    validation = validate_candidates(search, fresh_study)

    report = render_markdown(study, search, validation)
    assert "not confirmatory evidence" in report
    assert "underperformed configuration" in report
    forbidden_claims = [
        "is better than optimizer",
        "fundamentally weak",
        "algorithm-independent truth",
    ]
    for claim in forbidden_claims:
        assert claim not in report.lower()


def test_validation_result_type_guard() -> None:
    vr_fields = dict(ValidationResult.__dataclass_fields__)  # type: ignore[attr-defined]
    sr_fields = dict(SearchResult.__dataclass_fields__)  # type: ignore[attr-defined]
    assert "confirmed" in vr_fields and "failed" in vr_fields
    assert "reference_direction" in sr_fields
    assert not hasattr(SearchResult, "confirmed")
