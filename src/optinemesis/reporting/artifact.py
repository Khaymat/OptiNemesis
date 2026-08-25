"""Schema-versioned JSON reproduction artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optinemesis import SCHEMA_VERSION_ARTIFACT
from optinemesis.core.contract import StudySpec
from optinemesis.core.errors import OptinemesisError
from optinemesis.core.fingerprint import core_module_fingerprint, fingerprint
from optinemesis.core.results import (
    SEARCH_STAGE_LABEL,
    CandidateOutcome,
    SearchResult,
    ValidationResult,
)
from optinemesis.core.seeds import DERIVATION_RECIPE, SUBTREE_LABELS, SeedTree

REQUIRED_ARTIFACT_KEYS = (
    "artifact_schema_version",
    "created_utc",
    "contract",
    "seed_tree",
    "reference_direction",
    "search_stage",
    "validation_stage",
    "fingerprints",
)


class ArtifactError(OptinemesisError):
    """Raised when an artifact is malformed or version-incompatible."""


def _candidate_to_dict(outcome: CandidateOutcome) -> dict[str, Any]:
    return {
        "theta": list(outcome.theta),
        "dimension": outcome.dimension,
        "effect": outcome.effect,
        "effect_ci_low": outcome.effect_ci_low,
        "effect_ci_high": outcome.effect_ci_high,
        "p_value_raw": outcome.p_value_raw,
        "p_value_holm": outcome.p_value_holm,
        "test_name": outcome.test_name,
        "n_validation_seeds": outcome.n_validation_seeds,
        "median_regret_a": outcome.median_regret_a,
        "median_regret_b": outcome.median_regret_b,
        "median_gap": outcome.median_gap,
        "median_gap_ci_low": outcome.median_gap_ci_low,
        "median_gap_ci_high": outcome.median_gap_ci_high,
        "probability_superiority": outcome.probability_superiority,
        "criteria": {
            "holm_significant": outcome.criteria.holm_significant,
            "effect_threshold_met": outcome.criteria.effect_threshold_met,
            "ci_excludes_margin": outcome.criteria.ci_excludes_margin,
            "reference_direction_opposed": outcome.criteria.reference_direction_opposed,
        },
        "failure_reasons": list(outcome.failure_reasons),
    }


def build_artifact(
    study: StudySpec,
    search_result: SearchResult,
    validation_result: ValidationResult,
) -> dict[str, Any]:
    """Assemble the complete reproduction record (no executable content)."""
    reference_theta = study.family.center_theta()
    contract_dict = study.bind(reference_theta).to_dict()
    tree = SeedTree.create(int(study.seeds.root_entropy))
    registry_state = fingerprint(
        {"families_registered": _registered_family_keys(), "derivation": DERIVATION_RECIPE}
    )
    return {
        "artifact_schema_version": SCHEMA_VERSION_ARTIFACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract_dict,
        "seed_tree": {
            "root_entropy": tree.root_entropy,
            "derivation_recipe": DERIVATION_RECIPE,
            "subtree_labels": SUBTREE_LABELS,
        },
        "reference_direction": {
            "sample_size": search_result.reference_direction.sample_size,
            "median_gap": search_result.reference_direction.median_gap,
            "sign": search_result.reference_direction.sign,
            "note": (
                "Aggregate median gap over the neutral theta sample of the "
                "search stage; exploratory context only."
            ),
        },
        "search_stage": {
            "label": SEARCH_STAGE_LABEL,
            "evaluations": [
                {
                    "theta": list(e.theta),
                    "effect": e.effect,
                    "lcb": e.lcb,
                    "se": e.se,
                    "median_regret_a": e.median_regret_a,
                    "median_regret_b": e.median_regret_b,
                    "probability_superiority": e.probability_superiority,
                    "nominated": e.nominated,
                }
                for e in search_result.evaluations
            ],
        },
        "validation_stage": {
            "holm_family_alpha": validation_result.holm_family_alpha,
            "confirmed": [_candidate_to_dict(c) for c in validation_result.confirmed],
            "failed": [_candidate_to_dict(c) for c in validation_result.failed],
        },
        "fingerprints": {
            "core_module": core_module_fingerprint(),
            "registry": registry_state,
            "contract_id": contract_dict.get("contract_id", ""),
        },
    }


def _registered_family_keys() -> list[str]:
    from optinemesis.families.registry import list_families

    return list(list_families())


def save_artifact(
    study: StudySpec,
    search_result: SearchResult,
    validation_result: ValidationResult,
    path: str | Path,
) -> dict[str, Any]:
    artifact = build_artifact(study, search_result, validation_result)
    destination = Path(path)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return artifact


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate an artifact file (lightweight checks).

    Full JSON-Schema validation against ``schemas/artifact.schema.json`` is left
    to consumers who have ``jsonschema`` installed; this loader enforces only
    structural invariants needed for reconstruction.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ArtifactError("artifact root must be a JSON object")
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in data]
    if missing:
        raise ArtifactError(f"artifact missing required keys: {missing}")
    version = data["artifact_schema_version"]
    if version != SCHEMA_VERSION_ARTIFACT:
        raise ArtifactError(
            f"unsupported artifact schema version {version!r}; "
            f"this build supports {SCHEMA_VERSION_ARTIFACT!r}"
        )
    contract = data["contract"]
    if contract.get("schema_version") != "1":
        raise ArtifactError(
            f"unsupported contract schema version {contract.get('schema_version')!r}"
        )
    return data


def load_study_from_artifact(data: dict[str, Any]) -> StudySpec:
    """Rebuild a runnable :class:`StudySpec` from a loaded artifact."""
    contract = data["contract"]
    family = contract["family"]
    from optinemesis.core.bounds import Bounds
    from optinemesis.core.contract import (
        EnvironmentInfo,
        FamilyRef,
        MetricPlan,
        OptimizerSpec,
        ParamRange,
        SeedPlan,
        Thresholds,
    )

    optimizers = tuple(
        OptimizerSpec(
            name=name,
            implementation=payload["implementation"],
            config=dict(payload.get("config", {})),
            seeded=bool(payload.get("seeded", True)),
        )
        for name, payload in sorted(contract["optimizers"].items())
    )
    environment = None
    env_payload = contract.get("environment")
    if env_payload:
        environment = EnvironmentInfo(
            python=str(env_payload["python"]),
            numpy_version=str(env_payload["numpy"]),
            scipy_version=str(env_payload["scipy"]),
            optinemesis_version=str(env_payload["optinemesis"]),
            platform=str(env_payload["platform"]),
        )
    seeds_payload = contract["seeds"]
    return StudySpec(
        family=FamilyRef(
            name=family["name"],
            version=family["version"],
            parameter_box=tuple(
                ParamRange(name=p["name"], lower=p["lower"], upper=p["upper"])
                for p in family.get("parameter_box", [])
            ),
            transform_tags=tuple(family.get("transform_tags", ())),
        ),
        dimension=len(contract["bounds"]["lower"]),
        bounds=Bounds.from_spec(contract["bounds"]),
        budget=int(contract["budget"]),
        optimizers=(optimizers[0], optimizers[1]),
        seeds=SeedPlan(
            root_entropy=str(seeds_payload["root_entropy"]),
            n_theta_candidates=int(seeds_payload.get("n_theta_candidates", 64)),
            n_search_seeds=int(seeds_payload["n_search_seeds"]),
            n_validation_seeds=int(seeds_payload["n_validation_seeds"]),
            meta_search_method=str(seeds_payload.get("meta_search_method", "lhs")),
        ),
        metrics=MetricPlan(
            primary=contract["metrics"]["primary"],
            exploratory=tuple(contract["metrics"].get("exploratory", ())),
        ),
        thresholds=Thresholds(
            epsilon_min=float(contract["thresholds"]["epsilon_min"]),
            epsilon_zero=float(contract["thresholds"]["epsilon_zero"]),
            alpha_search=float(contract["thresholds"].get("alpha_search", 0.05)),
            alpha_validation=float(contract["thresholds"]["alpha_validation"]),
            bootstrap_resamples=int(contract["thresholds"].get("bootstrap_resamples", 10_000)),
        ),
        initialization_policy=str(contract["initialization_policy"]),
        environment=environment,
    )


def replay_instance(
    artifact: dict[str, Any], theta_index: int, instance_slot: int
) -> Any:
    """Reconstruct one search-stage problem instance bit-exactly.

    Uses the documented derivation recipe: the theta at ``theta_index`` of the
    recorded search evaluations and the ``instance_slot``-th problem-search
    leaf seed reproduce the exact instance used during the original run
    (same package versions assumed).
    """
    from optinemesis.core.bounds import Bounds
    from optinemesis.families.registry import get_family

    contract = artifact["contract"]
    family_ref = contract["family"]
    evaluation = artifact["search_stage"]["evaluations"][theta_index]
    tree = SeedTree.create(int(artifact["seed_tree"]["root_entropy"]))
    instance_seed = tree.problem_search.seeds(instance_slot + 1)[instance_slot]
    family = get_family(str(family_ref["name"]), str(family_ref["version"]))
    return family.sample(
        theta=tuple(float(t) for t in evaluation["theta"]),
        dimension=len(contract["bounds"]["lower"]),
        instance_seed=instance_seed,
        bounds=Bounds.from_spec(contract["bounds"]),
    )
