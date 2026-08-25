"""Stable fingerprints for contracts and artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(obj: Any) -> str:
    """Serialize to a stable JSON string with normalized numeric types."""

    def default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"not JSON-serializable: {type(o)!r}")

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
        allow_nan=False,
    )


def fingerprint(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON serialization."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def core_module_fingerprint() -> str:
    """Fingerprint over the public constants of the installed optinemesis package.

    This is a coarse change detector (schema versions and version string), not a
    full source hash; the artifact also records the package version.
    """
    from optinemesis import (
        SCHEMA_VERSION_ARTIFACT,
        SCHEMA_VERSION_CONTRACT,
        __version__,
    )

    return fingerprint(
        {
            "version": __version__,
            "contract_schema": SCHEMA_VERSION_CONTRACT,
            "artifact_schema": SCHEMA_VERSION_ARTIFACT,
        }
    )
