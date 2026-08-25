"""Runner layer: contract-bound execution and aggregation."""

from optinemesis.runners.execution import (
    PairedOutcome,
    execute_paired_instance,
    execute_run,
    normalized_regret,
)

__all__ = [
    "PairedOutcome",
    "execute_paired_instance",
    "execute_run",
    "normalized_regret",
]
