"""Adapters: optimizer protocol, built-in baseline, scipy bridges.

This layer imports only ``optinemesis.core`` inside the package.
"""

from optinemesis.adapters import scipy_adapter
from optinemesis.adapters.builtin import SeededRandomSearch
from optinemesis.adapters.protocol import (
    Optimizer,
    build_optimizer,
    list_implementations,
    register_optimizer_factory,
)

_ = scipy_adapter  # keep module reference for clarity

__all__ = [
    "Optimizer",
    "SeededRandomSearch",
    "build_optimizer",
    "list_implementations",
    "register_optimizer_factory",
]
