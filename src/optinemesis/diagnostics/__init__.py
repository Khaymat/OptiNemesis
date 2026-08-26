"""Diagnostics-lite: sensitivity marginals and transformation ablation."""

from optinemesis.diagnostics.ablation import ablate_candidate, neutral_theta_for_family
from optinemesis.diagnostics.sensitivity import sensitivity_marginals

__all__ = ["ablate_candidate", "neutral_theta_for_family", "sensitivity_marginals"]
