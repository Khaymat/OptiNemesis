"""Problem families: interpretable, transformed classical foundations."""

from optinemesis.families.base import FamilyBuild, ProblemFamily
from optinemesis.families.doublewell import DeceptiveDoubleWellFamily
from optinemesis.families.ellipsoidal import EllipsoidalFamily
from optinemesis.families.multimodal import MultimodalBlendFamily
from optinemesis.families.registry import (
    clear_registry,
    get_family,
    list_families,
    register_family,
)

__all__ = [
    "DeceptiveDoubleWellFamily",
    "EllipsoidalFamily",
    "FamilyBuild",
    "MultimodalBlendFamily",
    "ProblemFamily",
    "clear_registry",
    "get_family",
    "list_families",
    "register_family",
]
