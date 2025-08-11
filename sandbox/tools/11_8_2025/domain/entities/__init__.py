"""
Domain entities for the enactive consciousness system.

Entities represent core business objects with identity that persist
across the system lifecycle. They encapsulate critical business rules
and maintain consistency of core domain concepts.
"""

from .predictive_coding_core import PredictiveCodingCore
from .self_organizing_map import SelfOrganizingMap

__all__ = [
    "PredictiveCodingCore",
    "SelfOrganizingMap",
]