"""Diagnostics of a horizontal velocity on collocated grids, plane or sphere.

Vorticity, divergence, strain, Okubo–Weiss and geostrophic velocity are
gridax's, on the same grids; what is here has no name there.
"""

from __future__ import annotations

from diagnostix._src.kinematics.geostrophy import (
    ageostrophic_velocity,
    geostrophic_residual,
)
from diagnostix._src.kinematics.gradient import (
    VelocityGradient,
    curvature_vorticity,
    shear_vorticity,
    velocity_gradient,
)
from diagnostix._src.kinematics.scalars import advection, frontogenesis


__all__ = [
    "VelocityGradient",
    "advection",
    "ageostrophic_velocity",
    "curvature_vorticity",
    "frontogenesis",
    "geostrophic_residual",
    "shear_vorticity",
    "velocity_gradient",
]
