"""Diagnostics of a horizontal velocity on collocated grids, plane or sphere.

Vorticity, divergence, strain, Okubo–Weiss and geostrophic velocity are
gridax's, on the same grids; what is here has no name there.
"""

from __future__ import annotations

from diagnostix._src.kinematics.gradient import (
    VelocityGradient,
    curvature_vorticity,
    shear_vorticity,
    velocity_gradient,
)


__all__ = [
    "VelocityGradient",
    "curvature_vorticity",
    "shear_vorticity",
    "velocity_gradient",
]
