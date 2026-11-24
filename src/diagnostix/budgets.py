"""Budgets over a true control volume: fluxes through faces, and what is left over.

A region of a collocated grid has its own boundary faces; the flux through them
equals the area sum of the flux divergence inside, to round-off.
"""

from __future__ import annotations

from diagnostix._src.budgets.budget import (
    Budget,
    budget,
    cell_residual,
    region_integral,
)
from diagnostix._src.budgets.flux import (
    boundary_flux,
    face_fluxes,
    flux_divergence,
)
from diagnostix._src.budgets.volume import ControlVolume


__all__ = [
    "Budget",
    "ControlVolume",
    "boundary_flux",
    "budget",
    "cell_residual",
    "face_fluxes",
    "flux_divergence",
    "region_integral",
]
