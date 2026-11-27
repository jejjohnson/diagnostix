"""Scores of trajectories: separation from an observed one, skill, dispersion.

They take arrays of positions, such as the ``trajectory`` that
`tracerax.advect.integrate` saves, and need no run.
"""

from __future__ import annotations

from diagnostix._src.trajectories.dispersion import (
    absolute_dispersion,
    pair_dispersion,
)
from diagnostix._src.trajectories.distance import Distance
from diagnostix._src.trajectories.separation import (
    cumulative_separation,
    separation,
    trajectory_skill,
)


__all__ = [
    "Distance",
    "absolute_dispersion",
    "cumulative_separation",
    "pair_dispersion",
    "separation",
    "trajectory_skill",
]
