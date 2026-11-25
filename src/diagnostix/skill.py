"""How good a prediction is, as functions that can be losses."""

from __future__ import annotations

from diagnostix._src.skill.ensemble import (
    SpreadSkill,
    crps,
    energy_score,
    spread_skill,
)
from diagnostix._src.skill.pointwise import (
    bias,
    correlation,
    mae,
    nrmse_score,
    r2_score,
    rmse,
)
from diagnostix._src.skill.ranks import (
    Reliability,
    coverage,
    rank_histogram,
    reliability,
)


__all__ = [
    "Reliability",
    "SpreadSkill",
    "bias",
    "correlation",
    "coverage",
    "crps",
    "energy_score",
    "mae",
    "nrmse_score",
    "r2_score",
    "rank_histogram",
    "reliability",
    "rmse",
    "spread_skill",
]
