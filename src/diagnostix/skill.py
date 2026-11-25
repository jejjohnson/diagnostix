"""How good a prediction is, as functions that can be losses."""

from __future__ import annotations

from diagnostix._src.skill.pointwise import (
    bias,
    correlation,
    mae,
    nrmse_score,
    r2_score,
    rmse,
)


__all__ = [
    "bias",
    "correlation",
    "mae",
    "nrmse_score",
    "r2_score",
    "rmse",
]
