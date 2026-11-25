"""How good a prediction is, as functions that can be losses."""

from __future__ import annotations

from diagnostix._src.skill.ensemble import (
    SpreadSkill,
    crps,
    energy_score,
    spread_skill,
)
from diagnostix._src.skill.physical import (
    density_inversion_fraction,
    geostrophic_imbalance,
    qg_balance_residual,
    rms_divergence,
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
from diagnostix._src.skill.spectral import (
    PSDScore,
    band_rmse,
    psd_score,
    resolved_scale,
)


__all__ = [
    "PSDScore",
    "Reliability",
    "SpreadSkill",
    "band_rmse",
    "bias",
    "correlation",
    "coverage",
    "crps",
    "density_inversion_fraction",
    "energy_score",
    "geostrophic_imbalance",
    "mae",
    "nrmse_score",
    "psd_score",
    "qg_balance_residual",
    "r2_score",
    "rank_histogram",
    "reliability",
    "resolved_scale",
    "rms_divergence",
    "rmse",
    "spread_skill",
]
