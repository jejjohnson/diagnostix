"""Diagnostics of columns: stratification, mixed and boundary layers, integrals.

A column is a field on levels along one axis; the levels are shared, or each
column has its own. Functions here take plain arrays and levels in metres.
"""

from __future__ import annotations

from diagnostix._src.columns.boundary_layer import (
    boundary_layer_height,
    bulk_richardson,
)
from diagnostix._src.columns.crossing import first_crossing
from diagnostix._src.columns.derivatives import (
    buoyancy_frequency_squared,
    lapse_rate,
    vertical_derivative,
)
from diagnostix._src.columns.integrals import column_integral
from diagnostix._src.columns.mixed_layer import mixed_layer_depth


__all__ = [
    "boundary_layer_height",
    "bulk_richardson",
    "buoyancy_frequency_squared",
    "column_integral",
    "first_crossing",
    "lapse_rate",
    "mixed_layer_depth",
    "vertical_derivative",
]
