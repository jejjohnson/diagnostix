"""A field's annual cycle, and what is left without it.

Samples are grouped by integer slots of the year; a daily climatology uses a
leap-year calendar, on which a date has one slot in every year.
"""

from __future__ import annotations

from diagnostix._src.climate.calendar import calendar_day
from diagnostix._src.climate.climatology import (
    Climatology,
    climatology,
    harmonic_climatology,
    smooth_climatology,
)


__all__ = [
    "Climatology",
    "calendar_day",
    "climatology",
    "harmonic_climatology",
    "smooth_climatology",
]
