"""Slots of the year: the calendar day of a date, and the checks on slots."""

from __future__ import annotations

from typing import Any

import numpy as np
from pipekit_array.functions import elementwise

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["calendar_day", "checked_slots"]

#: The days of each month of a leap year, and those before each month.
MONTH_DAYS = np.asarray([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
DAYS_BEFORE = np.cumsum(MONTH_DAYS) - MONTH_DAYS


@elementwise()
def calendar_day(month: ArrayLike, day: ArrayLike) -> Any:
    """The day of a date on a leap-year calendar, the same in every year.

    29 February is 60 and 1 March is 61 in every year, so a date keeps its slot
    of a daily climatology; the day of the year does not.

    Args:
        month: The month, 1 to 12, concrete integers.
        day: The day of the month, 1 to 31, broadcast against ``month``.

    Returns:
        The calendar day, 1 to 366, int64: a NumPy array, or a DataArray where
        an argument is one.

    Raises:
        ValidationError: If an argument is not of integers, or a day is not one
            of its month.

    Examples:
        >>> calendar_day([2, 3, 12], [29, 1, 31])
        array([ 60,  61, 366])
        >>> int(calendar_day(1, 1))
        1
    """
    month, day = np.asarray(month), np.asarray(day)
    for name, values in (("month", month), ("day", day)):
        if not np.issubdtype(values.dtype, np.integer):
            msg = f"'{name}' must be of integers, got dtype {values.dtype}"
            raise ValidationError(msg)
    month, day = np.broadcast_arrays(month, day)
    no_month = (month < 1) | (month > 12)
    if no_month.any():
        msg = f"'month' must be in 1 to 12, got {month[no_month].flat[0]}"
        raise ValidationError(msg)
    index = month.astype(np.int64) - 1
    no_day = (day < 1) | (day > MONTH_DAYS[index])
    if no_day.any():
        msg = (
            f"'day' must be a day of its month, got {day[no_day].flat[0]} of month "
            f"{month[no_day].flat[0]}"
        )
        raise ValidationError(msg)
    # the days before the month, and the day
    return DAYS_BEFORE[index] + day.astype(np.int64)


def checked_slots(slot: ArrayLike, n_slots: int) -> np.ndarray:
    """The slot of each sample as a NumPy array of integers in 1 … n_slots.

    Raises:
        ValidationError: If ``slot`` is not a 1-D array of integers, or one is
            not in 1 … n_slots.
    """
    slots = np.asarray(slot)
    if slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer):
        msg = (
            "'slot' must be a 1-D array of integers, got shape "
            f"{slots.shape} and dtype {slots.dtype}"
        )
        raise ValidationError(msg)
    outside = (slots < 1) | (slots > n_slots)
    if outside.any():
        msg = (
            f"'slot' must be in 1 to {n_slots}, got {slots[outside][0]} at position "
            f"{int(np.argmax(outside))}"
        )
        raise ValidationError(msg)
    return slots.astype(np.int64)
