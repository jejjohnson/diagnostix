"""Anomalies from a climatology, and their inverse."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from pipekit_array.functions import floating_inputs, table_like

from diagnostix._src.climate.calendar import checked_slots
from diagnostix._src.climate.climatology import Climatology, check_slots_axis
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["add_climatology", "anomaly"]


def anomaly(
    x: ArrayLike, climatology: Climatology, slot: ArrayLike, *, axis: int = -1
) -> Any:
    """What is left of each sample without the climatology of its slot.

    x′(t) = x(t) − x̄(d(t)).

    Args:
        x: Samples, along ``axis``.
        climatology: x̄, with its slots on the same axis.
        slot: The slot of each sample, as for `climatology`, concrete integers.
        axis: The axis of the samples.

    Returns:
        x′, the shape of x, NaN where the slot of a sample is empty.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: If a slot is not one of the climatology's, or ``axis``
            does not hold one sample per slot.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.climate import climatology
        >>> month = np.asarray([1, 2, 1, 2])
        >>> x = np.asarray([1.0, 5.0, 3.0, 9.0])
        >>> anomaly(x, climatology(x, month, n_slots=12), month)
        array([-1., -2.,  1.,  2.])
    """
    _, x, background = _at_samples(x, climatology, slot, axis)
    # x′(t) = x(t) − x̄(d(t))
    return x - background


def add_climatology(
    anomaly: ArrayLike, climatology: Climatology, slot: ArrayLike, *, axis: int = -1
) -> Any:
    """The field whose anomaly is given: the inverse of `anomaly`.

    x(t) = x′(t) + x̄(d(t)).

    Args:
        anomaly: x′, along ``axis``.
        climatology: x̄, with its slots on the same axis.
        slot: The slot of each sample, as for `climatology`, concrete integers.
        axis: The axis of the samples.

    Returns:
        x, the shape of x′, NaN where the slot of a sample is empty.

    Raises:
        BackendError: If ``anomaly`` is a DataArray or of another library.
        ValidationError: As `anomaly`.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.climate import climatology
        >>> month = np.asarray([1, 2, 1, 2])
        >>> cycle = climatology([1.0, 5.0, 3.0, 9.0], month, n_slots=12)
        >>> add_climatology([0.5, 0.0], cycle, [2, 1])
        array([7.5, 2. ])
    """
    _, x, background = _at_samples(anomaly, climatology, slot, axis)
    # x(t) = x′(t) + x̄(d(t))
    return x + background


def _at_samples(
    x: ArrayLike, climatology: Climatology, slot: ArrayLike, axis: int
) -> tuple[ModuleType, Any, Any]:
    """The namespace, the samples, and the climatology at each of them."""
    slots = checked_slots(slot, climatology.n_slots)
    xp, (samples, mean) = floating_inputs(x, climatology.mean)
    check_slots_axis(mean, climatology.n_slots, axis)
    inside = -samples.ndim <= axis < samples.ndim
    if not inside or samples.shape[axis] != slots.size:
        msg = f"axis {axis} of shape {samples.shape} must hold the {slots.size} samples"
        raise ValidationError(msg)
    lookup = table_like(slots - 1, mean, xp)
    return xp, samples, xp.take(mean, lookup, axis=axis)
