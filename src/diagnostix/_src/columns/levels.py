"""The arguments of a function of columns: levels last, coordinates broadcast."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

from pipekit_array.functions import floating_inputs

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["column_inputs"]


def column_inputs(
    fields: Sequence[ArrayLike],
    coordinate: ArrayLike,
    *,
    axis: int,
    per_column: Sequence[ArrayLike] = (),
    params: tuple[Any, ...] = (),
) -> tuple[ModuleType, list[Any], Any, list[Any]]:
    """The namespace of a call on columns, and its inputs with the levels last.

    A 1-D field or coordinate is one column, shared by every point, whatever
    ``axis`` is; any other has its levels along ``axis``.

    Args:
        fields: The fields on levels.
        coordinate: The position of each level, 1-D or one column per point.
        axis: The axis of the levels.
        per_column: Values with one number per column, or one for all: the
            shape of a field without ``axis``.
        params: Python floats or 0-d arrays, as in `floating_inputs`.

    Returns:
        The namespace, the fields and the coordinate broadcast against each
        other with the levels last, and ``per_column`` with a last axis of 1.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If an input has no level axis, the coordinate does not
            have the fields' levels, or there are fewer than two.
    """
    xp, arrays = floating_inputs(*fields, coordinate, *per_column, params=params)
    n_fields = len(fields)
    on_levels = [_levels_last(xp, array, axis) for array in arrays[: n_fields + 1]]
    counts = {array.shape[-1] for array in on_levels}
    if len(counts) != 1:
        msg = f"the fields and the coordinate differ in levels: {sorted(counts)}"
        raise ValidationError(msg)
    n_levels = counts.pop()
    if n_levels < 2:
        msg = f"a column needs at least two levels, got {n_levels}"
        raise ValidationError(msg)
    *columns, z = xp.broadcast_arrays(*on_levels)
    single = [xp.expand_dims(array, axis=-1) for array in arrays[n_fields + 1 :]]
    return xp, columns, z, single


def _levels_last(xp: ModuleType, array: Any, axis: int) -> Any:
    """``array`` with its levels on the last axis."""
    if array.ndim == 1:
        return array
    if not -array.ndim <= axis < array.ndim:
        msg = f"no axis {axis} for the levels of an array of shape {array.shape}"
        raise ValidationError(msg)
    return xp.moveaxis(array, axis, -1)
