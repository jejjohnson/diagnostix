"""The arguments of a function of a horizontal velocity on a collocated grid."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gridax
from pipekit_array.functions import floating_inputs

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["grid_inputs"]

#: The grids of `diagnostix.kinematics`: every field at the same points.
COLLOCATED = (gridax.CartesianAGrid2D, gridax.SphericalAGrid2D)


def grid_inputs(
    grid: Any,
    *fields: ArrayLike,
    broadcast: tuple[ArrayLike, ...] = (),
    params: tuple[Any, ...] = (),
) -> tuple[ModuleType, list[Any]]:
    """The namespace of a call on a grid, and its fields in one floating dtype.

    Args:
        grid: The grid.
        *fields: Fields that end in the grid's (y, x).
        broadcast: Values that need only broadcast against a field, as a Coriolis
            parameter of each row.
        params: Python floats or 0-d arrays, as in `floating_inputs`.

    Returns:
        The namespace, and ``fields`` then ``broadcast`` as arrays.

    Raises:
        BackendError: If a value is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.
    """
    if not isinstance(grid, COLLOCATED):
        names = " or ".join(kind.__name__ for kind in COLLOCATED)
        msg = f"'grid' must be a {names}, got {type(grid).__name__}"
        raise ValidationError(msg)
    xp, arrays = floating_inputs(*fields, *broadcast, params=params)
    for field in arrays[: len(fields)]:
        if tuple(field.shape[-2:]) != grid.shape:
            msg = (
                f"a field on this grid ends in {grid.shape} (y, x), "
                f"got shape {tuple(field.shape)}"
            )
            raise ValidationError(msg)
    return xp, arrays
