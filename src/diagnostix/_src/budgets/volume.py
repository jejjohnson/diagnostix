"""A region of a collocated grid, and the signed faces that bound it."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from jaxtyping import Bool, Int

from diagnostix._src.budgets.faces import orientation
from diagnostix._src.grids import require_collocated
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["ControlVolume"]


@dataclasses.dataclass(frozen=True, eq=False)
class ControlVolume:
    """A region of a collocated grid and the signed faces that bound it.

    The tables are NumPy arrays and static: under JAX a `ControlVolume` is a
    closure or a static argument, and never a traced one. It is compared by
    identity, so build it once; a new one compiles again.

    Face (j, i) of ``east_sign`` lies between the cells (j, i) and (j, i + 1),
    and of ``north_sign`` between (j, i) and (j + 1, i); on a periodic axis the
    last face closes the circle.

    Attributes:
        inside: Which cells are in the region.
        east_sign: +1 where the region is left through the face going towards
            larger x, −1 where it is entered, 0 off the boundary.
        north_sign: The same towards larger y, whichever way y is stored.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_interior(4, 4, 4.0, 4.0)
        >>> inside = np.zeros((4, 4), dtype=bool)
        >>> inside[1:3, 1:3] = True
        >>> box = ControlVolume.from_region(inside, grid)
        >>> box.east_sign[1]
        array([-1,  0,  1], dtype=int8)
        >>> int(np.abs(box.east_sign).sum() + np.abs(box.north_sign).sum())
        8
    """

    inside: Bool[np.ndarray, "y x"]
    east_sign: Int[np.ndarray, "y fx"]
    north_sign: Int[np.ndarray, "fy x"]

    @classmethod
    def from_region(cls, inside: ArrayLike, grid: Any) -> ControlVolume:
        """The control volume of the cells of a region.

        Args:
            inside: Which cells are in the region (True = in), concrete, in the
                grid's (y, x). It may have holes and several parts.
            grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`,
                concrete.

        Returns:
            The control volume.

        Raises:
            ValidationError: If the grid is not collocated and 2-D, if
                ``inside`` is not boolean or not of its shape, if it has no cell
                or every cell, or if it touches an edge that does not wrap: a
                face of it would lie outside the data.
        """
        require_collocated(grid)
        # a copy: the caller may write to its array later
        region = np.array(inside)
        if region.dtype != np.bool_ or region.shape != grid.shape:
            msg = (
                f"'inside' must be boolean of shape {grid.shape}, got "
                f"{region.dtype} of shape {region.shape}"
            )
            raise ValidationError(msg)
        if not region.any() or region.all():
            msg = "'inside' must have some of the grid's cells, and not all of them"
            raise ValidationError(msg)
        for axis, name in ((0, "y"), (1, "x")):
            edges = np.take(region, [0, -1], axis=axis)
            if name not in grid.periodic and edges.any():
                msg = (
                    f"'inside' touches an edge of the grid along {name}, where a "
                    "face of it would lie outside the data"
                )
                raise ValidationError(msg)
        # +1 where the cell before the face is in and the one after it is out
        signs = [
            int(orientation(grid, name)) * _steps(region, axis, name in grid.periodic)
            for axis, name in ((1, "x"), (0, "y"))
        ]
        for table in (region, *signs):
            table.setflags(write=False)
        return cls(inside=region, east_sign=signs[0], north_sign=signs[1])


def _steps(region: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    """The region's indicator before each face minus the one after it."""
    indicator = region.astype(np.int8)
    if periodic:
        return indicator - np.roll(indicator, -1, axis=axis)
    return -np.diff(indicator, axis=axis)
