"""The faces between the cells of a collocated grid: lengths, means, differences."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gridax
from pipekit_array.functions import namespace_of

from diagnostix._src.grids import metric_like


__all__ = ["across", "face_lengths", "face_mean", "orientation"]


def orientation(grid: Any, axis: str) -> Any:
    """+1 if the coordinate of ``axis`` grows with the index, −1 if it falls."""
    coords = grid.coords(axis)
    return namespace_of(coords).sign(coords[-1] - coords[0])


def face_lengths(xp: ModuleType, grid: Any, ref: Any) -> tuple[Any, Any]:
    """The lengths of the faces across x and across y, in ``ref``'s dtype.

    On a sphere the first is R Δφ and the second R cos(φ_{j+½}) Δλ; on a plane
    they are Δy and Δx. On a periodic axis the last face closes the circle.

    Returns:
        Both broadcast against the faces: the first is a column, or one number.
    """
    host = namespace_of(grid.coords("x"), grid.coords("y"))
    # Δy of each row (R Δφ), and Δx of each column of a plane
    dy = host.reshape(host.abs(grid.spacing("y")), (-1, 1))
    if not isinstance(grid, gridax.SphericalAGrid2D):
        dx = host.abs(host.reshape(grid.spacing("x"), (1, -1)))
        return metric_like(xp, dy, ref), metric_like(xp, dx, ref)
    latitude = grid.coords("y")
    # R Δλ, from the row nearest the equator, where cos φ is largest
    row = host.argmax(host.cos(latitude))
    dx = host.abs(grid.spacing("x")[row, :]) / host.cos(latitude)[row]
    # φ_{j+½}, half-way between the rows
    between = 0.5 * (latitude[:-1] + latitude[1:])
    north = host.cos(between)[:, None] * dx[None, :]
    return metric_like(xp, dy, ref), metric_like(xp, north, ref)


def face_mean(xp: ModuleType, values: Any, *, axis: int, periodic: bool) -> Any:
    """The mean of the two cells of each face, ``axis`` one shorter or periodic."""
    if periodic:
        return 0.5 * (values + xp.roll(values, -1, axis=axis))
    before, after = _before_after(values, axis)
    return 0.5 * (before + after)


def across(xp: ModuleType, faces: Any, *, axis: int, periodic: bool) -> Any:
    """The face after each cell minus the one before it, NaN where there is none."""
    if periodic:
        return faces - xp.roll(faces, 1, axis=axis)
    before, after = _before_after(faces, axis)
    inner = after - before
    edge = xp.full_like(_first(faces, axis), xp.nan)
    return xp.concat([edge, inner, edge], axis=axis)


def _before_after(values: Any, axis: int) -> tuple[Any, Any]:
    """``values`` without its last and without its first entry along ``axis``."""
    if axis == -1:
        return values[..., :-1], values[..., 1:]
    return values[..., :-1, :], values[..., 1:, :]


def _first(values: Any, axis: int) -> Any:
    """The first entry of ``values`` along ``axis``, which is kept."""
    return values[..., :1] if axis == -1 else values[..., :1, :]
