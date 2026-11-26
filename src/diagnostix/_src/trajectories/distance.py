"""Distances between positions, on the plane and along great circles."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

import numpy as np
from geophysix.earth import Ellipsoid, great_circle_distance
from pipekit_array.arrays import concrete, floating_inputs

from diagnostix._src.skill.moments import root
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "Distance",
    "check_scalar",
    "distance_between",
    "known",
    "position_inputs",
    "squared_distance",
]

type Distance = Literal["euclidean", "sphere"]
"""How far apart two positions are: in their own frame, or along a great circle."""

_KINDS = ("euclidean", "sphere")


def position_inputs(
    kind: str, earth: Ellipsoid, *positions: ArrayLike, names: tuple[str, ...]
) -> tuple[ModuleType, list[Any]]:
    """The namespace and the positions as floating arrays, checked to fit a distance.

    A latitude beyond a pole is refused when it is concrete: it is usually the two
    columns in the other order.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If the kind is unknown, a set of positions has no columns,
            or, on the sphere, ``earth`` has a flattening or the positions are not
            longitudes and latitudes in degrees.
    """
    if kind not in _KINDS:
        raise ValidationError(f"'distance' must be one of {list(_KINDS)}, got {kind!r}")
    xp, arrays = floating_inputs(*positions)
    for name, array in zip(names, arrays, strict=True):
        if array.ndim == 0:
            raise ValidationError(f"'{name}' must have a column axis, got a scalar")
        if kind == "euclidean":
            continue
        if array.shape[-1] != 2:
            raise ValidationError(
                f"'{name}' must have 2 columns on the sphere, longitude and latitude "
                f"in degrees, got shape {tuple(array.shape)}"
            )
        values = concrete(array)
        if values is not None and (np.abs(values[..., 1]) > 90.0).any():
            raise ValidationError(
                f"'{name}' must hold latitudes between −90° and 90° in its second "
                f"column, got {np.nanmax(np.abs(values[..., 1]))} in size"
            )
    if kind == "sphere":
        flattening = concrete(earth.flattening)
        if flattening is not None and float(flattening) != 0.0:
            raise ValidationError(
                f"'earth' must be a sphere, got flattening {float(flattening)}"
            )
    return xp, arrays


def check_scalar(name: str, value: Any, *, zero: bool) -> None:
    """Raise unless a concrete number is finite, and above zero or on it.

    Raises:
        ValidationError: If it is not a scalar, or a concrete one is out of range.
    """
    if np.ndim(value) != 0:
        raise ValidationError(f"'{name}' must be a scalar, got shape {np.shape(value)}")
    number = concrete(value)
    if number is None:
        return
    words = "a finite length of at least 0" if zero else "finite and positive"
    above = number >= 0.0 if zero else number > 0.0
    if not (above and np.isfinite(number)):
        raise ValidationError(f"'{name}' must be {words}, got {float(number)}")


def known(xp: ModuleType, position: Any) -> Any:
    """Whether every column of each position is finite; NaN is a missing one."""
    return xp.all(xp.isfinite(position), axis=-1)


def squared_distance(
    xp: ModuleType, left: Any, right: Any, kind: Distance, earth: Ellipsoid
) -> Any:
    """d² between two sets of finite positions [m²]."""
    if kind == "euclidean":
        return xp.sum((left - right) ** 2, axis=-1)
    return _great_circle(left, right, earth) ** 2


def distance_between(
    xp: ModuleType, left: Any, right: Any, kind: Distance, earth: Ellipsoid
) -> Any:
    """d between two sets of finite positions [m], with gradient 0 where d = 0."""
    if kind == "sphere":
        return _great_circle(left, right, earth)
    # √, whose slope is infinite at 0; NaN is not 0 and stays
    return root(xp, squared_distance(xp, left, right, kind, earth))


def _great_circle(left: Any, right: Any, earth: Ellipsoid) -> Any:
    return great_circle_distance(
        left[..., 0], left[..., 1], right[..., 0], right[..., 1], earth=earth
    )
