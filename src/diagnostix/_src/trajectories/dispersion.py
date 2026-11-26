"""Dispersion: how far particles get from their start, and from each other."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from geophysix.earth import SPHERE, Ellipsoid
from pipekit_array.arrays import concrete, weighted_mean

from diagnostix._src.trajectories.distance import (
    Distance,
    known,
    position_inputs,
    squared_distance,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "absolute_dispersion",
    "pair_dispersion",
]


def _mean_squared(
    xp: ModuleType,
    left: Any,
    right: Any,
    counted: Any,
    kind: Distance,
    earth: Ellipsoid,
) -> Any:
    """The mean over the counted members of d², NaN where there is none."""
    # before any product: 0 times the derivative of a NaN is NaN
    filled = counted[..., None]
    left = xp.where(filled, left, xp.zeros_like(left))
    right = xp.where(filled, right, xp.zeros_like(right))
    squared = squared_distance(xp, left, right, kind, earth)
    return weighted_mean(squared, valid=counted, axis=-1)


def _checked(trajectory: ArrayLike, kind: str, earth: Ellipsoid) -> tuple[Any, Any]:
    """The namespace and the trajectory, checked to have times, particles and columns.

    Raises:
        ValidationError: If it does not, or does not fit the distance.
    """
    xp, (path,) = position_inputs(kind, earth, trajectory, names=("trajectory",))
    if path.ndim != 3 or path.shape[0] < 1:
        raise ValidationError(
            f"'trajectory' must be (time, particle, column), got {tuple(path.shape)}"
        )
    return xp, path


def absolute_dispersion(
    trajectory: ArrayLike,
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Any:
    """The mean squared distance of particles from where they started.

    A²(t) = (1/N_t) ∑_p d(x_p(t), x_p(0))², over the N_t particles whose position
    is known at t and at the start.

    Args:
        trajectory: The positions at each time, of shape (time, particle, column),
            the starts first, as the ``trajectory`` of `tracerax.advect.integrate`
            [m], or (longitude, latitude) in degrees. A position that is NaN is
            missing: give a particle NaN before its release, or it counts as one
            that stays.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances,
            which are right across the date line.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        A² at each time [m²], NaN where no particle is known.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If the trajectory has not three axes, or the kind of
            distance is unknown, or does not fit it.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.trajectories import absolute_dispersion
        >>> start = np.array([[0.0, 0.0], [5.0, 5.0]])
        >>> trajectory = np.stack([start, start + np.array([3.0, 4.0])])
        >>> absolute_dispersion(trajectory).tolist()
        [0.0, 25.0]

        Two steps of one degree on the equator, across the date line:

        >>> path = np.array([[[179.5, 0.0]], [[-179.5, 0.0]]])
        >>> spread = absolute_dispersion(path, distance="sphere")
        >>> bool(np.isclose(np.sqrt(spread[1]), 111_195.0, rtol=1e-5))
        True
    """
    xp, path = _checked(trajectory, distance, earth)
    starts = xp.broadcast_to(path[:1, ...], path.shape)
    counted = known(xp, path) & known(xp, starts)
    return _mean_squared(xp, path, starts, counted, distance, earth)


def pair_dispersion(
    trajectory: ArrayLike,
    pairs: ArrayLike,
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Any:
    """The mean squared distance between the partners of each pair.

    D²(t) = (1/N_t) ∑_(p,q) d(x_p(t), x_q(t))², over the N_t pairs whose two
    positions are known at t.

    Args:
        trajectory: The positions at each time, as in `absolute_dispersion`.
        pairs: The two particles of each pair, integers of shape (pair, 2).
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        D² at each time [m²], NaN where no pair is known.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: As `absolute_dispersion` does, or if ``pairs`` is not
            (pair, 2) or a concrete one names no particle.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.trajectories import pair_dispersion
        >>> start = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        >>> later = np.array([[0.0, 0.0], [4.0, 4.0], [5.0, 5.0]])
        >>> pair_dispersion(np.stack([start, later]), np.array([[0, 1]])).tolist()
        [1.0, 32.0]
    """
    xp, path = _checked(trajectory, distance, earth)
    pairs = xp.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValidationError(
            f"'pairs' must be (pair, 2), got shape {tuple(pairs.shape)}"
        )
    n_particles = path.shape[1]
    members = concrete(pairs)
    if members is not None and members.size:
        if not np.issubdtype(members.dtype, np.integer):
            raise ValidationError(
                f"'pairs' must be integers, got dtype {members.dtype}"
            )
        if members.min() < 0 or members.max() >= n_particles:
            raise ValidationError(
                f"'pairs' must name particles from 0 to {n_particles - 1}, "
                f"got {members.min()} to {members.max()}"
            )
    left = xp.take(path, pairs[:, 0], axis=1)
    right = xp.take(path, pairs[:, 1], axis=1)
    counted = known(xp, left) & known(xp, right)
    return _mean_squared(xp, left, right, counted, distance, earth)
