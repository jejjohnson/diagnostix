"""The separation of a trajectory from an observed one, and its skill."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from geophysix.earth import SPHERE, Ellipsoid
from pipekit_array.arrays import like, namespace_of

from diagnostix._src.trajectories.distance import (
    Distance,
    check_scalar,
    distance_between,
    known,
    position_inputs,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "cumulative_separation",
    "separation",
    "trajectory_skill",
]


def _paired(
    trajectory: ArrayLike, reference: ArrayLike, kind: str, earth: Ellipsoid
) -> tuple[ModuleType, Any, Any, Any]:
    """The two, broadcast and with something finite at each missing observation.

    Raises:
        ValidationError: If they have other times or columns, or do not fit the
            distance.
    """
    xp, (x, y) = position_inputs(
        kind, earth, trajectory, reference, names=("trajectory", "reference")
    )
    if x.ndim < 2 or y.ndim < 2 or x.shape[-2:] != y.shape[-2:]:
        raise ValidationError(
            "'trajectory' and 'reference' must have the same times and columns, "
            f"got shapes {tuple(x.shape)} and {tuple(y.shape)}"
        )
    x, y = xp.broadcast_arrays(x, y)
    observed = known(xp, y)
    # before any product: 0 times the derivative of a NaN is NaN
    filled = observed[..., None]
    x = xp.where(filled, x, xp.zeros_like(x))
    y = xp.where(filled, y, xp.zeros_like(y))
    return xp, x, y, observed


def separation(
    trajectory: ArrayLike,
    reference: ArrayLike,
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Any:
    """The distance between a trajectory and an observed one, time by time.

    d_k = d(x_k, y_k)

    Args:
        trajectory: x, simulated, of shape (…, time, column) [m], or (longitude,
            latitude) in degrees. Time comes before the columns: for the
            ``trajectory`` of `tracerax.advect.integrate`, which is (time,
            particle, column), move its particle axis first.
        reference: y, observed at the same times, broadcast against ``trajectory``;
            NaN where there is no observation.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        d, of shape (…, time) [m], NaN at a missing observation and where the
        trajectory is not finite. Its gradient is 0 where d = 0.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If the two have other times or columns, or the kind of
            distance is unknown, or does not fit them.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.trajectories import separation
        >>> observed = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan]])
        >>> simulated = np.array([[0.0, 0.0], [13.0, 4.0], [20.0, 0.0]])
        >>> separation(simulated, observed).tolist()
        [0.0, 5.0, nan]
    """
    xp, x, y, observed = _paired(trajectory, reference, distance, earth)
    apart = distance_between(xp, x, y, distance, earth)
    return xp.where(observed, apart, xp.full_like(apart, xp.nan))


def _length_travelled(
    xp: ModuleType, reference: Any, observed: Any, kind: Distance, earth: Ellipsoid
) -> Any:
    """l_k, the length of the observed path up to each time.

    A gap is bridged in a straight line, or along a great circle.
    """
    # c_k, the number of observations up to k, and the times of the observations
    # first, in order
    count = xp.cumulative_sum(xp.astype(observed, xp.int64), axis=-1)
    order = xp.argsort(xp.astype(~observed, xp.int32), axis=-1, stable=True)
    # the observation before k is the (c_k − 1)-th, counted from 1
    earlier = xp.clip(count - 2, min=0)
    before = xp.take_along_axis(order, earlier, axis=-1)
    gather = xp.broadcast_to(before[..., None], reference.shape)
    previous = xp.take_along_axis(reference, gather, axis=-2)
    piece = distance_between(xp, previous, reference, kind, earth)
    counted = observed & (count >= 2)
    piece = xp.where(counted, piece, xp.zeros_like(piece))
    return xp.cumulative_sum(piece, axis=-1)


def cumulative_separation(
    trajectory: ArrayLike,
    reference: ArrayLike,
    *,
    distance: Distance = "euclidean",
    min_length: ArrayLike = 0.0,
    earth: Ellipsoid = SPHERE,
) -> Any:
    """The normalised cumulative separation of a trajectory from an observed one.

        s = ∑ₖ d_k / ∑ₖ l_k

    with d_k the separation at time k and l_k the length of the observed path up
    to it, both summed over the times with an observation. It is 0 for a perfect
    trajectory, and about 1 for one whose errors grow like the path. It is also a
    loss: its gradient is finite where the two coincide.

    dask has no ``take_along_axis``: compute a dask array first.

    Args:
        trajectory: x, as in `separation`.
        reference: y, as in `separation`.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        min_length: A length added to each l_k [m], without which an observed
            path that does not move gives s = ∞, or NaN when d = 0 too.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        s, of shape (…,), NaN without any observation. Where it is ∞ or NaN for
        want of a length, its gradient is 0.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: As `separation` does, or if ``min_length`` is negative
            or not finite.

    References:
        Liu, Y. and Weisberg, R. H. (2011). Evaluation of trajectory modeling in
        different dynamic regions using normalized cumulative Lagrangian
        separation. J. Geophys. Res. 116, C09013.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.trajectories import cumulative_separation
        >>> observed = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        >>> simulated = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        >>> float(cumulative_separation(simulated, observed))
        0.5
    """
    check_scalar("min_length", min_length, zero=True)
    xp, x, y, observed = _paired(trajectory, reference, distance, earth)
    apart = distance_between(xp, x, y, distance, earth)
    length = _length_travelled(xp, y, observed, distance, earth)
    length = length + like(min_length, length, xp)
    zero = xp.zeros_like(apart)
    # ∑ₖ d_k and ∑ₖ l_k
    total = xp.sum(xp.where(observed, apart, zero), axis=-1)
    scale = xp.sum(xp.where(observed, length, zero), axis=-1)
    # without a length, s is ∞, or NaN where there is no separation either
    travelled = scale > 0.0
    quotient = total / xp.where(travelled, scale, xp.ones_like(scale))
    infinite = xp.full_like(total, xp.inf)
    empty = xp.where(total > 0.0, infinite, xp.full_like(total, xp.nan))
    return xp.where(travelled, quotient, empty)


def trajectory_skill(
    trajectory: ArrayLike,
    reference: ArrayLike,
    *,
    tolerance: ArrayLike = 1.0,
    distance: Distance = "euclidean",
    min_length: ArrayLike = 0.0,
    earth: Ellipsoid = SPHERE,
) -> Any:
    """The skill of a trajectory: 1 for a perfect one, 0 for one beyond tolerance.

    1 − s/n where s ≤ n, and 0 elsewhere, with s the `cumulative_separation`.

    dask has no ``take_along_axis``: compute a dask array first.

    Args:
        trajectory: x, as in `separation`.
        reference: y, as in `separation`.
        tolerance: n, the separation beyond which there is no skill, usually 1.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        min_length: As in `cumulative_separation`.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        The skill, of shape (…,), NaN without any observation. Beyond the
        tolerance its gradient is 0, so fit with `cumulative_separation`.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: As `cumulative_separation` does, or if ``tolerance`` is
            not positive and finite.

    References:
        Liu, Y. and Weisberg, R. H. (2011). Evaluation of trajectory modeling in
        different dynamic regions using normalized cumulative Lagrangian
        separation. J. Geophys. Res. 116, C09013.

    Examples:
        >>> import numpy as np
        >>> from diagnostix.trajectories import trajectory_skill
        >>> observed = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        >>> simulated = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        >>> float(trajectory_skill(simulated, observed))
        0.5
        >>> float(trajectory_skill(simulated, observed, tolerance=0.4))
        0.0
    """
    check_scalar("tolerance", tolerance, zero=False)
    s = cumulative_separation(
        trajectory,
        reference,
        distance=distance,
        min_length=min_length,
        earth=earth,
    )
    xp = namespace_of(s)
    limit = like(tolerance, s, xp)
    beyond = s > limit
    zero = xp.zeros_like(s)
    within = xp.where(beyond, zero, s)
    # 1 − s/n where s ≤ n, and NaN where s is
    return xp.where(beyond, zero, 1.0 - within / limit)
