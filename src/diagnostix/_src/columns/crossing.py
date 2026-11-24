"""Where a profile first reaches a level, between the levels it is given on."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from diagnostix._src.columns.levels import column_inputs
from diagnostix._src.reduce import on_device
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["crossing", "first_crossing"]


def first_crossing(
    values: ArrayLike,
    coordinate: ArrayLike,
    level: ArrayLike,
    *,
    axis: int = -1,
    start: int = 0,
) -> Any:
    """The first position along an axis where a profile reaches a level.

    With k the first index after ``start`` where f_k ≥ ℓ,

    z⋆ = z_{k−1} + (ℓ − f_{k−1}) / (f_k − f_{k−1}) · (z_k − z_{k−1}).

    Under autodiff the index k is piecewise constant, and the gradient flows
    through the interpolation.

    Args:
        values: f, with its levels along ``axis``.
        coordinate: z, the position of each level, 1-D or one column per point.
        level: ℓ, one number, or one per column (``values`` without ``axis``).
        axis: The axis of the levels.
        start: The index to search after. A profile that has reached the level
            there already crosses it there.

    Returns:
        z⋆, ``values`` without ``axis``, NaN where no level from ``start`` on
        reaches ℓ. A NaN between ``start`` and the crossing gives NaN. An
        infinite f_k gives z_{k−1}, the limit of the formula, and an infinite
        f_{k−1} below a finite f_k gives z_k.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, ``coordinate`` does
            not have those of ``values``, or ``start`` is not one of them or is
            the last.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([0.0, 10.0, 30.0])
        >>> float(first_crossing([0.0, 1.0, 3.0], z, 2.0))
        20.0

        One level per column, and a column that never reaches it:

        >>> f = np.asarray([[0.0, 1.0, 3.0], [0.0, 1.0, 1.5]])
        >>> first_crossing(f, z, np.asarray([0.5, 2.0]))
        array([ 5., nan])
    """
    xp, (f,), z, (threshold,) = column_inputs(
        (values,), coordinate, axis=axis, per_column=(level,)
    )
    n_levels = f.shape[-1]
    if not 0 <= start < n_levels - 1:
        msg = f"'start' must be in [0, {n_levels - 2}], got {start}"
        raise ValidationError(msg)
    found, position = crossing(xp, f, z, threshold, start=start)
    return xp.where(found, position, xp.full_like(position, xp.nan))


def crossing(
    xp: ModuleType, f: Any, z: Any, threshold: Any, *, start: int = 0
) -> tuple[Any, Any]:
    """Whether each column reaches a threshold from ``start`` on, and where.

    Args:
        xp: The namespace.
        f: The profiles, levels last.
        z: The position of each level, broadcast against ``f``.
        threshold: ℓ, broadcast against ``f`` with a last axis of 1.
        start: The index to search after.

    Returns:
        Whether the column reaches ℓ, true also where a NaN hides the answer,
        and z⋆, NaN there and meaningless where the column does not reach ℓ.
    """
    n_levels = f.shape[-1]
    from_start = on_device(xp, np.arange(n_levels) >= start, f)
    # not (f < ℓ), so a NaN stops the search instead of being passed over
    reached = from_start & ~(f < threshold)
    found = xp.any(reached, axis=-1)
    first = xp.argmax(xp.astype(reached, xp.int32), axis=-1, keepdims=True)
    # a column that is there at start, or never, reads start and the level after
    k = xp.clip(first, start + 1, n_levels - 1)
    f_lower, f_upper = _at(xp, f, k - 1), _at(xp, f, k)
    z_lower, z_upper = _at(xp, z, k - 1), _at(xp, z, k)
    target = xp.broadcast_to(threshold, f_lower.shape)

    # --- interpolate, where there is a rise to divide by ---
    rises = (f_lower < target) & (target <= f_upper)
    rises = rises & xp.isfinite(f_lower) & xp.isfinite(f_upper)
    # an unused branch reads 0 ≤ 0 < 1, so its value and gradient are finite
    lower = xp.where(rises, f_lower, xp.zeros_like(f_lower))
    upper = xp.where(rises, f_upper, xp.ones_like(f_upper))
    target = xp.where(rises, target, xp.zeros_like(target))
    # (ℓ − f_{k−1}) / (f_k − f_{k−1})
    fraction = (target - lower) / (upper - lower)
    # z⋆ = z_{k−1} + fraction · (z_k − z_{k−1})
    position = z_lower + fraction * (z_upper - z_lower)

    # --- the limits, and the NaNs ---
    from_below = xp.isinf(f_lower) & (f_lower < 0) & xp.isfinite(f_upper)
    position = xp.where(from_below, z_upper, position)
    hidden = xp.isnan(f_lower) | xp.isnan(threshold)
    hidden = hidden | (xp.isnan(f_upper) & (f_lower < threshold))
    position = xp.where(hidden, xp.full_like(position, xp.nan), position)
    return found, position[..., 0]


def _at(xp: ModuleType, columns: Any, index: Any) -> Any:
    """The entry of each column at its own index, as a last axis of 1."""
    return xp.take_along_axis(columns, index, axis=-1)
