"""Integrals over a column."""

from __future__ import annotations

from typing import Any

from diagnostix._src.columns.levels import column_inputs
from diagnostix._typing import ArrayLike


__all__ = ["column_integral"]


def column_integral(values: ArrayLike, z: ArrayLike, *, axis: int = -1) -> Any:
    """The integral over a column, by the trapezoid rule.

    ∫ f dz ≈ ∑ₖ ½ (f_k + f_{k+1}) (z_{k+1} − z_k).

    It equals ∑ₖ f_k Δz_k, with cells that end half-way to each neighbour and at
    the first and last levels. Exact for a field linear in z.

    Args:
        values: f, with its levels along ``axis``.
        z: The position of each level, increasing or decreasing, 1-D or one
            column per point. With a pressure p, divide by g:
            ∫ q dp / g is ``column_integral(q, p) / g``.
        axis: The axis of the levels.

    Returns:
        The integral (units of f times those of z), ``values`` without ``axis``.
        Its sign follows the order of z: it is negative over levels that
        decrease. A NaN at any level makes its column NaN.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or ``z`` does not
            have those of ``values``.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([0.0, 1.0, 3.0])
        >>> float(column_integral(2.0 * z, z))
        9.0
        >>> float(column_integral(2.0 * z[::-1], z[::-1]))
        -9.0
    """
    xp, (f,), columns, _ = column_inputs((values,), z, axis=axis)
    spacing = columns[..., 1:] - columns[..., :-1]
    # ½ (f_k + f_{k+1}) (z_{k+1} − z_k)
    area = 0.5 * (f[..., 1:] + f[..., :-1]) * spacing
    return xp.sum(area, axis=-1, dtype=area.dtype)
