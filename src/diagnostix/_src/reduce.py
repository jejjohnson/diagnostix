"""The inputs of a masked, weighted reduction, and host tables in a field's dtype.

The reductions themselves, and the groups, are pipekit's
(``pipekit_array.weighted``). These are for a reduction that is not one of them.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from pipekit_array.functions import boolean_like, floating_inputs, table_like

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["constants_like", "reduction_inputs", "require_axis", "used_weights"]


def reduction_inputs(
    values: ArrayLike,
    weights: ArrayLike | None,
    valid: ArrayLike | None,
    *,
    params: tuple[Any, ...] = (),
) -> tuple[ModuleType, Any, Any, Any]:
    """The namespace of a reduction, and its inputs broadcast against each other.

    Args:
        values: x.
        weights: w, or ``None``.
        valid: The validity, boolean, or ``None``.
        params: Other arguments, which help to choose the namespace only.

    Returns:
        The namespace, x and w in one floating dtype, and the validity as a
        boolean array; w and the validity stay ``None`` where they are.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValueError: If ``valid`` is not boolean.
    """
    if weights is None:
        xp, (x,) = floating_inputs(values, params=(valid, *params))
        w = None
    else:
        xp, (x, w) = floating_inputs(values, weights, params=(valid, *params))
    m = None if valid is None else boolean_like(valid, x, xp)
    given = [array for array in (x, w, m) if array is not None]
    broadcast = iter(xp.broadcast_arrays(*given))
    return (
        xp,
        next(broadcast),
        None if w is None else next(broadcast),
        None if m is None else next(broadcast),
    )


def used_weights(xp: ModuleType, x: Any, w: Any, m: Any) -> Any:
    """m w, the weight each sample enters with, in ``x``'s shape and dtype."""
    weight = xp.ones_like(x) if w is None else w
    return weight if m is None else xp.where(m, weight, xp.zeros_like(weight))


def constants_like(xp: ModuleType, host: np.ndarray, ref: Any) -> Any:
    """A host table of numbers in ``ref``'s library and dtype, on its device."""
    return xp.astype(table_like(host, ref, xp), ref.dtype)


def require_axis(values: Any, axis: int) -> None:
    """Raise unless an array has an axis.

    Raises:
        ValidationError: If it does not.
    """
    if not -values.ndim <= axis < values.ndim:
        msg = f"no axis {axis} for the samples of an array of shape {values.shape}"
        raise ValidationError(msg)
