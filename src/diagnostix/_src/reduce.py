"""Masked, weighted sums and means, on any array library.

The array API standard has no ``nansum``, and a NaN is not a mask here: which
samples count is said by ``valid``, boolean, True = use.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat
import numpy as np
from pipekit_array.functions import floating_inputs

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "boolean_like",
    "divide_or_nan",
    "on_device",
    "reduction_inputs",
    "weighted_mean",
    "weighted_sum",
]

type Axis = int | tuple[int, ...] | None


def weighted_sum(
    values: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The sum of w x over the valid samples.

    S = ∑ₙ mₙ wₙ xₙ, with m the validity, 1 or 0.

    Args:
        values: x, samples of any shape.
        weights: w, broadcast against ``values``; 1 if not given.
        valid: Which samples to use (True = use), broadcast against ``values``.
            A NaN in a valid sample gives NaN, and one in an invalid sample
            reaches neither the sum nor its gradient.
        axis: The axis or axes to sum over; every axis if not given.

    Returns:
        S, the array without the reduced axes. An empty sum is 0.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([1.0, np.nan, 3.0])
        >>> float(weighted_sum(x, valid=[True, False, True]))
        4.0
        >>> float(weighted_sum(x[[0, 2]], weights=[0.5, 2.0]))
        6.5
    """
    xp, x, w, m = reduction_inputs(values, weights, valid)
    # S = ∑ₙ mₙ wₙ xₙ
    return xp.sum(_kept(xp, x, w, m), axis=axis, dtype=x.dtype)


def weighted_mean(
    values: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The mean of x over the valid samples, weighted by w.

    x̄ = ∑ₙ mₙ wₙ xₙ / ∑ₙ mₙ wₙ, with m the validity, 1 or 0.

    Args:
        values: x, samples of any shape.
        weights: w ≥ 0, broadcast against ``values``; they need not sum to 1.
            Cell areas come from ``gridax.area_weights``.
        valid: Which samples to use (True = use), broadcast against ``values``.
            A NaN in a valid sample gives NaN, and one in an invalid sample
            reaches neither the mean nor its gradient.
        axis: The axis or axes to average over; every axis if not given.

    Returns:
        x̄, the array without the reduced axes, NaN where the weights used sum
        to 0.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 4.0], [2.0, np.nan]])
        >>> weighted_mean(x, weights=[3.0, 1.0], valid=~np.isnan(x), axis=-1)
        array([1.75, 2.  ])
        >>> float(weighted_mean(x[0], weights=[0.0, 0.0]))
        nan
    """
    xp, x, w, m = reduction_inputs(values, weights, valid)
    # S = ∑ₙ mₙ wₙ xₙ
    total = xp.sum(_kept(xp, x, w, m), axis=axis, dtype=x.dtype)
    # W = ∑ₙ mₙ wₙ
    weight = xp.sum(used_weights(xp, x, w, m), axis=axis, dtype=x.dtype)
    return divide_or_nan(xp, total, weight)


def reduction_inputs(
    values: ArrayLike, weights: ArrayLike | None, valid: ArrayLike | None
) -> tuple[ModuleType, Any, Any, Any]:
    """The namespace of a reduction, and its inputs broadcast against each other.

    Args:
        values: x.
        weights: w, or ``None``.
        valid: The validity, boolean, or ``None``.

    Returns:
        The namespace, x and w in one floating dtype, and the validity as a
        boolean array; w and the validity stay ``None`` where they are.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``valid`` is not boolean.
    """
    if weights is None:
        xp, (x,) = floating_inputs(values, params=(valid,))
        w = None
    else:
        xp, (x, w) = floating_inputs(values, weights, params=(valid,))
    m = None if valid is None else boolean_like(xp, valid, x)
    given = [array for array in (x, w, m) if array is not None]
    broadcast = iter(xp.broadcast_arrays(*given))
    return (
        xp,
        next(broadcast),
        None if w is None else next(broadcast),
        None if m is None else next(broadcast),
    )


def boolean_like(xp: ModuleType, valid: ArrayLike, ref: Any) -> Any:
    """``valid`` as a boolean array of ``ref``'s library, on its device.

    Raises:
        ValidationError: If ``valid`` is not boolean.
    """
    on_host = isinstance(valid, (bool, list, tuple, np.ndarray, np.generic))
    if on_host:
        valid = np.asarray(valid)
    # a NumPy dtype is one of every namespace that NumPy arrays join
    namespace = np if on_host else xp
    if not namespace.isdtype(valid.dtype, "bool"):
        msg = f"'valid' must be boolean (True = use), got dtype {valid.dtype}"
        raise ValidationError(msg)
    return on_device(xp, valid, ref) if on_host else valid


def on_device(xp: ModuleType, host: np.ndarray, ref: Any) -> Any:
    """A host table as an array of ``ref``'s library, on its device."""
    # a JAX tracer has no device, and JAX places its own constants
    if array_api_compat.is_jax_namespace(xp):
        return xp.asarray(host)
    return xp.asarray(host, device=array_api_compat.device(ref))


def _kept(xp: ModuleType, x: Any, w: Any, m: Any) -> Any:
    """m w x, where an invalid sample is replaced before the product."""
    # 0 in place of xₙ where mₙ = 0: NaN × 0 is NaN
    kept = x if m is None else xp.where(m, x, xp.zeros_like(x))
    return kept if w is None else kept * used_weights(xp, x, w, m)


def used_weights(xp: ModuleType, x: Any, w: Any, m: Any) -> Any:
    """m w, the weight each sample enters with, in ``x``'s shape and dtype."""
    weight = xp.ones_like(x) if w is None else w
    return weight if m is None else xp.where(m, weight, xp.zeros_like(weight))


def divide_or_nan(xp: ModuleType, total: Any, weight: Any) -> Any:
    """``total / weight``, NaN where the weight is 0, through a double ``where``."""
    defined = weight > 0
    # the unused branch divides by 1, so its gradient is finite too
    safe = xp.where(defined, weight, xp.ones_like(weight))
    return xp.where(defined, total / safe, xp.full_like(total, xp.nan))
