"""Linear trends, and running means."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.arrays import (
    divide_or_nan,
    reduction_inputs,
    register_record,
    used_weights,
)

from diagnostix._src.climate.window import check_window, window_sum
from diagnostix._src.reduce import require_axis
from diagnostix._src.times import time_offsets
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["Trend", "detrend", "linear_trend", "running_mean"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Trend:
    """x ≈ intercept + slope · (t − centre), fitted along one axis.

    Attributes:
        slope: β (units of x per unit of time).
        intercept: The weighted mean of x, which the line takes at ``centre``.
        centre: t̄, the weighted mean time.
    """

    slope: Float[Any, "..."]
    intercept: Float[Any, "..."]
    centre: Float[Any, "..."]


def linear_trend(
    x: ArrayLike,
    time: ArrayLike,
    *,
    axis: int = -1,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> Trend:
    """The straight line through samples by weighted least squares.

    β = ∑ₜ wₜ (t − t̄) xₜ / ∑ₜ wₜ (t − t̄)², with t̄ the weighted mean time and the
    weighted mean of x as the intercept. Where the gaps differ between points,
    so does t̄. No significance is given: that of a correlated series is a
    modelling choice.

    Args:
        x: Samples, along ``axis``.
        time: t, the time of each sample, 1-D, in any unit. Times on the host
            are centred in float64, so days since 1950 do for float32 samples.
        axis: The axis of the samples.
        weights: w ≥ 0, broadcast against ``x``; 1 if not given.
        valid: Which samples to use (True = use), broadcast against ``x``. A NaN
            in a valid sample gives NaN.

    Returns:
        The slope, the intercept and the centre, x without ``axis``. The slope
        is NaN where fewer than two different times are used, and all three
        where none is.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``time`` is not one time per sample.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> t = np.asarray([0.0, 1.0, 3.0, 4.0])
        >>> out = linear_trend(10.0 + 0.5 * t, t)
        >>> [float(part) for part in (out.slope, out.intercept, out.centre)]
        [0.5, 11.0, 2.0]
    """
    _, _, _, (slope, intercept, centre, origin) = _fit(x, time, axis, weights, valid)
    return Trend(
        slope=slope[..., 0],
        intercept=intercept[..., 0],
        centre=(centre + origin)[..., 0],
    )


def detrend(
    x: ArrayLike,
    time: ArrayLike,
    *,
    axis: int = -1,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> Any:
    """Samples without their mean and their linear trend.

    x − x̄ − β (t − t̄), with the `linear_trend`, computed around t̄ itself, so it
    loses nothing to a distant origin of time.

    Args:
        x: Samples, along ``axis``.
        time: t, as in `linear_trend`.
        axis: The axis of the samples.
        weights: w ≥ 0, as in `linear_trend`.
        valid: Which samples to fit, as in `linear_trend`; the line is taken
            from the others too.

    Returns:
        The residual, the shape of x, NaN along the axis where the trend is.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: As `linear_trend`.

    Examples:
        >>> import numpy as np
        >>> t = np.asarray([0.0, 1.0, 3.0, 4.0])
        >>> np.round(detrend(10.0 + 0.5 * t, t), 12) + 0.0
        array([0., 0., 0., 0.])
    """
    xp, samples, t, (slope, intercept, centre, _) = _fit(x, time, axis, weights, valid)
    # x − x̄ − β (t − t̄)
    return xp.moveaxis(samples - intercept - slope * (t - centre), -1, axis)


def running_mean(
    x: ArrayLike,
    window: int,
    *,
    axis: int = -1,
    valid: ArrayLike | None = None,
    periodic: bool = False,
) -> Any:
    """The mean of the valid samples in a window centred on each one.

    x̄(t) = ∑_{|s| ≤ h} m(t+s) x(t+s) / ∑_{|s| ≤ h} m(t+s), with h the half of the
    window and m the validity, 1 or 0.

    Args:
        x: Samples, evenly spaced along ``axis``.
        window: The length of the window in samples, odd, so that it is centred.
        axis: The axis of the samples.
        valid: Which samples to use (True = use), broadcast against ``x``. A NaN
            in a valid sample makes the windows that hold it NaN.
        periodic: Whether the samples are a whole period and the window wraps.
            If not, it is truncated at the ends.

    Returns:
        x̄, the shape of x, NaN where a window holds no valid sample.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``window`` is not odd and positive, or exceeds the
            period, or there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> running_mean([1.0, 2.0, 6.0, 4.0], 3)
        array([1.5, 3. , 4. , 5. ])
        >>> running_mean([1.0, 2.0, 6.0, 4.0], 3, periodic=True)
        array([2.33333333, 3.        , 4.        , 3.66666667])
        >>> running_mean([1.0, np.nan, 6.0], 3, valid=[True, False, True])
        array([1. , 3.5, 6. ])
    """
    xp, samples, _, use = reduction_inputs(x, None, valid)
    require_axis(samples, axis)
    check_window(window, samples.shape[axis], periodic=periodic)
    samples = xp.moveaxis(samples, axis, -1)
    if use is not None:
        use = xp.moveaxis(use, axis, -1)
        samples = xp.where(use, samples, xp.zeros_like(samples))
    weight = used_weights(xp, samples, None, use)
    mean = divide_or_nan(
        window_sum(xp, samples, window, periodic=periodic),
        window_sum(xp, weight, window, periodic=periodic),
        xp,
    )
    return xp.moveaxis(mean, -1, axis)


def _fit(
    x: ArrayLike,
    time: ArrayLike,
    axis: int,
    weights: ArrayLike | None,
    valid: ArrayLike | None,
) -> tuple[ModuleType, Any, Any, tuple[Any, Any, Any, Any]]:
    """The namespace, x with its samples last, t − t₀, and the line.

    Returns:
        The line as β, x̄ and t̄ − t₀, with a last axis of 1, and t₀.
    """
    traced = None if isinstance(time, (list, tuple, np.ndarray)) else time
    xp, samples, w, use = reduction_inputs(x, weights, valid, params=(traced,))
    require_axis(samples, axis)
    shape = tuple(np.shape(time))
    if shape != (samples.shape[axis],):
        msg = (
            f"'time' must be one time per sample of axis {axis} of shape "
            f"{tuple(samples.shape)}, got shape {shape}"
        )
        raise ValidationError(msg)
    t, origin = time_offsets(xp, time, samples)
    samples = xp.moveaxis(samples, axis, -1)
    w, use = (None if a is None else xp.moveaxis(a, axis, -1) for a in (w, use))

    def total(values: Any) -> Any:
        return xp.sum(values, axis=-1, keepdims=True, dtype=samples.dtype)

    # m w, and x where it counts
    used = used_weights(xp, samples, w, use)
    kept = samples if use is None else xp.where(use, samples, xp.zeros_like(samples))
    weight = total(used)
    # t̄ = ∑ w t / ∑ w, x̄ = ∑ w x / ∑ w
    centre = divide_or_nan(total(used * t), weight, xp)
    intercept = divide_or_nan(total(used * kept), weight, xp)
    defined = _has_two_times(xp, used, t)
    # an undefined line is fitted about 0, so that its gradient is finite
    about = xp.where(defined, centre, xp.zeros_like(centre))
    offset = t - about
    # β = ∑ w (t − t̄) x / ∑ w (t − t̄)²
    moment = total(used * offset * kept)
    spread = xp.where(defined, total(used * offset**2), xp.zeros_like(centre))
    slope = divide_or_nan(moment, spread, xp)
    return xp, samples, t, (slope, intercept, centre, origin)


def _has_two_times(xp: ModuleType, used: Any, t: Any) -> Any:
    """Whether two different times have a weight, with a last axis of 1.

    The spread of one time about its mean is the rounding of the mean, not 0.
    """
    counted = used > 0
    t = xp.broadcast_to(t, used.shape)
    infinite = xp.full_like(t, xp.inf)
    latest = xp.max(xp.where(counted, t, -infinite), axis=-1, keepdims=True)
    earliest = xp.min(xp.where(counted, t, infinite), axis=-1, keepdims=True)
    return latest > earliest
