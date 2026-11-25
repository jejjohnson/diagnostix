"""Whether one method's errors are smaller than another's by more than chance."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from jaxtyping import Float
from pipekit_array.arrays import floating_inputs, register_record

from diagnostix._src.reduce import require_axis
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["DieboldMariano", "diebold_mariano"]


@dataclasses.dataclass(frozen=True, eq=False)
class DieboldMariano:
    """The Diebold–Mariano statistic, and what it is compared with.

    Attributes:
        statistic: S, positive where the first method's errors are larger.
        degrees_of_freedom: T − 1, of the Student-t that S follows where the
            methods are equally good. Static under JAX.
    """

    statistic: Float[Any, "..."]
    degrees_of_freedom: int


register_record(DieboldMariano, meta_fields=("degrees_of_freedom",))


def diebold_mariano(
    errors_a: ArrayLike,
    errors_b: ArrayLike,
    *,
    axis: int = -1,
    horizon: int = 1,
    power: float = 2.0,
    correction: bool = True,
) -> DieboldMariano:
    """The Diebold–Mariano statistic of two methods' errors on the same cases.

    With d = |a|ᵖ − |b|ᵖ over T cases and γₖ its autocovariance at lag k,

    S = d̄ / ((γ₀ + 2 ∑ₖ γₖ) / T)^½, over the lags k = 1 … h − 1,

    times ((T + 1 − 2h + h (h − 1)/T) / T)^½, the small-sample correction of Harvey,
    Leybourne and Newbold (1997).

    A forecast of h steps is wrong for h steps in a row, so its errors are
    correlated up to lag h − 1. For h = 1, S is the one-sample t statistic of d.

    There is no p-value here: it is ``2 * scipy.stats.t.sf(abs(s.statistic),
    s.degrees_of_freedom)``.

    Args:
        errors_a: a, the errors of the first method, in order along ``axis``.
            They are raw: the power is taken here.
        errors_b: b, those of the second, broadcast against ``errors_a``.
        axis: The axis of the cases, of both once broadcast.
        horizon: h ≥ 1, in cases.
        power: p > 0, of the loss |error|ᵖ.
        correction: Whether to apply the correction.

    Returns:
        The statistic, without ``axis``, and the degrees of freedom. NaN where
        every difference is the same, or the variance of the mean is estimated
        as negative, which these autocovariances can make it. A NaN error gives
        NaN: to drop a case would change which are h apart.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis, ``horizon`` is not between 1
            and T − 1, or ``power`` is not positive.

    Examples:
        >>> import numpy as np
        >>> a = np.asarray([1.0, 2.0, 1.0, 3.0])
        >>> b = np.asarray([1.0, 1.0, 1.0, 2.0])
        >>> out = diebold_mariano(a, b)
        >>> round(float(out.statistic), 6), out.degrees_of_freedom
        (1.632993, 3)
    """
    if not power > 0.0:
        msg = f"'power' must be positive, got {power}"
        raise ValidationError(msg)
    xp, (a, b) = floating_inputs(errors_a, errors_b)
    a, b = xp.broadcast_arrays(a, b)
    require_axis(a, axis)
    a, b = (xp.moveaxis(errors, axis, -1) for errors in (a, b))
    n = a.shape[-1]
    if not 1 <= horizon < n:
        msg = f"'horizon' must be between 1 and {n - 1}, got {horizon}"
        raise ValidationError(msg)
    # d = |a|ᵖ − |b|ᵖ
    d = xp.abs(a) ** power - xp.abs(b) ** power
    mean = xp.mean(d, axis=-1)
    c = d - mean[..., None]
    # γ₀ + 2 ∑ₖ γₖ
    variance = xp.mean(c**2, axis=-1)
    for lag in range(1, horizon):
        gamma = xp.sum(c[..., lag:] * c[..., :-lag], axis=-1) / n
        variance = variance + 2.0 * gamma
    # the variance of one value is the rounding of its mean
    varies = xp.max(d, axis=-1) > xp.min(d, axis=-1)
    defined = varies & (variance > 0)
    safe = xp.where(defined, variance, xp.ones_like(variance))
    factor = math.sqrt(n)
    if correction:
        factor *= math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = mean / xp.sqrt(safe) * factor
    return DieboldMariano(
        statistic=xp.where(defined, statistic, xp.full_like(statistic, xp.nan)),
        degrees_of_freedom=n - 1,
    )
