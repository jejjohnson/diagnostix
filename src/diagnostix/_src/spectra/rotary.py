"""Rotary spectra: which way a vector turns, at each wavenumber."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.functions import constants_like, floating_inputs, squared_magnitude
from pipekit_array.records import register_record

from diagnostix._src.reduce import require_axis
from diagnostix._src.spectra.taper import (
    Detrend,
    Window,
    checked_spacing,
    tapered,
)
from diagnostix._typing import ArrayLike


__all__ = ["RotarySpectrum", "rotary_spectrum"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class RotarySpectrum:
    """A vector's variance, split by the way it turns, over wavenumber.

    The sum of both parts times the step of ``wavenumber`` is the variance of u
    plus that of v, twice the kinetic energy.

    Attributes:
        wavenumber: k, from 0 (cycles per unit length).
        counter_clockwise: The density of the part that turns from x to y as the
            position grows, with the wavenumbers on the last axis.
        clockwise: The density of the part that turns the other way.
    """

    wavenumber: Float[Any, " k"]
    counter_clockwise: Float[Any, "*batch k"]
    clockwise: Float[Any, "*batch k"]


def rotary_spectrum(
    u: ArrayLike,
    v: ArrayLike,
    spacing: float,
    *,
    axis: int = -1,
    window: Window = None,
    detrend: Detrend = "constant",
) -> RotarySpectrum:
    """A vector's variance over wavenumber, split by the way it turns.

    With ŵ the transform of the tapered u + iv and U the taper's mean square,
    the density at wavenumber k is (Δx / (N U)) |ŵ(k)|², counter-clockwise for
    k > 0 and clockwise for k < 0.

    The mean turns neither way and is split evenly. For even N there is one
    Nyquist bin, −N/2, and it is clockwise. Both parts together keep Parseval.

    Args:
        u: The x component of N ≥ 2 vectors, evenly spaced along ``axis``.
        v: The y component, broadcast against ``u``.
        spacing: Δx, the distance or the time between samples.
        axis: The axis of the samples, of both components once broadcast.
        window: As in `power_spectrum`, but none if not given.
        detrend: As in `power_spectrum`, but the mean if not given.

    Returns:
        Both parts, with N//2 + 1 wavenumbers on the last axis, wherever the
        samples were. A NaN in a series makes its whole spectrum NaN.

    Raises:
        BackendError: If a component is a DataArray or of another library.
        ValidationError: As `power_spectrum`.

    Examples:
        >>> import numpy as np
        >>> angle = 2.0 * np.pi * np.arange(8) / 4.0
        >>> out = rotary_spectrum(3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.5)
        >>> np.round(out.counter_clockwise * 0.25, 12)
        array([0., 0., 9., 0., 0.])
        >>> np.round(out.clockwise * 0.25, 12)
        array([0., 0., 0., 0., 0.])
    """
    (step,) = checked_spacing(spacing, 1)
    xp, (u, v) = floating_inputs(u, v)
    u, v = xp.broadcast_arrays(u, v)
    require_axis(u, axis)
    u, v = (xp.moveaxis(part, axis, -1) for part in (u, v))
    n = u.shape[-1]
    u, mean_square = tapered(xp, u, 1, window, detrend)
    v, _ = tapered(xp, v, 1, window, detrend)
    # ŵ(±k) = û(±k) + i v̂(±k), with û(−k) = û(k)* as u is real
    u_hat, v_hat = xp.fft.rfft(u, axis=-1), xp.fft.rfft(v, axis=-1)
    positive = squared_magnitude(u_hat + 1j * v_hat, xp)
    negative = squared_magnitude(u_hat - 1j * v_hat, xp)
    # the mean is half of each, and the one Nyquist bin is the negative one
    share = np.ones(n // 2 + 1)
    share[0] = 0.5
    turning = share.copy()
    if n % 2 == 0:
        turning[-1] = 0.0
    scale = step / (n * mean_square)
    return RotarySpectrum(
        wavenumber=constants_like(np.fft.rfftfreq(n, step), u, xp),
        counter_clockwise=positive * constants_like(turning * scale, u, xp),
        clockwise=negative * constants_like(share * scale, u, xp),
    )
