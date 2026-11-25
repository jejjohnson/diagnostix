"""Power spectra along one axis, over a plane, and over the rings of a plane."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.functions import floating_inputs

from diagnostix._src.records import register_record
from diagnostix._src.reduce import constants_like, require_axis
from diagnostix._src.spectra.rings import Rings
from diagnostix._src.spectra.taper import (
    Detrend,
    Window,
    as_complex,
    checked_spacing,
    one_sided,
    squared_magnitude,
    tapered,
)
from diagnostix._typing import ArrayLike


__all__ = [
    "Spectrum",
    "Spectrum2D",
    "isotropic_spectrum",
    "power_spectrum",
    "power_spectrum_2d",
]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Spectrum:
    """A one-sided density of variance over evenly spaced wavenumbers.

    The sum of ``power`` times the step of ``wavenumber`` is the variance.

    Attributes:
        wavenumber: k, from 0 (cycles per unit length).
        power: P(k), the square of the field's unit per cycle per unit length,
            with the wavenumbers on the last axis.
    """

    wavenumber: Float[Any, " k"]
    power: Float[Any, "*batch k"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Spectrum2D:
    """A two-sided density of variance over the wavenumber plane.

    The sum of ``power`` times both steps is the variance.

    Attributes:
        wavenumber_y: k_y, ascending (cycles per unit length).
        wavenumber_x: k_x, ascending.
        power: P₂(k_y, k_x), the square of the field's unit per (cycle per unit
            length)².
    """

    wavenumber_y: Float[Any, " ky"]
    wavenumber_x: Float[Any, " kx"]
    power: Float[Any, "*batch ky kx"]


def power_spectrum(
    x: ArrayLike,
    spacing: float,
    *,
    axis: int = -1,
    window: Window = "hann",
    detrend: Detrend = "linear",
) -> Spectrum:
    """The density of a series' variance over wavenumber or frequency.

    P(k) = c (Δx / (N U)) |x̂(k)|², with x̂ the transform of the tapered samples,
    U the taper's mean square, and c = 2 but for the mean and, for even N, the
    Nyquist bin, where it is 1.

    ∑ₖ P(k) Δk is the mean square of the tapered samples over U: the variance of
    the detrended ones where there is no taper, and on average where there is.

    Args:
        x: N ≥ 2 samples of a real series, evenly spaced along ``axis``.
        spacing: Δx, the distance or the time between samples.
        axis: The axis of the samples.
        window: ``"hann"``, periodic; a host array of N weights, as
            ``scipy.signal.windows`` makes; or ``None``.
        detrend: What to take away before the taper: the least-squares line
            (``"linear"``), the mean (``"constant"``) or nothing.

    Returns:
        The spectrum, with N//2 + 1 wavenumbers on the last axis, wherever the
        samples were. A NaN in a series makes its whole spectrum NaN: a gap is
        filled by the caller, or not at all.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: If ``spacing`` is not a positive number, or ``axis``,
            ``window`` or ``detrend`` is not one of those above.

    Examples:
        >>> import numpy as np
        >>> x = 3.0 * np.cos(2.0 * np.pi * np.arange(8) / 4.0)
        >>> out = power_spectrum(x, 0.5, window=None, detrend="constant")
        >>> out.wavenumber
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> np.round(out.power * 0.25, 12)  # a²/2 in the bin of k = 0.5
        array([0. , 0. , 4.5, 0. , 0. ])
    """
    (step,) = checked_spacing(spacing, 1)
    xp, (x,) = floating_inputs(x)
    require_axis(x, axis)
    x = xp.moveaxis(x, axis, -1)
    n = x.shape[-1]
    x, mean_square = tapered(xp, x, 1, window, detrend)
    # P(k) = c (Δx / (N U)) |x̂(k)|²
    density = squared_magnitude(xp, xp.fft.rfft(x, axis=-1))
    scale = one_sided(n) * (step / (n * mean_square))
    return Spectrum(
        wavenumber=constants_like(xp, np.fft.rfftfreq(n, step), x),
        power=density * constants_like(xp, scale, x),
    )


def power_spectrum_2d(
    x: ArrayLike,
    spacing: float | Sequence[float],
    *,
    window: Window = "hann",
    detrend: Detrend = "linear",
) -> Spectrum2D:
    """The density of a field's variance over the wavenumber plane.

    P₂(k) = (Δx Δy / (Nx Ny U)) |x̂(k)|², with x̂ the transform of the tapered
    field and U the taper's mean square.

    The grid is uniform: a field on longitude and latitude goes onto a local
    plane first, or covers a small domain.

    Args:
        x: A real field that ends in (y, x), or in (time, x) for a
            wavenumber–frequency spectrum, on an evenly spaced grid.
        spacing: (Δy, Δx), or one number for both.
        window: ``"hann"``, the product of one along each axis; a host array of
            (Ny, Nx) weights; or ``None``.
        detrend: What to take away before the taper: the least-squares plane
            (``"linear"``), the mean (``"constant"``) or nothing.

    Returns:
        The spectrum, both axes ascending, so 0 is in the middle. A NaN in a
        field makes its whole spectrum NaN.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: As `power_spectrum`.

    Examples:
        >>> import numpy as np
        >>> y, x = np.mgrid[:4, :6]
        >>> wave = np.cos(2.0 * np.pi * (y / 4.0 + x / 3.0))
        >>> out = power_spectrum_2d(wave, (1.0, 2.0), window=None)
        >>> out.power.shape, out.wavenumber_y
        ((4, 6), array([-0.5 , -0.25,  0.  ,  0.25]))
        >>> step = (1.0 / 4.0) * (1.0 / 12.0)
        >>> round(float(np.sum(out.power)) * step, 12)
        0.5
    """
    steps = checked_spacing(spacing, 2)
    xp, (x,) = floating_inputs(x)
    x, mean_square = tapered(xp, x, 2, window, detrend)
    shape = tuple(x.shape[-2:])
    z = xp.fft.fftn(as_complex(xp, x), axes=(-2, -1))
    z = xp.fft.fftshift(z, axes=(-2, -1))
    # P₂ = (Δx Δy / (Nx Ny U)) |x̂|²
    scale = steps[0] * steps[1] / (shape[0] * shape[1] * mean_square)
    wavenumbers = (
        np.fft.fftshift(np.fft.fftfreq(n, step))
        for n, step in zip(shape, steps, strict=True)
    )
    ky, kx = (constants_like(xp, k, x) for k in wavenumbers)
    return Spectrum2D(
        wavenumber_y=ky, wavenumber_x=kx, power=squared_magnitude(xp, z) * scale
    )


def isotropic_spectrum(
    x: ArrayLike,
    spacing: float | Sequence[float],
    *,
    window: Window = "hann",
    detrend: Detrend = "linear",
) -> Spectrum:
    """The density of a field's variance over the size of the wavenumber.

    P(kₙ) = (Δk_x Δk_y / Δk) ∑ P₂(k) over the ring of modes with |k| within Δk/2
    of kₙ = n Δk, where P₂ is the `power_spectrum_2d` and Δk the smaller of Δk_x
    and Δk_y.

    It is a sum over the ring, not a mean, so ∑ₙ P(kₙ) Δk is the variance, as of
    a one-dimensional spectrum, and a two-dimensional white field grows with k.
    Every mode is counted: the rings beyond the smaller of the two Nyquist
    wavenumbers are incomplete, and fall off whatever the field does.

    Args:
        x: A real field that ends in (y, x), on an evenly spaced grid.
        spacing: (Δy, Δx), or one number for both.
        window: As `power_spectrum_2d`.
        detrend: As `power_spectrum_2d`.

    Returns:
        The spectrum, with the rings on the last axis. A NaN in a field makes
        its whole spectrum NaN.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: As `power_spectrum`.

    Examples:
        >>> import numpy as np
        >>> y, x = np.mgrid[:16, :16]
        >>> wave = np.cos(2.0 * np.pi * (3 * y + 4 * x) / 16.0)  # |k| = 5 Δk
        >>> out = isotropic_spectrum(wave, 1.0, window=None, detrend=None)
        >>> np.round(out.power[3:8] / 16.0, 12)
        array([0. , 0. , 0.5, 0. , 0. ])
    """
    steps = checked_spacing(spacing, 2)
    xp, (x,) = floating_inputs(x)
    x, mean_square = tapered(xp, x, 2, window, detrend)
    ny, nx = x.shape[-2:]
    rings = Rings.of_grid((ny, nx), (steps[0], steps[1]))
    density = squared_magnitude(xp, xp.fft.rfftn(x, axes=(-2, -1)))
    # P = (Δk_x Δk_y / Δk) ∑ P₂, with P₂ = (Δx Δy / (Nx Ny U)) |x̂|²
    scale = steps[0] * steps[1] / (ny * nx * mean_square) * rings.cell / rings.step
    return Spectrum(
        wavenumber=constants_like(xp, rings.wavenumber, x),
        power=rings.sum(xp, density * scale),
    )
