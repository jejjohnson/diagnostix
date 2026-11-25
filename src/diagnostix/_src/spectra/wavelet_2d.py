"""A local isotropic spectrum: the variance about each point, by scale."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from jaxtyping import Bool, Float
from pipekit_array.functions import floating_inputs

from diagnostix._src.records import register_record
from diagnostix._src.reduce import constants_like
from diagnostix._src.spectra.taper import (
    Detrend,
    as_complex,
    checked_spacing,
    demean_or_detrend,
    require_samples,
    squared_magnitude,
)
from diagnostix._src.spectra.wavelet import checked_scales
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["LocalSpectrum", "wavelet_power_2d"]


@dataclasses.dataclass(frozen=True, eq=False)
class LocalSpectrum:
    """The density of a field's variance about each point, over wavenumber.

    Under JAX only ``power`` and ``wavenumber`` are traced.

    Attributes:
        wavenumber: k = 1/s of each scale, descending (cycles per unit length).
        power: P(k, y, x), the square of the field's unit per cycle per unit
            length, with the scales, y and x on the last three axes.
        scales: s, ascending, in the unit of the spacing.
        spacing: (Δy, Δx).
        periodic: Whether the field was taken to repeat.
    """

    wavenumber: Float[Any, " scale"]
    power: Float[Any, "*batch scale y x"]
    scales: tuple[float, ...]
    spacing: tuple[float, float]
    periodic: bool

    @property
    def inside_cone(self) -> Bool[np.ndarray, "scale y x"]:
        """Which points are at least √2 s away from the edges, on the host."""
        ny, nx = self.power.shape[-2:]
        offsets = (
            step * np.minimum(np.arange(n), n - 1 - np.arange(n))
            for n, step in zip((ny, nx), self.spacing, strict=True)
        )
        from_y, from_x = offsets
        distance = np.minimum(from_y[:, None], from_x[None, :])
        if self.periodic:
            distance = np.full((ny, nx), np.inf)
        reach = np.sqrt(2.0) * np.asarray(self.scales)
        return distance >= reach[:, None, None]


register_record(LocalSpectrum, meta_fields=("scales", "spacing", "periodic"))


def wavelet_power_2d(
    x: ArrayLike,
    spacing: float | Sequence[float],
    scales: ArrayLike,
    *,
    n_angles: int = 24,
    detrend: Detrend = "constant",
    periodic: bool = False,
) -> LocalSpectrum:
    """A field's isotropic spectrum about each point, from Morlet wavelets.

    P(1/s, r) = (s / C) ⟨|W(s, θ, r)|²⟩, the mean over directions θ in [0, π), where
    the wavelet of scale s is exp(−2π² |s k − e_θ|²) in the wavenumber plane: a
    wave of length s along e_θ under a Gaussian envelope of standard deviation s.

    C is such that the integral of P over k is the variance, so its mean over the
    plane is an `isotropic_spectrum`, smoothed over about a quarter of k.

    Args:
        x: A real field that ends in (y, x), on an evenly spaced grid.
        spacing: (Δy, Δx), or one number for both.
        scales: s, the wavelengths, concrete, positive and ascending, in the unit
            of ``spacing``. They are to be well above twice the larger spacing:
            at the Nyquist wavenumber half of a wavelet is missing.
        n_angles: The number of directions. The power of a plane wave at its own
            scale changes with its direction by 4 × 10⁻³ of itself for 16 and
            8 × 10⁻⁷ for 24.
        detrend: What to take away first: the mean (``"constant"``), the plane
            (``"linear"``) or nothing.
        periodic: Whether the field repeats. If not, it is padded with zeros to
            twice its size, so that nothing wraps.

    Returns:
        The spectrum, with the scales, y and x on the last three axes. A NaN in
        a field makes its whole spectrum NaN.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: If ``spacing`` or a scale is not positive, the scales
            do not ascend, or ``n_angles`` is below 1.

    Examples:
        >>> import numpy as np
        >>> y, x = np.mgrid[:64, :64]
        >>> wave = np.cos(2.0 * np.pi * 8.0 * x / 64.0)
        >>> scales = 2.0 ** np.arange(2.0, 4.01, 0.125)
        >>> out = wavelet_power_2d(wave, 1.0, scales, periodic=True)
        >>> density = out.power[:, 10, 20]
        >>> area = -np.trapezoid(density, out.wavenumber)
        >>> round(float(area), 2)  # a²/2
        0.5
    """
    dy, dx = checked_spacing(spacing, 2)
    s = checked_scales(scales)
    if n_angles < 1:
        msg = f"'n_angles' must be at least 1, got {n_angles}"
        raise ValidationError(msg)
    xp, (x,) = floating_inputs(x)
    ny, nx = require_samples(x, 2)
    x = demean_or_detrend(xp, x, 2, detrend)
    shape = (ny, nx) if periodic else (2 * ny, 2 * nx)
    x_hat = xp.fft.fftn(as_complex(xp, x), s=shape, axes=(-2, -1))
    x_hat = x_hat[..., None, :, :]
    s_ky = s[:, None] * np.fft.fftfreq(shape[0], dy)
    s_kx = s[:, None] * np.fft.fftfreq(shape[1], dx)
    total = None
    for theta in np.pi * np.arange(n_angles) / n_angles:
        # exp(−2π² |s k − e_θ|²), the product of one factor per axis
        along_y = np.exp(-2.0 * np.pi**2 * (s_ky - np.sin(theta)) ** 2)
        along_x = np.exp(-2.0 * np.pi**2 * (s_kx - np.cos(theta)) ** 2)
        w_hat = x_hat * constants_like(xp, along_y[:, :, None], x)
        w_hat = w_hat * constants_like(xp, along_x[:, None, :], x)
        w = xp.fft.ifftn(w_hat, axes=(-2, -1))[..., :ny, :nx]
        power = squared_magnitude(xp, w)
        total = power if total is None else total + power
    # P = (s / C) ⟨|W|²⟩
    scale = s[:, None, None] / (_admissibility() * n_angles)
    return LocalSpectrum(
        wavenumber=constants_like(xp, 1.0 / s, x),
        power=total * constants_like(xp, scale, x),
        scales=tuple(s.tolist()),
        spacing=(dy, dx),
        periodic=periodic,
    )


def _admissibility() -> float:
    """C = ∫ ⟨|ψ̂(s e)|²⟩ ds/s over every scale, the mean over directions.

    The mean of exp(−4π² |s e − e_θ|²) over θ is exp(−4π² (s² + 1)) I₀(8π² s).
    """
    # the integrand is below e⁻²⁵ of its largest outside
    log_s = np.linspace(np.log(0.15), np.log(2.0), 4001)
    s = np.exp(log_s)
    mean = np.exp(-4.0 * np.pi**2 * (s**2 + 1.0)) * np.i0(8.0 * np.pi**2 * s)
    return float(np.trapezoid(mean, log_s))
