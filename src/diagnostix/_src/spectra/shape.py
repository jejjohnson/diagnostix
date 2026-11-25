"""One number that says what a spectrum looks like: a slope, a scale."""

from __future__ import annotations

from typing import Any

from pipekit_array.arrays import divide_or_nan, floating_inputs

from diagnostix._src.climate.trend import linear_trend
from diagnostix._src.spectra.power import Spectrum
from diagnostix.exceptions import ValidationError


__all__ = ["integral_scale", "spectral_slope"]


def spectral_slope(spectrum: Spectrum, k_min: float, k_max: float) -> Any:
    """The slope of a spectrum over a band: s in P(k) ∝ kˢ.

    The least-squares line of log P against log k, over the bins with
    k_min ≤ k ≤ k_max and k > 0, each with the same weight.

    Args:
        spectrum: P(k).
        k_min: The smallest wavenumber of the band (cycles per unit length).
        k_max: The largest. Beyond the smaller of the two Nyquist wavenumbers
            the rings of an `isotropic_spectrum` are incomplete.

    Returns:
        s, dimensionless, without the axis of the wavenumbers. NaN where the
        band holds fewer than two bins, or a bin of it no power.

    Raises:
        BackendError: If the spectrum is of DataArrays or of another library.

    Examples:
        >>> import numpy as np
        >>> k = 0.1 * np.arange(1.0, 9.0)
        >>> round(float(spectral_slope(Spectrum(k, 5.0 * k**-3), 0.2, 0.7)), 12)
        -3.0
    """
    xp, (k, power) = floating_inputs(spectrum.wavenumber, spectrum.power)
    in_band = (k >= k_min) & (k <= k_max) & (k > 0)
    positive = power > 0
    # log 1 in place of the logarithms that do not exist, or are not used
    log_k = xp.log(xp.where(in_band, k, xp.ones_like(k)))
    log_power = xp.log(xp.where(in_band & positive, power, xp.ones_like(power)))
    missing = in_band & xp.logical_not(positive)
    log_power = xp.where(missing, xp.full_like(power, xp.nan), log_power)
    return linear_trend(log_power, log_k, valid=in_band).slope


def integral_scale(spectrum: Spectrum, *, moment: int = 1) -> Any:
    """A length that a spectrum's variance is spread about.

    ℓ = (∑ P / ∑ kᵐ P)^(1/m), over the bins of the spectrum: the integral scale
    for m = 1, and the Taylor microscale for m = 2.

    It is a wavelength, the inverse of a wavenumber in cycles per unit length,
    with no factor of 2π. The mean of a field counts as variance at k = 0, so
    the spectrum is one of a field without it.

    Args:
        spectrum: P(k), over evenly spaced wavenumbers.
        moment: m ≥ 1. The larger it is, the more the small scales count.

    Returns:
        ℓ, in the unit of 1/k, without the axis of the wavenumbers. NaN where
        all of the power is at k = 0, or there is none.

    Raises:
        BackendError: If the spectrum is of DataArrays or of another library.
        ValidationError: If ``moment`` is below 1.

    Examples:
        >>> import numpy as np
        >>> k = np.asarray([0.0, 0.1, 0.2])
        >>> round(
        ...     float(integral_scale(Spectrum(k, np.asarray([0.0, 1.0, 1.0])))), 9
        ... )
        6.666666667
    """
    if moment < 1:
        msg = f"'moment' must be at least 1, got {moment}"
        raise ValidationError(msg)
    xp, (k, power) = floating_inputs(spectrum.wavenumber, spectrum.power)
    # ℓᵐ = ∑ P / ∑ kᵐ P
    total = xp.sum(power, axis=-1)
    weighted = xp.sum(k**moment * power, axis=-1)
    return divide_or_nan(total, weighted, xp) ** (1.0 / moment)
