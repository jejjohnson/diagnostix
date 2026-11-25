"""The continuous wavelet transform along one axis, its inverse and significance."""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist
from typing import Any

import numpy as np
from jaxtyping import Bool, Complex, Float
from pipekit_array.arrays import (
    constants_like,
    floating_inputs,
    namespace_of,
    register_record,
    squared_magnitude,
)

from diagnostix._src.reduce import require_axis
from diagnostix._src.spectra.mothers import Mother
from diagnostix._src.spectra.taper import (
    Detrend,
    as_complex,
    checked_spacing,
    demean_or_detrend,
    require_samples,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "WaveletTransform",
    "checked_scales",
    "inverse_wavelet_transform",
    "wavelet_significance",
    "wavelet_transform",
]

#: How far the ratios of geometric scales may be from each other.
GEOMETRIC_RTOL = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class WaveletTransform:
    """The wavelet coefficients of a series, and what they were made with.

    Under JAX only ``coefficients`` is traced.

    Attributes:
        coefficients: W(s, t), complex, in the unit of the series, with the
            scales and the samples on the last two axes.
        scales: s, ascending, in the unit of the spacing.
        spacing: Δt, the distance or the time between samples.
        mother: The name of the mother wavelet.
        param: Its parameter.
        periodic: Whether the series was taken to repeat.
    """

    coefficients: Complex[Any, "*batch scale n"]
    scales: tuple[float, ...]
    spacing: float
    mother: str
    param: float
    periodic: bool

    @property
    def power(self) -> Float[Any, "*batch scale n"]:
        """|W|², the square of the unit of the series.

        Divided by the scale it is the power of Liu et al. (2007), in which
        waves of one amplitude and different periods are equally high.
        """
        xp = namespace_of(self.coefficients)
        return squared_magnitude(self.coefficients, xp)

    @property
    def period(self) -> Float[np.ndarray, " scale"]:
        """The Fourier period of each scale, on the host."""
        return np.asarray(self.scales) * Mother(self.mother, self.param).period

    @property
    def inside_cone(self) -> Bool[np.ndarray, "scale n"]:
        """Which coefficients are away from the ends of the series, on the host.

        An end is felt over √2 s (s/√2 for ``"paul"``), where the power of a spike
        there has fallen by a factor of e². A periodic series has no ends.
        """
        n = self.coefficients.shape[-1]
        position = np.arange(n)
        distance = self.spacing * np.minimum(position, n - 1 - position)
        if self.periodic:
            distance = np.full(n, np.inf)
        reach = Mother(self.mother, self.param).reach
        return distance >= reach * np.asarray(self.scales)[:, None]


register_record(
    WaveletTransform,
    meta_fields=("scales", "spacing", "mother", "param", "periodic"),
)


def wavelet_transform(
    x: ArrayLike,
    spacing: float,
    scales: ArrayLike,
    *,
    axis: int = -1,
    mother: str = "morlet",
    param: float | None = None,
    detrend: Detrend = "constant",
    periodic: bool = False,
) -> WaveletTransform:
    """The continuous wavelet transform of a series (Torrence and Compo 1998).

    W(s, t) = ∑ₖ x̂ₖ (2πs/Δt)^½ ψ̂₀(s ωₖ)* exp(i ωₖ t), with x̂ₖ the coefficients of
    the series and ψ̂₀ the transform of the mother, whose energy is 1. White
    noise of variance σ² has a mean power |W|² of σ² at every scale.

    Args:
        x: N ≥ 2 samples of a real series, evenly spaced along ``axis``.
        spacing: Δt, the distance or the time between samples.
        scales: s, concrete, positive and ascending, in the unit of ``spacing``:
            usually from 2 Δt, in steps of a fraction of an octave. The smallest
            ones reach beyond the Nyquist frequency, and are cut there.
        axis: The axis of the samples.
        mother: ``"morlet"``, ``"paul"``, or ``"dog"``, a derivative of a
            Gaussian.
        param: ω₀ of a Morlet (6 if not given), the order of a Paul (4), or the
            derivative (2, the Mexican hat).
        detrend: What to take away first: the mean (``"constant"``), the line
            (``"linear"``) or nothing. A mean left in is a step at both ends.
        periodic: Whether the series repeats. If not, it is padded with zeros to
            twice its length, so that nothing wraps.

    Returns:
        The transform, with the scales and the samples on the last two axes. A
        NaN in a series makes all its coefficients NaN.

    Raises:
        BackendError: If ``x`` is a DataArray or of another library.
        ValidationError: If ``spacing`` or a scale is not positive, the scales
            do not ascend, or ``axis``, ``mother``, ``param`` or ``detrend`` is
            not one of those above.

    Examples:
        >>> import numpy as np
        >>> t = 0.5 * np.arange(256)
        >>> x = np.cos(2.0 * np.pi * t / 8.0)
        >>> out = wavelet_transform(x, 0.5, 2.0 ** np.arange(0.0, 5.0, 0.25))
        >>> peak = int(np.argmax(out.power[:, 128]))
        >>> round(float(out.period[peak]), 2)
        8.26
        >>> int(np.sum(out.inside_cone[peak]))  # of 256
        210
    """
    (step,) = checked_spacing(spacing, 1)
    kind = Mother.of(mother, param)
    s = checked_scales(scales)
    xp, (x,) = floating_inputs(x)
    require_axis(x, axis)
    x = xp.moveaxis(x, axis, -1)
    (n,) = require_samples(x, 1)
    x = demean_or_detrend(xp, x, 1, detrend)
    length = n if periodic else 2 * n
    omega = 2.0 * np.pi * np.fft.fftfreq(length, step)
    # (2πs/Δt)^½ ψ̂₀(s ω)*
    daughters = np.sqrt(2.0 * np.pi * s / step)[:, None] * kind.fourier(
        s[:, None] * omega
    )
    x_hat = xp.fft.fft(as_complex(xp, x), n=length, axis=-1)
    w = x_hat[..., None, :] * constants_like(daughters, x, xp)
    phase = kind.phase.real if kind.is_real_in_fourier else kind.phase
    w = xp.fft.ifft(w * phase, axis=-1)[..., :n]
    return WaveletTransform(
        coefficients=w,
        scales=tuple(s.tolist()),
        spacing=step,
        mother=kind.name,
        param=kind.param,
        periodic=periodic,
    )


def inverse_wavelet_transform(
    transform: WaveletTransform, *, band: tuple[float, float] | None = None
) -> Any:
    """A series back from its wavelet coefficients, or a band of periods of it.

    x(t) = (δj Δt^½ / C) ∑ⱼ Re W(sⱼ, t) / sⱼ^½, with δj the octaves between scales
    and C the integral of ψ̂₀(η)/η (Torrence and Compo 1998, eq. 11).

    It is a sum over scales in place of an integral, so it is as good as they are
    dense and wide. The mean is at no scale, and is not given back.

    Args:
        transform: The transform, with scales in a geometric sequence.
        band: The shortest and the longest period to keep; all if not given.

    Returns:
        x, with the samples on the last axis.

    Raises:
        ValidationError: If there are fewer than two scales, they are not a
            geometric sequence, or the mother has no such inverse: an odd
            derivative of a Gaussian, or a Morlet with ω₀ < 5.

    Examples:
        >>> import numpy as np
        >>> t = 0.5 * np.arange(256)
        >>> x = np.cos(2.0 * np.pi * t / 8.0) + np.cos(2.0 * np.pi * t / 32.0)
        >>> scales = 0.25 * 2.0 ** np.arange(0.0, 10.0, 0.125)
        >>> out = wavelet_transform(x, 0.5, scales, periodic=True)
        >>> back = inverse_wavelet_transform(out)
        >>> float(np.max(np.abs(back - x))) < 1e-5
        True
        >>> fast = inverse_wavelet_transform(out, band=(4.0, 16.0))
        >>> float(np.max(np.abs(fast - np.cos(2.0 * np.pi * t / 8.0)))) < 0.02
        True
    """
    constant = Mother(transform.mother, transform.param).inverse_constant()
    scales = np.asarray(transform.scales)
    octaves = np.diff(np.log2(scales))
    if octaves.size == 0 or np.ptp(octaves) > GEOMETRIC_RTOL * np.mean(octaves):
        msg = "the inverse needs at least 2 scales, in a geometric sequence"
        raise ValidationError(msg)
    # δj Δt^½ / (C sⱼ^½)
    weights = np.mean(octaves) * np.sqrt(transform.spacing / scales) / constant
    if band is not None:
        period = transform.period
        weights = np.where((period >= band[0]) & (period <= band[1]), weights, 0.0)
    w = transform.coefficients
    xp = namespace_of(w)
    real = xp.real(w)
    # x = ∑ⱼ weightⱼ Re W(sⱼ)
    return xp.sum(real * constants_like(weights[:, None], real, xp), axis=-2)


def wavelet_significance(
    transform: WaveletTransform,
    variance: ArrayLike,
    *,
    lag1: ArrayLike = 0.0,
    confidence: float = 0.95,
) -> Any:
    """The power at each scale that red noise exceeds with a small probability.

    σ² (1 − α²) / (1 + α² − 2α cos(2πΔt/period)) · χ²_q / q, with α the lag-1
    autocorrelation of the noise, and χ²_q the quantile at the confidence for
    q = 2 degrees of freedom, or 1 for the real ``"dog"`` (Torrence and Compo
    1998, eqs. 16–18).

    The probability is that of one coefficient. Of many, the rest exceed it too,
    and neighbours exceed it together.

    Args:
        transform: The transform.
        variance: σ², of the noise or of the series, one per series.
        lag1: α in (−1, 1), one or one per series; 0 is white noise.
        confidence: The probability that noise stays below, in (0, 1).

    Returns:
        The threshold of ``transform.power``, with the scales on the last axis.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``confidence`` is not in (0, 1).

    Examples:
        >>> import numpy as np
        >>> x = np.random.default_rng(0).standard_normal(4096)
        >>> out = wavelet_transform(x, 1.0, 2.0 ** np.arange(1.0, 6.0))
        >>> threshold = wavelet_significance(out, np.var(x))
        >>> high = out.power > threshold[:, None]
        >>> round(float(np.mean(high[out.inside_cone])), 2)
        0.05
    """
    if not 0.0 < confidence < 1.0:
        msg = f"'confidence' must be in (0, 1), got {confidence}"
        raise ValidationError(msg)
    w = transform.coefficients
    xp, (sigma2, alpha) = floating_inputs(variance, lag1, params=(w,))
    real = xp.real(w)
    sigma2, alpha = (xp.astype(a, real.dtype)[..., None] for a in (sigma2, alpha))
    degrees = Mother(transform.mother, transform.param).degrees_of_freedom
    if degrees == 2:
        # χ²₂ is exponential of mean 2
        quantile = -2.0 * math.log1p(-confidence)
    else:
        # χ²₁ is the square of a normal draw
        quantile = NormalDist().inv_cdf(0.5 * (1.0 + confidence)) ** 2
    cosine = np.cos(2.0 * np.pi * transform.spacing / transform.period)
    cosine = constants_like(cosine, real, xp)
    # (1 − α²) / (1 + α² − 2α cos(2πΔt/period))
    red = (1.0 - alpha**2) / (1.0 + alpha**2 - 2.0 * alpha * cosine)
    return sigma2 * red * (quantile / degrees)


def checked_scales(scales: ArrayLike) -> Float[np.ndarray, " scale"]:
    """Scales as a host table.

    Raises:
        ValidationError: If they are not 1-D, positive, finite and ascending.
    """
    s = np.asarray(scales, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all():
        msg = f"'scales' must be 1-D and finite, got shape {s.shape}"
        raise ValidationError(msg)
    if s[0] <= 0.0 or (np.diff(s) <= 0.0).any():
        msg = "'scales' must be positive and ascending"
        raise ValidationError(msg)
    return s
