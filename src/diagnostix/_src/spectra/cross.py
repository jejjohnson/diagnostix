"""Cross spectra and coherence, averaged over segments and realisations."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Complex, Float
from pipekit_array.functions import floating_inputs

from diagnostix._src.records import register_record
from diagnostix._src.reduce import (
    constants_like,
    on_device,
    require_axis,
)
from diagnostix._src.spectra.taper import (
    Detrend,
    Window,
    checked_spacing,
    one_sided,
    squared_magnitude,
    taper,
    tapered,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["Coherence", "CrossSpectrum", "coherence", "cross_spectrum"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class CrossSpectrum:
    """A one-sided density of the covariance of two series over wavenumber.

    The sum of the real part of ``cross`` times the step of ``wavenumber`` is
    the covariance.

    Attributes:
        wavenumber: k, from 0 (cycles per unit length).
        cross: ⟨x̂ ŷ*⟩ as a density, complex, with the wavenumbers on the last
            axis. Its phase is positive where x leads y.
    """

    wavenumber: Float[Any, " k"]
    cross: Complex[Any, "*batch k"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Coherence:
    """The share of two series' variance that is linearly related, by wavenumber.

    Attributes:
        wavenumber: k, from 0 (cycles per unit length).
        coherence: C(k) in [0, 1], the magnitude squared, with the wavenumbers
            on the last axis.
    """

    wavenumber: Float[Any, " k"]
    coherence: Float[Any, "*batch k"]


def cross_spectrum(
    x: ArrayLike,
    y: ArrayLike,
    spacing: float,
    *,
    axis: int = -1,
    segment: int | None = None,
    overlap: float = 0.5,
    window: Window = "hann",
    detrend: Detrend = "constant",
) -> CrossSpectrum:
    """The density of two series' covariance over wavenumber or frequency.

    P_xy(k) = c (Δx / (L U)) ⟨x̂(k) ŷ(k)*⟩, with ⟨·⟩ the mean over segments of L
    samples, each detrended and tapered by itself, U the taper's mean square,
    and c = 2 but for the mean and, for even L, the Nyquist bin.

    The phase is positive where x leads y. SciPy's ``csd(x, y)`` is the complex
    conjugate. With y = x it is Welch's power spectrum.

    Args:
        x: N ≥ 2 samples of a real series, evenly spaced along ``axis``.
        y: The other series, broadcast against ``x``.
        spacing: Δx, the distance or the time between samples.
        axis: The axis of the samples, of both series once broadcast.
        segment: L, the samples of a segment; all N if not given. Samples after
            the last whole segment are not used.
        overlap: The share of a segment that the next one repeats, in [0, 1).
        window: ``"hann"``, periodic; a host array of L weights; or ``None``.
        detrend: What to take away from each segment: the mean (``"constant"``),
            the least-squares line (``"linear"``) or nothing.

    Returns:
        The spectrum, with L//2 + 1 wavenumbers on the last axis, wherever the
        samples were.

    Raises:
        BackendError: If a series is a DataArray or of another library.
        ValidationError: If ``spacing`` is not a positive number, a segment is
            not between 2 and N samples, ``overlap`` is not in [0, 1), or
            ``axis``, ``window`` or ``detrend`` is not one of those above.

    Examples:
        >>> import numpy as np
        >>> t = 0.25 * np.arange(16)
        >>> x = np.cos(2.0 * np.pi * t)
        >>> y = 3.0 * np.cos(2.0 * np.pi * (t - 0.125))  # an eighth later
        >>> out = cross_spectrum(x, y, 0.25, window=None)
        >>> float(out.wavenumber[4])
        1.0
        >>> cross = out.cross[4]
        >>> round(float(np.abs(cross)), 9), round(float(np.angle(cross) / np.pi), 9)
        (6.0, 0.25)
    """
    xp, (x, y), _ = _series(x, y, axis, ())
    transform = _transform(xp, x, spacing, segment, overlap, window, detrend)
    x_hat, y_hat = transform(x), transform(y)
    # P_xy = c (Δx / (L U)) ⟨x̂ ŷ*⟩
    cross = xp.mean(x_hat * xp.conj(y_hat), axis=-2)
    scale = constants_like(xp, transform.scale, x)
    return CrossSpectrum(
        wavenumber=constants_like(xp, transform.wavenumber, x), cross=cross * scale
    )


def coherence(
    x: ArrayLike,
    y: ArrayLike,
    spacing: float,
    *,
    axis: int = -1,
    segment: int | None = None,
    average: int | Sequence[int] | None = None,
    overlap: float = 0.5,
    window: Window = "hann",
    detrend: Detrend = "constant",
) -> Coherence:
    """How much of two series is linearly related, at each wavenumber.

    C(k) = |⟨x̂ ŷ*⟩|² / (⟨|x̂|²⟩ ⟨|ŷ|²⟩), with ⟨·⟩ the mean over segments, over
    realisations, or both.

    Without an average C is 1 for any two series, so one is required. An
    average of M independent transforms of independent noise gives 1/M, not 0.

    Args:
        x: N samples of a real series, evenly spaced along ``axis``.
        y: The other series, broadcast against ``x``.
        spacing: Δx, the distance or the time between samples.
        axis: The axis of the samples, of both series once broadcast.
        segment: L, the samples of a segment, as in `cross_spectrum`.
        average: The axis or axes of realisations to average over.
        overlap: The share of a segment that the next one repeats, in [0, 1).
        window: As in `cross_spectrum`.
        detrend: As in `cross_spectrum`.

    Returns:
        C, without the averaged axes, with the wavenumbers on the last one. It
        is NaN where either series has no power. Where the mean is removed and
        there is no taper, bin 0 holds rounding only.

    Raises:
        BackendError: If a series is a DataArray or of another library.
        ValidationError: As `cross_spectrum`, or if ``segment`` and ``average``
            together give one transform only, or ``average`` names the axis of
            the samples.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> shared, noise = rng.standard_normal((2, 4096))
        >>> out = coherence(shared, shared + noise, 1.0, segment=256)
        >>> round(float(np.mean(out.coherence)), 2)  # half of the variance
        0.5
    """
    axes = () if average is None else np.atleast_1d(np.asarray(average)).tolist()
    xp, (x, y), averaged = _series(x, y, axis, axes)
    transform = _transform(xp, x, spacing, segment, overlap, window, detrend)
    count = transform.n_segments * int(np.prod([x.shape[a] for a in averaged]))
    if count < 2:
        msg = (
            "coherence is 1 without an average: 'segment' and 'average' must "
            f"give at least 2 transforms, got {count}"
        )
        raise ValidationError(msg)
    x_hat, y_hat = transform(x), transform(y)
    over = (*averaged, x_hat.ndim - 2)

    def mean(values: Any) -> Any:
        return xp.mean(values, axis=over)

    power_x = mean(squared_magnitude(xp, x_hat))
    power_y = mean(squared_magnitude(xp, y_hat))
    defined = (power_x > 0) & (power_y > 0)
    one = xp.ones_like(power_x)
    # the product of small powers would underflow in float32, and so the roots
    root_x = xp.sqrt(xp.where(defined, power_x, one))
    root_y = xp.sqrt(xp.where(defined, power_y, one))
    # C = |⟨x̂ ŷ*⟩ / (⟨|x̂|²⟩ ⟨|ŷ|²⟩)^½|²
    related = mean(x_hat * xp.conj(y_hat)) / (root_x * root_y)
    share = squared_magnitude(xp, related)
    return Coherence(
        wavenumber=constants_like(xp, transform.wavenumber, x),
        coherence=xp.where(defined, share, xp.full_like(share, xp.nan)),
    )


def _series(
    x: ArrayLike, y: ArrayLike, axis: int, average: Sequence[int]
) -> tuple[ModuleType, tuple[Any, Any], tuple[int, ...]]:
    """Two series broadcast, with their samples last, and the axes to average.

    Raises:
        ValidationError: If there is no such axis, or one is named twice.
    """
    xp, (x, y) = floating_inputs(x, y)
    x, y = xp.broadcast_arrays(x, y)
    for one in (axis, *average):
        require_axis(x, one)
    named = [one % x.ndim for one in (axis, *average)]
    if len(set(named)) < len(named):
        msg = f"'average' must not repeat an axis or name that of the samples, {axis}"
        raise ValidationError(msg)
    x, y = (xp.moveaxis(values, axis, -1) for values in (x, y))
    # an axis after that of the samples is one place earlier now
    return xp, (x, y), tuple(one - (one > named[0]) for one in named[1:])


@dataclasses.dataclass(frozen=True)
class _Segments:
    """The transform of each segment of a series.

    Attributes:
        index: The position of each sample of each segment.
        wavenumber: k.
        scale: c Δx / (L U), which makes a density.
    """

    xp: ModuleType
    index: np.ndarray
    wavenumber: np.ndarray
    scale: np.ndarray
    window: Window
    detrend: Detrend

    @property
    def n_segments(self) -> int:
        return self.index.shape[0]

    def __call__(self, x: Any) -> Any:
        """(…, N) samples as (…, segment, k) coefficients, not scaled."""
        xp = self.xp
        if self.n_segments == 1 and self.index.shape[1] == x.shape[-1]:
            rows = x[..., None, :]
        else:
            flat = on_device(xp, self.index.reshape(-1), x)
            rows = xp.take(x, flat, axis=-1)
            rows = xp.reshape(rows, (*x.shape[:-1], *self.index.shape))
        rows, _ = tapered(xp, rows, 1, self.window, self.detrend)
        return xp.fft.rfft(rows, axis=-1)


def _transform(
    xp: ModuleType,
    x: Any,
    spacing: float,
    segment: int | None,
    overlap: float,
    window: Window,
    detrend: Detrend,
) -> _Segments:
    """The segments of a series of x's length, and their scale.

    Raises:
        ValidationError: As `cross_spectrum`.
    """
    (step,) = checked_spacing(spacing, 1)
    n = x.shape[-1]
    length = n if segment is None else segment
    if n < 2 or not 2 <= length <= n:
        msg = f"a segment must have between 2 samples and the {n} given, got {length}"
        raise ValidationError(msg)
    if not 0.0 <= overlap < 1.0:
        msg = f"'overlap' must be in [0, 1), got {overlap}"
        raise ValidationError(msg)
    hop = length - int(overlap * length)
    starts = np.arange(0, n - length + 1, hop)
    _, mean_square = taper(window, (length,))
    return _Segments(
        xp=xp,
        index=starts[:, None] + np.arange(length),
        wavenumber=np.fft.rfftfreq(length, step),
        scale=one_sided(length) * (step / (length * mean_square)),
        window=window,
        detrend=detrend,
    )
