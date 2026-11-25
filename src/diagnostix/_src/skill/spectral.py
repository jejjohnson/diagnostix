"""Scores by scale: the PSD score, the wavelength it resolves, errors in bands."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.functions import divide_or_nan, floating_inputs, table_like
from pipekit_array.records import register_record

from diagnostix._src.columns.crossing import first_crossing
from diagnostix._src.skill.members import axes_of
from diagnostix._src.skill.moments import Cells, root
from diagnostix._src.spectra.power import isotropic_spectrum, power_spectrum
from diagnostix._src.spectra.taper import (
    Detrend,
    Window,
    checked_spacing,
    require_samples,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["PSDScore", "band_rmse", "psd_score", "resolved_scale"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class PSDScore:
    """The share of the reference's variance that a prediction has, by scale.

    Attributes:
        wavenumber: k, from 0 (cycles per unit length).
        score: S(k): 1 where the error has no power, 0 where it has as much as
            the reference, and negative beyond, with the wavenumbers on the
            last axis.
    """

    wavenumber: Float[Any, " k"]
    score: Float[Any, "*batch k"]


def psd_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    spacing: float | Sequence[float],
    *,
    isotropic: bool = True,
    average: int | tuple[int, ...] | None = None,
    window: Window = "hann",
    detrend: Detrend = "linear",
) -> PSDScore:
    """1 minus the power of the error over that of the reference, by wavenumber.

    S(k) = 1 − ⟨P(ŷ − y)⟩(k) / ⟨P(y)⟩(k), with P a spectrum and ⟨·⟩ the mean over
    ``average``, taken before the ratio.

    Args:
        prediction: ŷ, a real field on an evenly spaced grid.
        reference: y, broadcast against ``prediction``.
        spacing: (Δy, Δx), or one number, as in `isotropic_spectrum`; Δx without
            ``isotropic``.
        isotropic: Whether P is the `isotropic_spectrum` of the last two axes, or
            the `power_spectrum` along the last one.
        average: The axis or axes to average over, none of those.
        window: As in `power_spectrum`.
        detrend: As in `power_spectrum`.

    Returns:
        The score, without the averaged axes. NaN where the reference has no
        power, and in a ring that holds no mode, which a long grid may have
        beyond the smaller of its Nyquist wavenumbers. Where the mean is removed,
        the score at k = 0 says nothing.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: As `power_spectrum`, or if ``average`` names an axis
            that is transformed, or does not exist.

    Examples:
        >>> import numpy as np
        >>> x = np.arange(32)
        >>> slow = np.cos(2.0 * np.pi * 2 * x / 32)
        >>> fast = np.cos(2.0 * np.pi * 8 * x / 32)
        >>> out = psd_score(
        ...     slow, slow + fast, 1.0, isotropic=False, window=None, detrend=None
        ... )
        >>> np.round(out.score[[2, 8]], 12)
        array([1., 0.])
    """
    xp, (p, r) = floating_inputs(prediction, reference)
    p, r = xp.broadcast_arrays(p, r)
    n_axes = 2 if isotropic else 1
    require_samples(r, n_axes)
    axes = axes_of(r, average)
    if any(axis >= r.ndim - n_axes for axis in axes):
        msg = f"'average' must not name an axis that is transformed, got {average}"
        raise ValidationError(msg)
    if isotropic:
        error = isotropic_spectrum(p - r, spacing, window=window, detrend=detrend)
        truth = isotropic_spectrum(r, spacing, window=window, detrend=detrend)
    else:
        (step,) = checked_spacing(spacing, 1)
        error = power_spectrum(p - r, step, window=window, detrend=detrend)
        truth = power_spectrum(r, step, window=window, detrend=detrend)
    of_error, of_truth = (
        xp.mean(spectrum.power, axis=axes) if axes else spectrum.power
        for spectrum in (error, truth)
    )
    # S = 1 − ⟨P(ŷ − y)⟩ / ⟨P(y)⟩
    return PSDScore(
        wavenumber=truth.wavenumber,
        score=1.0 - divide_or_nan(of_error, of_truth, xp),
    )


def resolved_scale(score: PSDScore, *, level: float = 0.5) -> Any:
    """The wavelength down to which a prediction has the reference's variance.

    1/k at the first k > 0 where the score falls to ``level``, interpolated in
    log k. Compute a dask array first, as for `first_crossing`.

    Args:
        score: S(k).
        level: The score that counts as resolved.

    Returns:
        The wavelength, in the unit of 1/k, without the axis of the wavenumbers.
        NaN where the score never falls to the level, or is NaN before it does.
        A score that is below it at the first wavenumber gives the longest
        wavelength.

    Raises:
        BackendError: If the score is of DataArrays or of another library.
        ValidationError: If there are fewer than three wavenumbers.

    Examples:
        >>> import numpy as np
        >>> k = 0.01 * np.arange(6)
        >>> score = PSDScore(k, np.asarray([1.0, 1.0, 1.0, 0.6, 0.2, 0.0]))
        >>> round(float(resolved_scale(score)), 2)  # 1/k is 33 and 25 around it
        31.02
    """
    xp, (k, s) = floating_inputs(score.wavenumber, score.score)
    # 1 − S rises through 1 − level; k = 0 has no logarithm
    log_k = first_crossing(1.0 - s[..., 1:], xp.log(k[1:]), 1.0 - level)
    return xp.exp(-log_k)


def band_rmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    spacing: float | Sequence[float],
    bands: Sequence[tuple[float, float]],
    *,
    weights: ArrayLike | None = None,
) -> Any:
    """The root mean square error in bands of wavenumber.

    The error is filtered to the modes with k_min ≤ |k| < k_max, and its square
    averaged over the plane. Without weights, the squares of bands that cover
    every wavenumber once add up to the mean square error.

    The box is periodic, and nothing is tapered: a field that is not periodic has
    an error at its edges in every band.

    Args:
        prediction: ŷ, a real field that ends in (y, x), evenly spaced.
        reference: y, broadcast against ``prediction``.
        spacing: (Δy, Δx), or one number for both.
        bands: (k_min, k_max) of each, concrete (cycles per unit length); ∞ is
            allowed.
        weights: w ≥ 0 of each point, broadcast against the fields; 1 if not
            given.

    Returns:
        The error of each band, in the unit of the fields, with the bands in
        place of (y, x). A NaN in a field makes every band NaN.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If a band is empty, reversed or negative, ``spacing`` is
            not positive, or there are fewer than 2 points along an axis.

    Examples:
        >>> import numpy as np
        >>> y, x = np.mgrid[:16, :16]
        >>> error = 2.0 * np.cos(2.0 * np.pi * 5 * x / 16)  # k = 0.3125
        >>> bands = [(0.0, 0.2), (0.2, np.inf)]
        >>> np.round(band_rmse(error, 0.0 * error, 1.0, bands), 12) + 0.0
        array([0.        , 1.41421356])
    """
    steps = checked_spacing(spacing, 2)
    given = () if weights is None else (weights,)
    xp, (p, r, *w) = floating_inputs(prediction, reference, *given)
    p, r = xp.broadcast_arrays(p, r)
    shape = require_samples(p, 2)
    modes = _band_modes(bands, (shape[0], shape[1]), steps)
    z = xp.fft.rfftn(p - r, axes=(-2, -1))[..., None, :, :]
    kept = xp.where(table_like(modes, p, xp), z, xp.zeros_like(z))
    filtered = xp.fft.irfftn(kept, s=shape, axes=(-2, -1))
    used = xp.broadcast_to(w[0], p.shape) if w else xp.ones_like(p)
    cells = Cells(xp, used[..., None, :, :], (-2, -1))
    return root(xp, cells.mean(filtered**2))


def _band_modes(
    bands: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    steps: tuple[float, ...],
) -> np.ndarray:
    """Which modes of an ``rfftn`` are in each band, (band, Ny, Nx//2 + 1).

    Raises:
        ValidationError: If there is no band, or one is empty, reversed or
            negative.
    """
    limits = np.asarray(bands, dtype=np.float64)
    if limits.ndim != 2 or limits.shape[-1] != 2 or limits.shape[0] == 0:
        msg = f"'bands' must be pairs of wavenumbers, got {bands!r}"
        raise ValidationError(msg)
    if not ((limits[:, 0] >= 0.0) & (limits[:, 0] < limits[:, 1])).all():
        msg = f"a band must be 0 ≤ k_min < k_max, got {bands!r}"
        raise ValidationError(msg)
    (ny, nx), (dy, dx) = shape, steps
    ky, kx = np.fft.fftfreq(ny, dy)[:, None], np.fft.rfftfreq(nx, dx)
    size = np.hypot(ky, kx)
    modes = (size >= limits[:, :1, None]) & (size < limits[:, 1:, None])
    for band, some in zip(limits, modes.any(axis=(1, 2)), strict=True):
        if not some:
            msg = f"no wavenumber of this grid is in the band {tuple(band.tolist())}"
            raise ValidationError(msg)
    return modes
