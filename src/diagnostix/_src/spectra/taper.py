"""What every transform starts with: the spacing, the trend and the taper."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, Literal

import numpy as np

from diagnostix._src.reduce import constants_like
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "Detrend",
    "Window",
    "checked_spacing",
    "demean_or_detrend",
    "one_sided",
    "require_samples",
    "squared_magnitude",
    "taper",
    "tapered",
]

type Window = Literal["hann"] | ArrayLike | None
type Detrend = Literal["constant", "linear"] | None


def checked_spacing(spacing: float | Sequence[float], n_axes: int) -> tuple[float, ...]:
    """The spacing of each transformed axis, in metres or any other unit.

    Raises:
        ValidationError: If there is neither one nor one per axis, or one is not
            positive and finite.
    """
    steps = np.asarray(spacing, dtype=np.float64)
    if steps.shape not in ((), (n_axes,)):
        msg = f"'spacing' must be one number or {n_axes}, got {spacing!r}"
        raise ValidationError(msg)
    if not (np.isfinite(steps).all() and (steps > 0.0).all()):
        msg = f"'spacing' must be positive and finite, got {spacing!r}"
        raise ValidationError(msg)
    return tuple(float(step) for step in np.broadcast_to(steps, (n_axes,)))


def require_samples(x: Any, n_axes: int) -> tuple[int, ...]:
    """The sizes of the last axes of an array, each at least 2.

    Raises:
        ValidationError: If the array has fewer axes, or one is shorter.
    """
    shape = tuple(x.shape[-n_axes:])
    if x.ndim < n_axes or min(shape) < 2:
        msg = (
            f"a transform needs at least 2 samples along each of {n_axes} axes, "
            f"got shape {tuple(x.shape)}"
        )
        raise ValidationError(msg)
    return shape


def taper(window: Window, shape: tuple[int, ...]) -> tuple[np.ndarray | None, float]:
    """The taper over the transformed axes, and its mean square U.

    Args:
        window: ``"hann"``, periodic and separable; a host array of ``shape``;
            or ``None`` for none.
        shape: The sizes of the transformed axes.

    Returns:
        w, or ``None`` where it is 1 everywhere, and U.

    Raises:
        ValidationError: If the window is another name, not of ``shape``, or 0
            everywhere.
    """
    if window is None:
        return None, 1.0
    if isinstance(window, str):
        if window != "hann":
            msg = f"'window' must be 'hann', None or an array, got {window!r}"
            raise ValidationError(msg)
        w = np.ones(shape)
        for axis, n in enumerate(shape):
            # wₙ = ½ (1 − cos(2π n / N)): a whole period, without its last point
            one = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
            w *= np.expand_dims(one, tuple(a for a in range(len(shape)) if a != axis))
    else:
        w = np.asarray(window, dtype=np.float64)
    mean_square = float(np.mean(w**2)) if w.shape == shape else 0.0
    if not (np.isfinite(mean_square) and mean_square > 0.0):
        msg = (
            f"a window must be finite, of shape {shape} and not 0 everywhere, "
            f"got shape {w.shape}"
        )
        raise ValidationError(msg)
    return w, mean_square


def demean_or_detrend(xp: ModuleType, x: Any, n_axes: int, detrend: Detrend) -> Any:
    """Samples without their mean, or without their line or plane.

    Args:
        xp: The namespace.
        x: The samples, on the last ``n_axes`` axes.
        n_axes: How many axes are transformed.
        detrend: ``"constant"``, ``"linear"`` or ``None``.

    Raises:
        ValidationError: If ``detrend`` is none of them.
    """
    if detrend not in (None, "constant", "linear"):
        msg = f"'detrend' must be 'linear', 'constant' or None, got {detrend!r}"
        raise ValidationError(msg)
    if detrend is None:
        return x
    axes = tuple(range(-n_axes, 0))
    x = x - xp.mean(x, axis=axes, keepdims=True)
    if detrend == "constant":
        return x
    fitted = xp.zeros_like(x)
    for axis in axes:
        n = x.shape[axis]
        # s, the offset from the middle, orthogonal to 1 and to the other axes'
        s = np.arange(n) - 0.5 * (n - 1)
        offset = constants_like(xp, s.reshape((n,) + (1,) * (-axis - 1)), x)
        # β = ⟨s x⟩ / ⟨s²⟩
        slope = xp.mean(offset * x, axis=axes, keepdims=True) / float(np.mean(s**2))
        fitted = fitted + slope * offset
    return x - fitted


def tapered(
    xp: ModuleType, x: Any, n_axes: int, window: Window, detrend: Detrend
) -> tuple[Any, float]:
    """Samples detrended and then tapered, and the taper's mean square U.

    Raises:
        ValidationError: As `require_samples`, `taper` and `demean_or_detrend`.
    """
    w, mean_square = taper(window, require_samples(x, n_axes))
    x = demean_or_detrend(xp, x, n_axes, detrend)
    return (x if w is None else x * constants_like(xp, w, x)), mean_square


def one_sided(n: int) -> np.ndarray:
    """How many modes each bin of an ``rfft`` of n samples stands for: 1 or 2."""
    modes = np.full(n // 2 + 1, 2.0)
    modes[0] = 1.0
    if n % 2 == 0:
        modes[-1] = 1.0
    return modes


def squared_magnitude(xp: ModuleType, z: Any) -> Any:
    """|z|², with a gradient at 0, which that of ``abs`` does not have."""
    return xp.real(z) ** 2 + xp.imag(z) ** 2
