"""Sums over a centred window along the last axis."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from diagnostix.exceptions import ValidationError


__all__ = ["check_window", "window_sum"]


def check_window(window: int, n: int, *, periodic: bool) -> None:
    """Raise unless a window is odd, positive, and no longer than a period.

    Raises:
        ValidationError: If it is not.
    """
    if window < 1 or window % 2 == 0:
        msg = f"'window' must be odd and positive, got {window}"
        raise ValidationError(msg)
    if periodic and window > n:
        msg = f"'window' must not exceed the period of {n}, got {window}"
        raise ValidationError(msg)


def window_sum(xp: ModuleType, x: Any, window: int, *, periodic: bool) -> Any:
    """The sum of the ``window`` samples centred on each one.

    Sums of 1, 2, 4, … neighbours are built from each other and added, about
    log₂ of the window's length of them. A difference of cumulative sums would
    lose float32's precision, and spread one NaN to every later window.

    Args:
        xp: The namespace.
        x: The samples, on the last axis.
        window: The length of the window, odd.
        periodic: Whether the window wraps. If not, it is truncated at the ends.

    Returns:
        The sums, the shape of ``x``.
    """
    n, half = x.shape[-1], window // 2
    if periodic:
        before, after = x[..., n - half :], x[..., :half]
    else:
        # zeros on x's device; a window may be longer than the series
        edge = xp.zeros_like(x[..., :1])
        before = after = xp.broadcast_to(edge, (*x.shape[:-1], half))
    # window k is the sum of padded[k : k + window]
    block = xp.concat([before, x, after], axis=-1)
    size, offset, left = 1, 0, window
    total = None
    while left:
        if left % 2:
            part = block[..., offset : offset + n]
            total = part if total is None else total + part
            offset += size
        left //= 2
        if left:
            # the sums of 2 · size neighbours, from those of size
            block = block[..., :-size] + block[..., size:]
            size *= 2
    return total
