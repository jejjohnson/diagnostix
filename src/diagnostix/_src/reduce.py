"""The check of an axis."""

from __future__ import annotations

from typing import Any

from diagnostix.exceptions import ValidationError


__all__ = ["require_axis"]


def require_axis(values: Any, axis: int) -> None:
    """Raise unless an array has an axis.

    Raises:
        ValidationError: If it does not.
    """
    if not -values.ndim <= axis < values.ndim:
        msg = f"no axis {axis} for the samples of an array of shape {values.shape}"
        raise ValidationError(msg)
