"""Weighted moments of a prediction and its reference, over the same cells."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

from pipekit_array.functions import boolean_like, divide_or_nan, floating_inputs
from pipekit_array.weighted import used_weights

from diagnostix._src.reduce import require_axis
from diagnostix._typing import ArrayLike


__all__ = ["Axis", "Cells", "paired_inputs", "ratio_or_inf", "root"]

type Axis = int | tuple[int, ...] | None


@dataclasses.dataclass(frozen=True)
class Cells:
    """Which cells count in a score, and by how much.

    Attributes:
        xp: The namespace.
        used: m w of each cell, 0 in an invalid one.
        axis: The axis or axes that are pooled.
    """

    xp: ModuleType
    used: Any
    axis: Axis

    def mean(self, values: Any, *, keepdims: bool = False) -> Any:
        """∑ m w x / ∑ m w, NaN where the weights used sum to 0."""
        xp = self.xp
        total = xp.sum(
            self.used * values, axis=self.axis, keepdims=keepdims, dtype=values.dtype
        )
        weight = xp.sum(
            self.used, axis=self.axis, keepdims=keepdims, dtype=values.dtype
        )
        return divide_or_nan(total, weight, xp)

    def variance(self, values: Any) -> Any:
        """The mean square of x − x̄, and 0 where x is constant.

        The variance of one value is otherwise the rounding of its mean.
        """
        xp = self.xp
        spread = self.mean((values - self.mean(values, keepdims=True)) ** 2)
        counted = self.used > 0
        infinite = xp.full_like(values, xp.inf)
        largest = xp.max(xp.where(counted, values, -infinite), axis=self.axis)
        smallest = xp.min(xp.where(counted, values, infinite), axis=self.axis)
        varies = (largest > smallest) | xp.isnan(spread)
        return xp.where(varies, spread, xp.zeros_like(spread))


def paired_inputs(
    prediction: ArrayLike,
    reference: ArrayLike,
    weights: ArrayLike | None,
    valid: ArrayLike | None,
    axis: Axis,
) -> tuple[ModuleType, Any, Any, Cells]:
    """The namespace, both fields broadcast, 0 in an invalid cell, and the cells.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.
    """
    given = () if weights is None else (weights,)
    xp, (p, r, *w) = floating_inputs(prediction, reference, *given, params=(valid,))
    m = () if valid is None else (boolean_like(valid, p, xp),)
    p, r, *rest = xp.broadcast_arrays(p, r, *w, *m)
    mask = rest.pop() if m else None
    weight = rest.pop() if w else None
    for one in () if axis is None else (axis,) if isinstance(axis, int) else axis:
        require_axis(p, one)
    if mask is not None:
        # before any product: 0 times the derivative of a NaN is NaN
        p, r = (xp.where(mask, a, xp.zeros_like(a)) for a in (p, r))
    return xp, p, r, Cells(xp, used_weights(xp, p, weight, mask), axis)


def root(xp: ModuleType, square: Any) -> Any:
    """The square root, with a derivative of 0 at 0 in place of an infinite one."""
    positive = square > 0
    safe = xp.sqrt(xp.where(positive, square, xp.ones_like(square)))
    return xp.where(positive, safe, square)


def ratio_or_inf(xp: ModuleType, numerator: Any, denominator: Any) -> Any:
    """A ratio of numbers ≥ 0: 0 for 0 / 0, and ∞ for anything else over 0."""
    positive = denominator > 0
    safe = xp.where(positive, denominator, xp.ones_like(denominator))
    # 0, or NaN, stays itself
    over_zero = xp.where(numerator > 0, xp.full_like(numerator, xp.inf), numerator)
    return xp.where(positive, numerator / safe, over_zero)
