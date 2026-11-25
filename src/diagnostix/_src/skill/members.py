"""The inputs of an ensemble score: members last, and the cases that count."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from pipekit_array.functions import boolean_like, floating_inputs
from pipekit_array.weighted import used_weights

from diagnostix._src.reduce import require_axis
from diagnostix._src.skill.moments import Axis, Cells
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["axes_of", "ensemble_inputs", "require_members"]


def ensemble_inputs(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    member_axis: int,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> tuple[ModuleType, Any, Any, Cells]:
    """The namespace, the members and the reference of each case, and the cells.

    The cases are the axes of the ensemble without its members, broadcast against
    the reference, the weights and the mask.

    Returns:
        x of shape (*case, member), y of shape (*case), both 0 in an invalid
        case, and the cells.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.
    """
    given = () if weights is None else (weights,)
    xp, (x, y, *w) = floating_inputs(ensemble, reference, *given, params=(valid,))
    require_axis(x, member_axis)
    x = xp.moveaxis(x, member_axis, -1)
    require_members(x.shape[-1], 1, "an ensemble")
    m = () if valid is None else (boolean_like(valid, x, xp),)
    cases = xp.broadcast_arrays(x[..., 0], y, *w, *m)
    _, y, *rest = cases
    x = xp.broadcast_to(x, (*y.shape, x.shape[-1]))
    mask = rest.pop() if m else None
    weight = rest.pop() if w else None
    axes_of(y, axis)
    if mask is not None:
        # before any product: 0 times the derivative of a NaN is NaN
        x = xp.where(mask[..., None], x, xp.zeros_like(x))
        y = xp.where(mask, y, xp.zeros_like(y))
    return xp, x, y, Cells(xp, used_weights(xp, y, weight, mask), axis)


def axes_of(y: Any, axis: Axis) -> tuple[int, ...]:
    """The axes of the cases that an argument names, none if it is ``None``.

    Raises:
        ValidationError: If there is no such axis.
    """
    named = () if axis is None else (axis,) if isinstance(axis, int) else tuple(axis)
    for one in named:
        require_axis(y, one)
    return tuple(one % y.ndim for one in named)


def require_members(n_members: int, at_least: int, what: str) -> None:
    """Raise unless an ensemble has enough members.

    Raises:
        ValidationError: If it has not.
    """
    if n_members < at_least:
        msg = f"{what} needs at least {at_least} members, got {n_members}"
        raise ValidationError(msg)
