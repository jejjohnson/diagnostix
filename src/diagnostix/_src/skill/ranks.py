"""Where a reference falls among the members, and probabilities against events."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.functions import boolean_like, divide_or_nan, floating_inputs
from pipekit_array.records import register_record

from diagnostix._src.reduce import constants_like, used_weights
from diagnostix._src.skill.members import ensemble_inputs
from diagnostix._src.skill.moments import Axis
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["Reliability", "coverage", "rank_histogram", "reliability"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Reliability:
    """What happened when an event was given a probability.

    Attributes:
        mean_probability: The mean probability of the cases in each bin.
        observed_frequency: The share of them in which the event happened; for
            calibrated probabilities it is the mean probability.
        weight: The sum of their weights, their number without weights.
    """

    mean_probability: Float[Any, " bin"]
    observed_frequency: Float[Any, " bin"]
    weight: Float[Any, " bin"]


def rank_histogram(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    member_axis: int = 0,
    valid: ArrayLike | None = None,
) -> Any:
    """How often the reference falls in each slot between the sorted members.

    Slot r holds the cases with r members below the reference. A reference that
    equals t members is shared evenly over the t + 1 slots around them.

    The histogram of a reliable ensemble is flat. High ends mean that it is too
    narrow, a high middle that it is too wide, and a slope that it is biased.

    Args:
        ensemble: x, the E members along ``member_axis``.
        reference: y, broadcast against the ensemble without that axis.
        member_axis: The axis of the members.
        valid: Which cases to count (True = count), broadcast. A NaN in a valid
            case makes the histogram NaN.

    Returns:
        The E + 1 counts, floating because of ties. Their sum is the number of
        valid cases.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        >>> rank_histogram(x, [0.0, 1.5, 2.0])
        array([1. , 1.5, 0.5])
    """
    xp, x, y, cells = ensemble_inputs(
        ensemble, reference, member_axis=member_axis, valid=valid
    )
    shares = _slot_shares(xp, x, y)
    counted = cells.used[..., None] * shares
    return xp.sum(xp.reshape(counted, (-1, counted.shape[-1])), axis=0, dtype=x.dtype)


def coverage(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    level: float = 0.9,
    member_axis: int = 0,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The share of references inside the central interval of an ensemble.

    The E + 1 slots between the sorted members hold the probabilities
    [r, r + 1] / (E + 1), and a reference in slot r counts by the share of them
    in the central interval [(1 − level)/2, (1 + level)/2].

    So the expected coverage of a reliable ensemble is the level, for any number
    of members. The share inside interpolated quantiles is less: at 0.9 it is
    0.62 for 5 members and 0.82 for 20.

    Args:
        ensemble: x, the E members along ``member_axis``.
        reference: y, broadcast against the ensemble without that axis.
        level: The probability of the interval, in (0, 1].
        member_axis: The axis of the members.
        weights: w ≥ 0 of each case, broadcast; 1 if not given.
        valid: Which cases to use (True = use), broadcast. A NaN in a valid case
            gives NaN.
        axis: The axis or axes of the cases to pool, of the ensemble without its
            members; every one if not given.

    Returns:
        The coverage, without the pooled axes. Its gradient is 0.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``level`` is not in (0, 1], or there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([0.0, 1.0, 2.0, 3.0])[:, None]
        >>> y = np.asarray([-1.0, 0.5, 1.5, 2.5, 9.0])  # one in each slot
        >>> round(float(coverage(x, y, level=0.6)), 12)
        0.6
        >>> round(float(coverage(x, y, level=0.9)), 12)
        0.9
    """
    if not 0.0 < level <= 1.0:
        msg = f"'level' must be in (0, 1], got {level}"
        raise ValidationError(msg)
    xp, x, y, cells = ensemble_inputs(
        ensemble,
        reference,
        member_axis=member_axis,
        weights=weights,
        valid=valid,
        axis=axis,
    )
    slots = x.shape[-1] + 1
    # the share of slot r's probabilities, [r, r + 1] / (E + 1), in the interval
    upper = np.minimum(0.5 * (1.0 + level) * slots, np.arange(1, slots + 1))
    lower = np.maximum(0.5 * (1.0 - level) * slots, np.arange(slots))
    inside = constants_like(xp, np.maximum(upper - lower, 0.0), x)
    return cells.mean(xp.sum(_slot_shares(xp, x, y) * inside, axis=-1))


def reliability(
    probability: ArrayLike,
    event: ArrayLike,
    *,
    edges: ArrayLike,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> Reliability:
    """How often an event happens, by the probability it was given.

    Args:
        probability: p of each case, in [0, 1].
        event: Whether it happened, boolean or 0 and 1, broadcast against
            ``probability``.
        edges: The edges of the bins, concrete and ascending. A bin holds its
            lower edge, and the last its upper one too. A case outside them is
            in no bin.
        weights: w ≥ 0 of each case, broadcast; 1 if not given.
        valid: Which cases to use (True = use), broadcast. A NaN probability in
            a valid case makes every bin NaN, and a NaN event its own.

    Returns:
        The bins, pooled over every case; the first two are NaN in an empty one.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``edges`` is not 1-D, finite and ascending.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> out = reliability(
        ...     [0.1, 0.2, 0.9, 1.0],
        ...     [False, True, True, True],
        ...     edges=[0.0, 0.5, 1.0],
        ... )
        >>> out.observed_frequency, out.weight
        (array([0.5, 1. ]), array([2., 2.]))
    """
    table = np.asarray(edges, dtype=np.float64)
    if table.ndim != 1 or table.size < 2 or not (np.diff(table) > 0.0).all():
        msg = f"'edges' must be 1-D and ascending, at least 2, got {edges!r}"
        raise ValidationError(msg)
    given = () if weights is None else (weights,)
    xp, (p, o, *w) = floating_inputs(probability, event, *given, params=(valid,))
    m = () if valid is None else (boolean_like(valid, p, xp),)
    p, o, *rest = xp.broadcast_arrays(p, o, *w, *m)
    mask = rest.pop() if m else None
    used = used_weights(xp, p, rest.pop() if w else None, mask)
    if mask is not None:
        p, o = (xp.where(mask, a, xp.zeros_like(a)) for a in (p, o))
    p, o, used = (xp.reshape(a, (-1, 1)) for a in (p, o, used))
    lower = constants_like(xp, table[:-1], p)
    upper = constants_like(xp, table[1:], p)
    last = constants_like(xp, np.arange(table.size - 1) == table.size - 2, p) > 0
    member = (p >= lower) & ((p < upper) | (last & (p == upper)))

    def total(values: Any) -> Any:
        kept = xp.where(member, values, xp.zeros_like(values))
        return xp.sum(kept, axis=0, dtype=p.dtype)

    weight = total(used)
    # a probability that is not a number is in no bin, or in any of them
    unknown = xp.sum(xp.where(xp.isnan(p), p, xp.zeros_like(p)))
    return Reliability(
        mean_probability=divide_or_nan(total(used * p), weight, xp) + unknown,
        observed_frequency=divide_or_nan(total(used * o), weight, xp) + unknown,
        weight=weight + unknown,
    )


def _slot_shares(xp: ModuleType, x: Any, y: Any) -> Any:
    """The share of each case in each of the E + 1 slots, (*case, slot).

    1 in one slot, or 1/(t + 1) in each of those around t members that equal the
    reference. NaN where a member or the reference is.
    """
    n = x.shape[-1]
    reference = y[..., None]

    def count(condition: Any) -> Any:
        ones = xp.astype(condition, x.dtype)
        return xp.sum(ones, axis=-1, keepdims=True, dtype=x.dtype)

    below, equal = count(x < reference), count(x == reference)
    slot = constants_like(xp, np.arange(n + 1.0), x)
    inside = (slot >= below) & (slot <= below + equal)
    share = 1.0 / (equal + 1.0)
    shares = xp.where(inside, share, xp.zeros_like(share))
    unknown = xp.any(xp.isnan(x), axis=-1, keepdims=True) | xp.isnan(reference)
    return xp.where(unknown, xp.full_like(shares, xp.nan), shares)
