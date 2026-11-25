"""Scores of an ensemble: CRPS, energy score, spread and skill."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.arrays import constants_like, divide_or_nan, register_record

from diagnostix._src.skill.members import axes_of, ensemble_inputs, require_members
from diagnostix._src.skill.moments import Axis, root
from diagnostix._typing import ArrayLike


__all__ = ["SpreadSkill", "crps", "energy_score", "spread_skill"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class SpreadSkill:
    """The spread of an ensemble beside the error of its mean.

    Attributes:
        spread: The root of the mean variance of the members, unbiased (E − 1).
        skill: The root mean square error of the ensemble mean.
        ratio: ((E + 1) / E)^½ spread / skill: 1 for a reliable ensemble of any
            size, and below 1 for one that is too narrow. NaN where the skill
            is 0.
    """

    spread: Float[Any, "..."]
    skill: Float[Any, "..."]
    ratio: Float[Any, "..."]


def crps(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    member_axis: int = 0,
    fair: bool = True,
) -> Any:
    """The continuous ranked probability score of each case.

    (1/E) ∑ₑ |xₑ − y| − ∑ₑₑ′ |xₑ − xₑ′| / (2 E (E − 1)), with E members x. The sum
    over pairs is 2 ∑ᵢ (2i − E − 1) x₍ᵢ₎ over the sorted members.

    The fair score does not depend on the number of members: that of a reliable
    ensemble of 2 is, on average, that of its distribution. With E² in place of
    E (E − 1) it is the score of the members as a distribution, larger for a small
    ensemble.

    Args:
        ensemble: x, the members along ``member_axis``.
        reference: y, broadcast against the ensemble without that axis.
        member_axis: The axis of the members.
        fair: Whether to divide by E (E − 1), or by E². One member and
            ``fair=False`` give the absolute error.

    Returns:
        The score of each case, in the unit of the fields, 0 for perfect. NaN
        where a member or the reference is.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis, or ``fair`` and one member.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 2.0], [3.0, 2.0]])
        >>> crps(x, [2.0, 5.0])
        array([0., 3.])
        >>> crps(x, [2.0, 5.0], fair=False)
        array([0.5, 3. ])
    """
    xp, x, y, _ = ensemble_inputs(ensemble, reference, member_axis=member_axis)
    n = x.shape[-1]
    _require_fair(n, fair)
    # from y, so that float32 keeps the differences of warm temperatures
    d = x - y[..., None]
    # ∑ₑₑ′ |xₑ − xₑ′| = 2 ∑ᵢ (2i − E − 1) x₍ᵢ₎
    factor = constants_like(2.0 * np.arange(1, n + 1) - n - 1.0, x, xp)
    half_pairs = xp.sum(factor * xp.sort(d, axis=-1), axis=-1)
    return xp.mean(xp.abs(d), axis=-1) - half_pairs / _pairs(n, fair)


def energy_score(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    axis: int | tuple[int, ...],
    member_axis: int = 0,
    weights: ArrayLike | None = None,
    fair: bool = True,
) -> Any:
    """The energy score: the CRPS of vectors, or of whole fields.

    (1/E) ∑ₑ ‖xₑ − y‖ − ∑ₑₑ′ ‖xₑ − xₑ′‖ / (2 E (E − 1)), with ‖d‖ = (∑ w d²)^½ over
    ``axis``. Unlike the mean of the CRPS over the axis, it sees whether the
    elements vary together.

    Args:
        ensemble: x, the members along ``member_axis``.
        reference: y, broadcast against the ensemble without that axis.
        axis: The axis or axes of a vector, of the ensemble without its members.
        member_axis: The axis of the members.
        weights: w ≥ 0 of each element, broadcast against the cases; 1 if not
            given.
        fair: Whether to divide by E (E − 1), as `crps`, or by E².

    Returns:
        The score, the cases without ``axis``, 0 for perfect. NaN where an
        element of a member or of the reference is.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis, or ``fair`` and one member.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[3.0, 4.0], [0.0, 0.0]])
        >>> float(energy_score(x, [0.0, 0.0], axis=-1))  # 5/2 − (5 + 5)/4
        0.0
        >>> float(energy_score(x, [0.0, 0.0], axis=-1, fair=False))
        1.25
    """
    xp, x, y, cells = ensemble_inputs(
        ensemble, reference, member_axis=member_axis, weights=weights
    )
    n = x.shape[-1]
    _require_fair(n, fair)
    axes = axes_of(y, axis)
    w = cells.used[..., None]

    def norms(d: Any) -> Any:
        """The sum over the members of ‖d‖."""
        return xp.sum(root(xp, xp.sum(w * d**2, axis=axes)), axis=-1)

    # every pair is one of E − 1 cyclic shifts, and shift s is shift E − s
    pairs = xp.zeros_like(norms(x))
    for shift in range(1, n // 2 + 1):
        times = 1.0 if 2 * shift == n else 2.0
        pairs = pairs + times * norms(x - xp.roll(x, shift, axis=-1))
    return norms(x - y[..., None]) / n - pairs / (2.0 * _pairs(n, fair))


def spread_skill(
    ensemble: ArrayLike,
    reference: ArrayLike,
    *,
    member_axis: int = 0,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> SpreadSkill:
    """The spread of an ensemble, the error of its mean, and their ratio.

    spread = ⟨Varₑ(x)⟩^½ and skill = ⟨(x̄ − y)²⟩^½, with ⟨·⟩ the weighted mean over
    the valid cases. The mean of E members has an error of its own, so a reliable
    ensemble has a skill of ((E + 1) / E)^½ times its spread, and `ratio` is 1.

    Args:
        ensemble: x, the members along ``member_axis``.
        reference: y, broadcast against the ensemble without that axis.
        member_axis: The axis of the members.
        weights: w ≥ 0 of each case, broadcast; 1 if not given.
        valid: Which cases to use (True = use), broadcast. A NaN in a valid case
            gives NaN.
        axis: The axis or axes of the cases to pool, of the ensemble without its
            members; every one if not given.

    Returns:
        The three, without the pooled axes.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis, or one member.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[0.0, 4.0], [2.0, 6.0]])
        >>> out = spread_skill(x, [3.0, 3.0])
        >>> float(out.skill)
        2.0
        >>> round(float(out.ratio), 12)  # (3/2)^½ 2^½ / 2
        0.866025403784
    """
    xp, x, y, cells = ensemble_inputs(
        ensemble,
        reference,
        member_axis=member_axis,
        weights=weights,
        valid=valid,
        axis=axis,
    )
    n = x.shape[-1]
    require_members(n, 2, "a spread")
    mean = xp.mean(x, axis=-1)
    variance = xp.sum((x - mean[..., None]) ** 2, axis=-1) / (n - 1)
    spread = root(xp, cells.mean(variance))
    skill = root(xp, cells.mean((mean - y) ** 2))
    ratio = divide_or_nan(spread, skill, xp) * math.sqrt((n + 1) / n)
    return SpreadSkill(spread=spread, skill=skill, ratio=ratio)


def _require_fair(n_members: int, fair: bool) -> None:
    """Raise unless there are members enough for a score."""
    require_members(n_members, 2 if fair else 1, "a fair score" if fair else "a score")


def _pairs(n_members: int, fair: bool) -> float:
    """E (E − 1), or E²."""
    return float(n_members * (n_members - 1 if fair else n_members))
