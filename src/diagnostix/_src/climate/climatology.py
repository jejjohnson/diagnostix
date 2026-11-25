"""Climatologies: the mean of each slot of the year, fitted or smoothed."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from pipekit_array.functions import divide_or_nan, floating_inputs, table_like
from pipekit_array.records import register_record
from pipekit_array.weighted import NO_GROUP, Groups

from diagnostix._src.climate.calendar import checked_slots
from diagnostix._src.climate.window import check_window, window_sum
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "Climatology",
    "climatology",
    "harmonic_climatology",
    "smooth_climatology",
]


@dataclasses.dataclass(frozen=True, eq=False)
class Climatology:
    """The mean of each slot of the year, on the axis the samples were on.

    Attributes:
        mean: x̄(d), NaN in a slot with no valid sample.
        count: n(d), the number of valid samples of each slot, int32.
        n_slots: The number of slots: 366 days or 12 months. Static under JAX.
    """

    mean: Float[Any, "..."]
    count: Int[Any, "..."]
    n_slots: int


register_record(Climatology, meta_fields=("n_slots",))


def climatology(
    x: ArrayLike,
    slot: ArrayLike,
    *,
    n_slots: int = 366,
    axis: int = -1,
    valid: ArrayLike | None = None,
    base: ArrayLike | None = None,
) -> Climatology:
    """The mean of each slot of the year over the base period.

    x̄(d) = (1 / n_d) ∑ₜ x(t) over the valid samples t of slot d.

    Args:
        x: Samples of any unit, along ``axis``.
        slot: The slot of each sample, 1 … ``n_slots``, concrete integers:
            `calendar_day` for a daily climatology, the month with
            ``n_slots=12``, or the caller's own labels (seasons, weekdays, the
            days of a 360-day calendar).
        n_slots: The number of slots.
        axis: The axis of the samples.
        valid: Which samples to use (True = use), broadcast against ``x``. A NaN
            in a valid sample gives NaN.
        base: Which samples are in the base period, concrete booleans, one per
            sample; all if not given. A climatology fitted on the period it is
            later scored on leaks into the score.

    Returns:
        The mean and the count of each slot, on ``axis``. An empty slot is NaN.

    Raises:
        BackendError: If ``x`` or ``valid`` is a DataArray or of another library.
        ValidationError: If a slot is not in 1 … ``n_slots``, or ``base`` is not
            boolean.
        ValueError: If ``axis`` does not hold one sample per slot, or ``valid`` is
            not boolean.

    Examples:
        >>> import numpy as np
        >>> month = np.asarray([1, 2, 1, 2, 1])
        >>> x = np.asarray([1.0, 5.0, 3.0, 7.0, 100.0])
        >>> out = climatology(x, month, n_slots=3, base=month * 0 + x < 50.0)
        >>> out.mean, out.count
        (array([ 2.,  6., nan]), array([2, 2, 0], dtype=int32))
    """
    groups = _groups(slot, n_slots, base)
    return Climatology(
        mean=groups.mean(x, axis=axis, valid=valid),
        count=groups.count(x, axis=axis, valid=valid),
        n_slots=n_slots,
    )


def harmonic_climatology(
    x: ArrayLike,
    slot: ArrayLike,
    *,
    n_slots: int = 366,
    n_harmonics: int = 3,
    axis: int = -1,
    valid: ArrayLike | None = None,
    base: ArrayLike | None = None,
) -> Climatology:
    """The annual cycle as a mean and its first harmonics, defined in every slot.

    x̃(d) = a₀ + ∑ₖ aₖ cos(2π k d / D) + bₖ sin(2π k d / D), with D the number of
    slots and k = 1 … K, fitted to the slot means by least squares weighted by
    their counts. dask has no batched solve, so compute a dask array first if there
    is a ``valid``.

    Args:
        x: Samples of any unit, along ``axis``.
        slot: The slot of each sample, as in `climatology`.
        n_slots: D, the number of slots and the period of the cycle.
        n_harmonics: K. 2K + 1 slots must have data.
        axis: The axis of the samples.
        valid: Which samples to use, as in `climatology`.
        base: Which samples are in the base period, as in `climatology`.

    Returns:
        The fit in every slot, empty ones included, and the count of each slot.
        NaN where a point has fewer than 2K + 1 slots with data.

    Raises:
        BackendError: If ``x`` or ``valid`` is a DataArray or of another library.
        ValidationError: As `climatology`, or if 2K + 1 exceeds ``n_slots``, or
            the base period has samples in fewer than 2K + 1 slots.

    Examples:
        >>> import numpy as np
        >>> slot = np.asarray([1, 2, 3, 5, 6, 8])
        >>> x = 4.0 + np.cos(2.0 * np.pi * slot / 8.0)
        >>> out = harmonic_climatology(x, slot, n_slots=8, n_harmonics=1)
        >>> np.round(out.mean[[1, 3, 7]], 9)
        array([4., 3., 5.])
        >>> out.count
        array([1, 1, 1, 0, 1, 1, 0, 1], dtype=int32)
    """
    n_terms = 2 * n_harmonics + 1
    if n_harmonics < 0 or n_terms > n_slots:
        msg = (
            f"'n_harmonics' must be at least 0 and 2K + 1 at most {n_slots}, got "
            f"{n_harmonics}"
        )
        raise ValidationError(msg)
    groups = _groups(slot, n_slots, base)
    sampled = np.count_nonzero(groups.member.any(axis=-1))
    if sampled < n_terms:
        msg = f"{n_terms} slots need data for {n_harmonics} harmonics, got {sampled}"
        raise ValidationError(msg)
    basis = _harmonics(n_slots, n_harmonics)
    count = groups.count(x, axis=axis, valid=valid)
    # the weighted normal equations need n_d x̄(d), which is the sum of the slot
    total = groups.sum(x, axis=axis, valid=valid)
    xp, (total,) = floating_inputs(total)
    last = xp.moveaxis(total, axis, -1)
    if valid is None:
        fit = _fit_shared(xp, last, basis, groups.member.sum(axis=-1))
    else:
        weights = xp.astype(xp.moveaxis(count, axis, -1), last.dtype)
        fit = _fit_each(xp, last, basis, weights)
    return Climatology(mean=xp.moveaxis(fit, -1, axis), count=count, n_slots=n_slots)


def smooth_climatology(
    climatology: Climatology, window: int, *, axis: int = -1
) -> Climatology:
    """A climatology smoothed by a running mean that wraps around the year.

    x̃(d) = ∑_{|s| ≤ h} n_{d+s} x̄(d + s) / ∑_{|s| ≤ h} n_{d+s}, with h the half
    of the window, so a slot with few samples counts for little.

    Args:
        climatology: The climatology.
        window: The length of the window in slots, odd, so that it is centred.
        axis: The axis of the slots.

    Returns:
        The smoothed mean, NaN where a whole window is empty, with the counts of
        the slots.

    Raises:
        ValidationError: If the window is not odd or exceeds the year, or
            ``axis`` does not hold the slots.

    Examples:
        >>> import numpy as np
        >>> out = climatology(
        ...     [4.0, 1.0, np.nan, 7.0],
        ...     [1, 2, 3, 4],
        ...     n_slots=4,
        ...     valid=[True, True, False, True],
        ... )
        >>> smooth_climatology(out, 3).mean
        array([4. , 2.5, 4. , 5.5])
    """
    n_slots = climatology.n_slots
    check_window(window, n_slots, periodic=True)
    xp, (mean,) = floating_inputs(climatology.mean)
    check_slots_axis(mean, n_slots, axis)
    mean = xp.moveaxis(mean, axis, -1)
    count = xp.moveaxis(climatology.count, axis, -1)
    weights = xp.astype(count, mean.dtype)
    # an empty slot is NaN, and its weight is 0
    total = xp.where(count > 0, weights * mean, xp.zeros_like(mean))
    smooth = divide_or_nan(
        window_sum(xp, total, window, periodic=True),
        window_sum(xp, weights, window, periodic=True),
        xp,
    )
    return Climatology(
        mean=xp.moveaxis(smooth, -1, axis),
        count=climatology.count,
        n_slots=n_slots,
    )


def check_slots_axis(values: Any, n_slots: int, axis: int) -> None:
    """Raise unless an axis holds the slots of a climatology.

    Raises:
        ValidationError: If it does not.
    """
    if not -values.ndim <= axis < values.ndim or values.shape[axis] != n_slots:
        msg = f"axis {axis} of shape {values.shape} must hold the {n_slots} slots"
        raise ValidationError(msg)


def _groups(slot: ArrayLike, n_slots: int, base: ArrayLike | None) -> Groups:
    """The samples of each slot that are in the base period."""
    if n_slots < 1:
        msg = f"'n_slots' must be at least 1, got {n_slots}"
        raise ValidationError(msg)
    labels = checked_slots(slot, n_slots) - 1
    if base is not None:
        chosen = np.asarray(base)
        if chosen.dtype != np.bool_ or chosen.shape != labels.shape:
            msg = (
                f"'base' must be boolean, one per sample of {labels.shape}, got "
                f"{chosen.dtype} of shape {chosen.shape}"
            )
            raise ValidationError(msg)
        labels = np.where(chosen, labels, NO_GROUP)
    return Groups.from_labels(labels, n_slots)


def _harmonics(n_slots: int, n_harmonics: int) -> np.ndarray:
    """B, (slots, 2K + 1): 1, and cos and sin of each harmonic at d = 1 … D."""
    day = np.arange(1, n_slots + 1)
    angles = 2.0 * np.pi * np.outer(day, np.arange(1, n_harmonics + 1)) / n_slots
    ones = np.ones((n_slots, 1))
    return np.concatenate([ones, np.cos(angles), np.sin(angles)], axis=-1)


def _fit_shared(xp: ModuleType, total: Any, basis: np.ndarray, count: Any) -> Any:
    """The fit where every point has the same counts: one matrix, on the host."""
    # A = Bᵀ N B, with N = diag(n_d)
    normal = basis.T @ (count[:, None] * basis)
    # a = A⁻¹ Bᵀ s from the sums s, as s (B A⁻¹), because A is symmetric
    solve = np.linalg.solve(normal, basis.T).T
    tables = (table_like(table, total, xp) for table in (solve, basis.T))
    to_terms, to_slots = (xp.astype(table, total.dtype) for table in tables)
    return xp.matmul(xp.matmul(total, to_terms), to_slots)


def _fit_each(xp: ModuleType, total: Any, basis: np.ndarray, count: Any) -> Any:
    """The fit where the counts differ: one set of normal equations per point."""
    n_terms = basis.shape[-1]
    table = xp.astype(table_like(basis, total, xp), total.dtype)
    outer = basis[:, :, None] * basis[:, None, :]
    outer = outer.reshape(basis.shape[0], -1)
    outer = xp.astype(table_like(outer, total, xp), total.dtype)
    # Bᵀ N B / ∑ n, so that its entries are of order 1
    samples = xp.sum(count, axis=-1, keepdims=True, dtype=count.dtype)
    scale = xp.where(samples > 0, samples, xp.ones_like(samples))
    normal = xp.matmul(count / scale, outer)
    normal = xp.reshape(normal, (*count.shape[:-1], n_terms, n_terms))
    sums = xp.matmul(total / scale, table)
    # trigonometric polynomials of 2K + 1 terms are independent on 2K + 1 slots
    sampled = xp.sum(xp.astype(count > 0, xp.int32), axis=-1, dtype=xp.int32)
    enough = sampled >= n_terms
    identity = xp.astype(table_like(np.eye(n_terms), total, xp), total.dtype)
    safe = xp.where(enough[..., None, None], normal, identity)
    coefficients = xp.linalg.solve(safe, sums[..., None])[..., 0]
    fit = xp.matmul(coefficients, xp.matrix_transpose(table))
    return xp.where(enough[..., None], fit, xp.full_like(fit, xp.nan))
