"""Pointwise scores of a prediction against its reference."""

from __future__ import annotations

from typing import Any, Literal

from diagnostix._src.skill.moments import Axis, paired_inputs, ratio_or_inf, root
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["bias", "correlation", "mae", "nrmse_score", "r2_score", "rmse"]


def rmse(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The root mean square error.

    (∑ m w (ŷ − y)² / ∑ m w)^½, with m the validity, 1 or 0.

    Its gradient at a perfect prediction is 0.

    Args:
        prediction: ŷ, of any shape and unit.
        reference: y, broadcast against ``prediction``.
        weights: w ≥ 0, broadcast against both; 1 if not given. Cell areas come
            from ``gridax.area_weights``.
        valid: Which cells to use (True = use), broadcast against both. A NaN in
            a valid cell gives NaN.
        axis: The axis or axes to pool; every one if not given. The score over
            space and time is not the mean over time of the scores over space.

    Returns:
        The score, in the unit of the fields, NaN where no weight counts.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> y = np.asarray([1.0, 2.0, np.nan])
        >>> score = rmse([1.0, 5.0, 3.0], y, weights=[8, 1, 1], valid=~np.isnan(y))
        >>> round(float(score), 12)  # (9 / 9)^½
        1.0
    """
    xp, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    return root(xp, cells.mean((p - r) ** 2))


def mae(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The mean absolute error.

    ∑ m w |ŷ − y| / ∑ m w, with m the validity, 1 or 0.

    Args:
        prediction: ŷ, as in `rmse`.
        reference: y, as in `rmse`.
        weights: w, as in `rmse`.
        valid: Which cells to use, as in `rmse`.
        axis: The axis or axes to pool, as in `rmse`.

    Returns:
        The score, in the unit of the fields, NaN where no weight counts.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> float(mae([1.0, 5.0], [2.0, 2.0]))
        2.0
    """
    xp, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    return cells.mean(xp.abs(p - r))


def bias(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The mean error, positive where the prediction is too large.

    ∑ m w (ŷ − y) / ∑ m w, with m the validity, 1 or 0.

    Args:
        prediction: ŷ, as in `rmse`.
        reference: y, as in `rmse`.
        weights: w, as in `rmse`.
        valid: Which cells to use, as in `rmse`.
        axis: The axis or axes to pool, as in `rmse`.

    Returns:
        The score, in the unit of the fields, NaN where no weight counts.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> float(bias([1.0, 5.0], [2.0, 2.0]))
        1.0
    """
    _, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    return cells.mean(p - r)


def correlation(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The linear correlation of a prediction and its reference.

    ⟨(ŷ − ⟨ŷ⟩)(y − ⟨y⟩)⟩ / (⟨(ŷ − ⟨ŷ⟩)²⟩ ⟨(y − ⟨y⟩)²⟩)^½, with ⟨·⟩ the weighted mean
    over the same valid cells of both.

    Args:
        prediction: ŷ, as in `rmse`.
        reference: y, as in `rmse`.
        weights: w, as in `rmse`.
        valid: Which cells to use, as in `rmse`.
        axis: The axis or axes to pool, as in `rmse`.

    Returns:
        The correlation, NaN where either field is constant.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> round(float(correlation([1.0, 2.0, 4.0], [5.0, 4.0, 2.0])), 12)
        -1.0
    """
    xp, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    variance_p, variance_r = cells.variance(p), cells.variance(r)
    defined = (variance_p > 0) & (variance_r > 0)
    one = xp.ones_like(variance_p)
    # a root of each, because the product of two small variances underflows
    scale_p = xp.sqrt(xp.where(defined, variance_p, one))
    scale_r = xp.sqrt(xp.where(defined, variance_r, one))
    p = p - cells.mean(p, keepdims=True)
    r = r - cells.mean(r, keepdims=True)
    value = cells.mean(p * r) / (scale_p * scale_r)
    return xp.where(defined, value, xp.full_like(value, xp.nan))


def r2_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """The share of the reference's variance that the prediction explains.

    R² = 1 − ⟨(ŷ − y)²⟩ / ⟨(y − ⟨y⟩)²⟩, with ⟨·⟩ the weighted mean over the same
    valid cells. 1 is perfect, and 0 is as good as the mean of the reference.

    Args:
        prediction: ŷ, as in `rmse`.
        reference: y, as in `rmse`.
        weights: w, as in `rmse`.
        valid: Which cells to use, as in `rmse`.
        axis: The axis or axes to pool, as in `rmse`.

    Returns:
        R². Where the reference is constant, 1 for a perfect prediction and −∞
        otherwise.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> round(float(r2_score([1.0, 2.0, 6.0], [1.0, 2.0, 3.0])), 12)
        -3.5
    """
    xp, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    return 1.0 - ratio_or_inf(xp, cells.mean((p - r) ** 2), cells.variance(r))


def nrmse_score(
    prediction: ArrayLike,
    reference: ArrayLike,
    *,
    normalise: Literal["std", "rms"] = "std",
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    axis: Axis = None,
) -> Any:
    """1 minus the root mean square error over the size of the reference.

    1 − RMSE / σ, with σ the standard deviation of the reference, or its root
    mean square, over the same valid cells. 1 is perfect.

    Args:
        prediction: ŷ, as in `rmse`.
        reference: y, as in `rmse`.
        normalise: ``"std"``, or ``"rms"``, which counts the mean of the
            reference as signal.
        weights: w, as in `rmse`.
        valid: Which cells to use, as in `rmse`.
        axis: The axis or axes to pool, as in `rmse`.

    Returns:
        The score. Where σ is 0, 1 for a perfect prediction and −∞ otherwise.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If ``normalise`` is neither, or there is no such axis.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> float(nrmse_score([3.0, -3.0], [4.0, -4.0]))
        0.75
    """
    if normalise not in ("std", "rms"):
        msg = f"'normalise' must be 'std' or 'rms', got {normalise!r}"
        raise ValidationError(msg)
    xp, p, r, cells = paired_inputs(prediction, reference, weights, valid, axis)
    square = cells.variance(r) if normalise == "std" else cells.mean(r**2)
    error = root(xp, cells.mean((p - r) ** 2))
    return 1.0 - ratio_or_inf(xp, error, root(xp, square))
