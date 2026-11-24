"""The mixed-layer depth, by a density threshold."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Literal

import numpy as np

from diagnostix._src.columns.crossing import crossing
from diagnostix._src.columns.levels import column_inputs
from diagnostix._src.reduce import boolean_like, on_device
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["mixed_layer_depth"]


def mixed_layer_depth(
    density: ArrayLike,
    depth: ArrayLike,
    *,
    axis: int = -1,
    reference_depth: float = 10.0,
    threshold: float = 0.03,
    unreached: Literal["bottom", "nan"] = "bottom",
    valid: ArrayLike | None = None,
) -> Any:
    """The depth where the density first exceeds its near-surface value.

    h is the first z below z_ref where ρ(z) − ρ(z_ref) = Δρ, with ρ linear
    between the levels, at z_ref too.

    Args:
        density: ρ (kg m⁻³), usually potential density, with its levels along
            ``axis``.
        depth: z (m), positive down and increasing, 1-D or one column per point.
        axis: The axis of the levels.
        reference_depth: z_ref (m). A column whose first valid level is deeper
            uses that level in its place, and one that ends above it has not
            left the mixed layer.
        threshold: Δρ > 0 (kg m⁻³).
        unreached: What a column that never exceeds the threshold gives:
            ``"bottom"``, its deepest valid depth, or ``"nan"``.
        valid: Which levels to use (True = use), broadcast against ``density``,
            or 1-D along the levels; every level if not given. An invalid level
            is left out of its column, and what is under it is not read.

    Returns:
        h (m), ``density`` without ``axis``, NaN in a column with no valid level.
        A NaN in a valid level at or above h gives NaN.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, ``depth`` does not
            have those of ``density``, ``valid`` is not boolean, or
            ``unreached`` is not one of the two.

    References:
        de Boyer Montégut et al. (2004), doi:10.1029/2004JC002378.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([0.0, 10.0, 20.0, 40.0])
        >>> rho = np.asarray([0.0, 0.0, 0.02, 0.06])
        >>> round(float(mixed_layer_depth(rho, z)), 9)
        25.0

        Land, and a column mixed to its bottom at 20 m:

        >>> rho = np.zeros((2, 4))
        >>> wet = np.asarray([[False] * 4, [True, True, True, False]])
        >>> mixed_layer_depth(rho, z, valid=wet)
        array([nan, 20.])
    """
    if unreached not in ("bottom", "nan"):
        msg = f"'unreached' must be 'bottom' or 'nan', got {unreached!r}"
        raise ValidationError(msg)
    xp, (rho,), z, (z_ref, delta) = column_inputs(
        (density,),
        depth,
        axis=axis,
        per_column=(reference_depth, threshold),
        params=(valid,),
    )
    has_level = xp.ones_like(rho[..., 0], dtype=xp.bool)
    if valid is not None:
        usable = _levels_of(xp, valid, rho, axis)
        rho, z, has_level = _valid_first(xp, rho, z, usable)

    # --- the density at the reference depth ---
    # ρ_ref = ρ(z_ref): the crossing of z through z_ref, on ρ
    reaches, at_reference = crossing(xp, z, rho, z_ref)
    # a column that ends above z_ref is all mixed layer: its last level is ρ_ref
    rho_ref = xp.where(reaches, at_reference, rho[..., -1])
    anomaly = rho - xp.expand_dims(rho_ref, axis=-1)

    # --- the first level below it that is denser by the threshold ---
    # a level at or above z_ref is never the answer, however dense it is
    below = z > z_ref
    capped = xp.where(anomaly > 0, xp.zeros_like(anomaly), anomaly)
    anomaly = xp.where(below, anomaly, capped)
    found, mixed = crossing(xp, anomaly, z, delta)

    missing = xp.full_like(mixed, xp.nan)
    fallback = z[..., -1] if unreached == "bottom" else missing
    out = xp.where(found, mixed, fallback)
    return xp.where(has_level, out, missing)


def _levels_of(xp: ModuleType, valid: ArrayLike, rho: Any, axis: int) -> Any:
    """``valid`` with its levels last, broadcast against the density."""
    usable = boolean_like(xp, valid, rho)
    if usable.ndim > 1:
        usable = xp.moveaxis(usable, axis, -1)
    return xp.broadcast_to(usable, rho.shape)


def _valid_first(xp: ModuleType, rho: Any, z: Any, usable: Any) -> tuple[Any, Any, Any]:
    """Each column's valid levels in order, the deepest repeated to the end.

    A repeated level adds nothing: it never rises above the one before it.

    Returns:
        The density and the depth, and whether the column has a valid level; a
        column without one is 0 at depth 0.
    """
    rho = xp.where(usable, rho, xp.zeros_like(rho))
    z = xp.where(usable, z, xp.zeros_like(z))
    # the valid levels come first, in their order
    order = xp.argsort(xp.astype(~usable, xp.int32), axis=-1, stable=True)
    n_valid = xp.sum(xp.astype(usable, order.dtype), axis=-1, keepdims=True)
    levels = on_device(xp, np.arange(rho.shape[-1]), rho)
    levels = xp.broadcast_to(xp.astype(levels, order.dtype), order.shape)
    last = xp.maximum(n_valid - 1, xp.zeros_like(n_valid))
    chosen = xp.take_along_axis(order, xp.minimum(levels, last), axis=-1)
    return (
        xp.take_along_axis(rho, chosen, axis=-1),
        xp.take_along_axis(z, chosen, axis=-1),
        n_valid[..., 0] > 0,
    )
