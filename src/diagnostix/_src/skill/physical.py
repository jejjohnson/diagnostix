"""Scores of physical consistency: balance, divergence, stratification."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gridax
from geophysix.constants import STANDARD
from pipekit_array.functions import (
    boolean_like,
    constants_like,
    divide_or_nan,
    floating_inputs,
)
from pipekit_array.weighted import used_weights

from diagnostix._src.columns.levels import column_inputs
from diagnostix._src.grids import grid_inputs
from diagnostix._src.kinematics.geostrophy import geostrophic_residual
from diagnostix._src.skill.moments import Cells, root
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = [
    "density_inversion_fraction",
    "geostrophic_imbalance",
    "qg_balance_residual",
    "rms_divergence",
]


def geostrophic_imbalance(
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    grid: Any,
    *,
    f: ArrayLike,
    gravity: float = STANDARD.standard_gravity,
    weights: ArrayLike | None = None,
    mask: Any = None,
) -> Any:
    """The forces left over by geostrophic balance, relative to Coriolis.

    ((⟨r_u²⟩ + ⟨r_v²⟩) / (⟨(f v)²⟩ + ⟨(f u)²⟩))^½, with r the `geostrophic_residual`
    and ⟨·⟩ the mean over the wet cells, weighted by area.

    0 for a geostrophic flow, and about 1 where the pressure gradient has
    nothing to do with the flow. Its gradient at a geostrophic flow is 0.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        height: η, the height of the surface (m), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        f: Coriolis parameter (s⁻¹), broadcast against u.
        gravity: g (m s⁻²).
        weights: w ≥ 0 of each cell, broadcast against u; the cell areas if not
            given.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read. A row
            at a pole has no eastward gradient, so it is to be dry.

    Returns:
        The imbalance, dimensionless, u without (y, x). NaN for a state of rest,
        and where no cell is wet.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 2.0], x=[0.0, 1.0, 2.0]
        ... )
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> u, v = -np.ones(grid.shape), 2.0 * np.ones(grid.shape)
        >>> balanced = 2.0 * x + y  # f u = −g ∂η/∂y, f v = g ∂η/∂x
        >>> options = {"f": 1e-4, "gravity": 1e-4}
        >>> float(geostrophic_imbalance(u, v, balanced, grid, **options))
        0.0
        >>> float(geostrophic_imbalance(u, v, 0.0 * x, grid, **options))
        1.0
    """
    given = () if weights is None else (weights,)
    xp, (u, v, eta, f, *w) = grid_inputs(
        grid, u, v, height, broadcast=(f, *given), params=(gravity,)
    )
    cells, wet = _wet_cells(xp, grid, mask, w, u)
    # before any product: 0 times the derivative of a NaN is NaN
    u, v = _wet(xp, wet, u), _wet(xp, wet, v)
    r_u, r_v = geostrophic_residual(u, v, eta, grid, f=f, gravity=gravity, mask=mask)
    left = cells.mean(_wet(xp, wet, r_u) ** 2) + cells.mean(_wet(xp, wet, r_v) ** 2)
    coriolis = cells.mean((f * u) ** 2) + cells.mean((f * v) ** 2)
    return root(xp, divide_or_nan(left, coriolis, xp))


def rms_divergence(
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    *,
    weights: ArrayLike | None = None,
    mask: Any = None,
) -> Any:
    """The root mean square of the horizontal divergence.

    ⟨(∇·u)²⟩^½, with `gridax.divergence` and ⟨·⟩ the mean over the wet cells,
    weighted by area. Its gradient at a non-divergent flow is 0.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        weights: w ≥ 0 of each cell, broadcast against u; the cell areas if not
            given.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        The score (s⁻¹), u without (y, x), NaN where no cell is wet.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 1.0, 2.0]
        ... )
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> float(rms_divergence(2.0 * x - y, 3.0 * y, grid))
        5.0
    """
    given = () if weights is None else (weights,)
    xp, (u, v, *w) = grid_inputs(grid, u, v, broadcast=given)
    cells, wet = _wet_cells(xp, grid, mask, w, u)
    divergence = gridax.divergence(u, v, grid, mask=mask)
    return root(xp, cells.mean(_wet(xp, wet, divergence) ** 2))


def density_inversion_fraction(
    density: ArrayLike,
    z: ArrayLike,
    *,
    axis: int = -1,
    valid: ArrayLike | None = None,
) -> Any:
    """The share of neighbouring levels of a column with the denser one on top.

    ∑ [Δρ / Δz > 0] / ∑ 1 over the pairs of neighbouring levels that are both
    valid. Its gradient is 0.

    Args:
        density: ρ (kg m⁻³), usually potential density, with its levels along
            ``axis``.
        z: Height (m), upward, in either order, 1-D or one column per point. From
            a depth, positive down, pass ``-depth``.
        axis: The axis of the levels.
        valid: Which levels to use (True = use), 1-D or broadcast against
            ``density``. A NaN in a valid level gives NaN.

    Returns:
        The share, in [0, 1], ``density`` without ``axis``. NaN in a column with
        no pair of valid levels.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or ``z`` does not
            have those of ``density``.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> depth = [0.0, 10.0, 20.0, 30.0, 40.0]
        >>> rho = [1025.0, 1025.5, 1025.3, 1026.0, 1026.5]
        >>> float(density_inversion_fraction(rho, [-d for d in depth]))
        0.25
    """
    xp, (rho,), z, _ = column_inputs((density,), z, axis=axis)
    pairs = xp.ones_like(rho[..., 1:]) > 0
    if valid is not None:
        usable = boolean_like(valid, rho, xp)
        if usable.ndim > 1:
            usable = xp.moveaxis(usable, axis, -1)
        usable = xp.broadcast_to(usable, rho.shape)
        pairs = usable[..., 1:] & usable[..., :-1]
        rho = xp.where(usable, rho, xp.zeros_like(rho))
        z = xp.where(usable, z, xp.zeros_like(z))
    rise = rho[..., 1:] - rho[..., :-1]
    up = z[..., 1:] - z[..., :-1]
    inverted = ((rise > 0) & (up > 0)) | ((rise < 0) & (up < 0))
    share = xp.where(inverted, xp.ones_like(rise), xp.zeros_like(rise))
    # a comparison hides a NaN
    share = xp.where(xp.isnan(rise) | xp.isnan(up), xp.full_like(rise, xp.nan), share)
    one = xp.ones_like(rise)
    return Cells(xp, used_weights(xp, rise, one, pairs), -1).mean(share)


def qg_balance_residual(
    q: ArrayLike,
    psi: ArrayLike,
    grid: Any,
    *,
    f0: float,
    beta: float = 0.0,
    y0: float = 0.0,
    A: ArrayLike | None = None,
    mask: Any = None,
) -> Any:
    """How far a potential vorticity is from that of its own streamfunction.

    (⟨(q − q(ψ))²⟩ / ⟨q²⟩)^½, with q(ψ) the `gridax.qg_potential_vorticity`, ∇²ψ / f₀
    + β (y − y₀) / f₀ − f₀ ∑ A ψ, and ⟨·⟩ the mean over the wet interior.

    0 for a state in quasi-geostrophic balance. Its gradient there is 0.

    Args:
        q: The potential vorticity, as that function defines it, at T, ending in
            the grid's (y, x).
        psi: ψ, the streamfunction, as q, with its ghost ring filled.
        grid: A `gridax.CartesianCGrid2D`, the grid of that function.
        f0: f₀ (s⁻¹).
        beta: β (m⁻¹ s⁻¹).
        y0: Where β is centred (m).
        A: The coupling matrix of the layers; ψ is then (…, layer, y, x).
        mask: A `gridax.Mask2D`, the wet cells. ψ is read as 0 in a dry one, and
            q is not read there.

    Returns:
        The residual, dimensionless, q without (y, x). NaN where q is 0.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not a `gridax.CartesianCGrid2D`, or a
            field does not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianCGrid2D.from_interior(3, 3, 3.0, 3.0)
        >>> psi = np.broadcast_to(grid.coords("x") ** 2, grid.shape)
        >>> q = np.full(grid.shape, 4.0)  # ∇²ψ / f₀
        >>> float(qg_balance_residual(q, psi, grid, f0=0.5))
        0.0
        >>> float(qg_balance_residual(1.25 * q, psi, grid, f0=0.5))
        0.2
    """
    if not isinstance(grid, gridax.CartesianCGrid2D):
        msg = f"'grid' must be a CartesianCGrid2D, got {type(grid).__name__}"
        raise ValidationError(msg)
    given = () if A is None else (A,)
    xp, (q, psi, *coupling) = floating_inputs(q, psi, *given, params=(f0, beta, y0))
    for field in (q, psi):
        if tuple(field.shape[-2:]) != grid.shape:
            msg = (
                f"a field on this grid ends in {grid.shape} (y, x), "
                f"got shape {tuple(field.shape)}"
            )
            raise ValidationError(msg)
    of_psi = gridax.qg_potential_vorticity(
        psi,
        grid,
        f0=f0,
        beta=beta,
        y0=y0,
        A=coupling[0] if coupling else None,
        mask=mask,
    )
    interior = (..., slice(1, -1), slice(1, -1))
    q, of_psi = q[interior], of_psi[interior]
    wet = None if mask is None else boolean_like(mask.wet()[1:-1, 1:-1], q, xp)
    q, of_psi = _wet(xp, wet, q), _wet(xp, wet, of_psi)
    cells = Cells(xp, used_weights(xp, q, None, wet), (-2, -1))
    ratio = divide_or_nan(cells.mean((q - of_psi) ** 2), cells.mean(q**2), xp)
    return root(xp, ratio)


def _wet_cells(
    xp: ModuleType, grid: Any, mask: Any, weights: list[Any], ref: Any
) -> tuple[Cells, Any]:
    """The cells of a grid, weighted by area if not otherwise, and the wet ones."""
    w = weights[0] if weights else constants_like(gridax.area_weights(grid), ref, xp)
    w = xp.broadcast_to(w, ref.shape)
    wet = None if mask is None else boolean_like(mask.wet(), ref, xp)
    return Cells(xp, used_weights(xp, ref, w, wet), (-2, -1)), wet


def _wet(xp: ModuleType, wet: Any, values: Any) -> Any:
    """Values with 0 in the dry cells."""
    return values if wet is None else xp.where(wet, values, xp.zeros_like(values))
