"""What a horizontal velocity does to a scalar: advection and frontogenesis."""

from __future__ import annotations

from typing import Any

import gridax

from diagnostix._src.kinematics.gradient import velocity_gradient
from diagnostix._src.kinematics.inputs import grid_inputs
from diagnostix._typing import ArrayLike


__all__ = ["advection", "frontogenesis"]


def advection(
    scalar: ArrayLike, u: ArrayLike, v: ArrayLike, grid: Any, *, mask: Any = None
) -> Any:
    """The tendency of a scalar by horizontal advection.

    −(u ∂ₓc + v ∂ᵧc), with ∂ₓ and ∂ᵧ along arc length
    (`gridax.gradient`).

    Args:
        scalar: c, ending in the grid's (y, x).
        u: Eastward velocity (m s⁻¹), as c.
        v: Northward velocity (m s⁻¹), as c.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        −(u ∂ₓc + v ∂ᵧc) (units of c s⁻¹), NaN in a dry cell and on a row at a pole.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 2.0, 3.0]
        ... )
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> wind = np.ones(grid.shape)
        >>> advection(2.0 * x - y, 3.0 * wind, 1.0 * wind, grid)[0]
        array([-5., -5., -5.])
    """
    _, (c, u, v) = grid_inputs(grid, scalar, u, v)
    dc_dx, dc_dy = gridax.gradient(c, grid, mask=mask)
    # −(u ∂ₓc + v ∂ᵧc)
    return -(u * dc_dx + v * dc_dy)


def frontogenesis(
    scalar: ArrayLike, u: ArrayLike, v: ArrayLike, grid: Any, *, mask: Any = None
) -> Any:
    """The rate at which the flow sharpens the gradient of a scalar.

    F = D|∇c|/Dt by the kinematics of a conserved c alone,

    F = −((∂ₓc)² a_xx + (∂ᵧc)² a_yy + ∂ₓc ∂ᵧc (a_xy + a_yx)) / |∇c|,

    with a the `velocity_gradient`. This is Petterssen's
    ½ |∇c| (D cos 2β − δ), with D the total deformation, β the angle between
    the axis of dilatation and the isolines, and δ the divergence.

    Args:
        scalar: c, ending in the grid's (y, x).
        u: Eastward velocity (m s⁻¹), as c.
        v: Northward velocity (m s⁻¹), as c.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        F (units of c m⁻¹ s⁻¹), positive where the gradient sharpens, 0 where
        c is uniform, NaN in a dry cell and on a row at a pole.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.

    Examples:
        A flow that converges along y sharpens a gradient along y:

        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 2.0, 3.0]
        ... )
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> frontogenesis(3.0 * y, 2.0 * x, -2.0 * y, grid)[0]
        array([6., 6., 6.])
    """
    xp, (c, u, v) = grid_inputs(grid, scalar, u, v)
    dc_dx, dc_dy = gridax.gradient(c, grid, mask=mask)
    a = velocity_gradient(u, v, grid, mask=mask)
    norm_squared = dc_dx**2 + dc_dy**2
    # the sharpening is 0 where c is uniform: 0 / 1 there, with a finite gradient
    uniform = norm_squared == 0
    safe = xp.where(uniform, xp.ones_like(norm_squared), norm_squared)
    # (∂ₓc)² a_xx + (∂ᵧc)² a_yy + ∂ₓc ∂ᵧc (a_xy + a_yx)
    stretching = dc_dx**2 * a.dudx + dc_dy**2 * a.dvdy
    stretching = stretching + dc_dx * dc_dy * (a.dudy + a.dvdx)
    return -stretching / xp.sqrt(safe)
