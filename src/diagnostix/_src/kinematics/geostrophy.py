"""How far a flow is from geostrophic balance."""

from __future__ import annotations

from typing import Any

import gridax
from geophysix.constants import STANDARD
from pipekit_array.functions import like

from diagnostix._src.kinematics.inputs import grid_inputs
from diagnostix._typing import ArrayLike


__all__ = ["ageostrophic_velocity", "geostrophic_residual"]


def ageostrophic_velocity(
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    grid: Any,
    *,
    f: ArrayLike,
    gravity: float = STANDARD.standard_gravity,
    mask: Any = None,
) -> tuple[Any, Any]:
    """What is left of a velocity without its geostrophic part.

    (u, v) − (u_g, v_g), with u_g = −(g/f) ∂ᵧη and v_g = (g/f) ∂ₓη
    (`gridax.geostrophic_velocity`).

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        height: η, the height of the surface (m), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        f: Coriolis parameter (s⁻¹), broadcast against u: a number, a column of
            one value per row, or a field (`geophysix.rotation.coriolis`,
            `geophysix.rotation.beta_plane`).
        gravity: g (m s⁻²).
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read. To
            leave out a band around the equator, make it dry.

    Returns:
        The two components (m s⁻¹), NaN where f = 0, in a dry cell and on a row
        at a pole.

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
        >>> u, v = ageostrophic_velocity(
        ...     wind, wind, x - 2.0 * y, grid, f=4.0, gravity=8.0
        ... )
        >>> u[0], v[0]
        (array([-3., -3., -3.]), array([-1., -1., -1.]))
    """
    _, (u, v, eta, f) = grid_inputs(grid, u, v, height, broadcast=(f,))
    u_g, v_g = gridax.geostrophic_velocity(eta, grid, f=f, gravity=gravity, mask=mask)
    return u - u_g, v - v_g


def geostrophic_residual(
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    grid: Any,
    *,
    f: ArrayLike,
    gravity: float = STANDARD.standard_gravity,
    mask: Any = None,
) -> tuple[Any, Any]:
    """The forces left over by geostrophic balance, defined at the equator too.

    r_u = f u + g ∂ᵧη and r_v = f v − g ∂ₓη, both 0 in a geostrophic flow.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        height: η, the height of the surface (m), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        f: Coriolis parameter (s⁻¹), broadcast against u: a number, a column of
            one value per row, or a field (`geophysix.rotation.coriolis`,
            `geophysix.rotation.beta_plane`).
        gravity: g (m s⁻²).
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        r_u and r_v (m s⁻²), NaN in a dry cell; r_v is NaN on a row at a pole.

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
        >>> r_u, r_v = geostrophic_residual(
        ...     wind, wind, x - 2.0 * y, grid, f=4.0, gravity=8.0
        ... )
        >>> r_u[0], r_v[0]
        (array([-12., -12., -12.]), array([-4., -4., -4.]))
    """
    xp, (u, v, eta, f) = grid_inputs(
        grid, u, v, height, broadcast=(f,), params=(gravity,)
    )
    g = like(gravity, eta, xp)
    slope_x, slope_y = gridax.gradient(eta, grid, mask=mask)
    # r_u = f u + g ∂ᵧη, r_v = f v − g ∂ₓη
    return f * u + g * slope_y, f * v - g * slope_x
