"""The covariant velocity gradient on collocated grids, and the vorticity's parts."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import gridax
from jaxtyping import Float
from pipekit_array.functions import floating_inputs, namespace_of
from pipekit_array.records import register_record

from diagnostix._src.grids import grid_inputs
from diagnostix._typing import ArrayLike


__all__ = [
    "VelocityGradient",
    "curvature_vorticity",
    "shear_vorticity",
    "velocity_gradient",
]

#: cos φ below which a row is at a pole, where tan φ has no value (gridax's).
POLE = 1e-12


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class VelocityGradient:
    """The covariant derivatives of a horizontal velocity.

    On a sphere each carries its metric term, so they are the components of one
    tensor: δ = ``dudx + dvdy`` is the divergence, ζ = ``dvdx − dudy`` the
    vorticity, and ``dudx − dvdy`` and ``dvdx + dudy`` are the two strains.

    Attributes:
        dudx: a_xx = ∂ₓu − v tan φ / R (s⁻¹).
        dudy: a_xy = ∂ᵧu (s⁻¹).
        dvdx: a_yx = ∂ₓv + u tan φ / R (s⁻¹).
        dvdy: a_yy = ∂ᵧv (s⁻¹).
    """

    dudx: Float[Any, "*batch y x"]
    dudy: Float[Any, "*batch y x"]
    dvdx: Float[Any, "*batch y x"]
    dvdy: Float[Any, "*batch y x"]


def velocity_gradient(
    u: ArrayLike, v: ArrayLike, grid: Any, *, mask: Any = None
) -> VelocityGradient:
    """The covariant gradient of a horizontal velocity.

    With ∂ₓ and ∂ᵧ derivatives along arc length (`gridax.difference`: centred
    inside, one-sided at the edges and beside a dry cell),

    a_xx = ∂ₓu − v tan φ / R,  a_xy = ∂ᵧu,
    a_yx = ∂ₓv + u tan φ / R,  a_yy = ∂ᵧv;

    the terms in tan φ vanish on a plane.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        The four derivatives (s⁻¹), NaN in a dry cell. ``dudx`` and ``dvdx`` are
        NaN on a row at a pole, where tan φ has no value.

    Raises:
        BackendError: If u or v is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a field does
            not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 2.0, 3.0]
        ... )
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> gradient = velocity_gradient(2.0 * x - 3.0 * y, 0.5 * x, grid)
        >>> gradient.dvdx - gradient.dudy
        array([[3.5, 3.5, 3.5],
               [3.5, 3.5, 3.5],
               [3.5, 3.5, 3.5]])
    """
    xp, (u, v) = grid_inputs(grid, u, v)
    u, v = xp.broadcast_arrays(u, v)
    dudx = gridax.difference(u, grid, axis="x", mask=mask)
    dvdx = gridax.difference(v, grid, axis="x", mask=mask)
    if isinstance(grid, gridax.SphericalAGrid2D):
        # a_xx = ∂ₓu − v tan φ / R, a_yx = ∂ₓv + u tan φ / R
        dudx = dudx - _turning(xp, v, grid, mask)
        dvdx = dvdx + _turning(xp, u, grid, mask)
    return VelocityGradient(
        dudx=dudx,
        dudy=gridax.difference(u, grid, axis="y", mask=mask),
        dvdx=dvdx,
        dvdy=gridax.difference(v, grid, axis="y", mask=mask),
    )


def shear_vorticity(u: ArrayLike, v: ArrayLike, grid: Any, *, mask: Any = None) -> Any:
    """The part of the vorticity due to the speed changing across the flow.

    ζ_s = (u v a_xx + v² a_yx − u² a_xy − u v a_yy) / (u² + v²), with a the
    `velocity_gradient`. ζ_s + `curvature_vorticity` = a_yx − a_xy, which differs
    from `gridax.curl` at truncation order on a sphere.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        ζ_s (s⁻¹), 0 at a calm point (u = v = 0), NaN in a dry cell and on a row
        at a pole.

    Raises:
        BackendError: If u or v is a DataArray or of another library.
        ValidationError: As `velocity_gradient`.

    Examples:
        A jet along x: all of its vorticity is shear.

        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 2.0, 3.0]
        ... )
        >>> y = np.broadcast_to(grid.coords("y")[:, None], grid.shape)
        >>> shear_vorticity(1.0 + 2.0 * y, 0.0 * y, grid)[:, 0]
        array([-2., -2., -2.])
    """
    u, v, a, speed_squared = _gradient_and_speed(u, v, grid, mask)
    shear = u * v * (a.dudx - a.dvdy) + v**2 * a.dvdx - u**2 * a.dudy
    return shear / speed_squared


def curvature_vorticity(
    u: ArrayLike, v: ArrayLike, grid: Any, *, mask: Any = None
) -> Any:
    """The part of the vorticity due to the flow changing direction.

    ζ_c = (u² a_yx − v² a_xy − u v a_xx + u v a_yy) / (u² + v²), with a the
    `velocity_gradient`. `shear_vorticity` + ζ_c = a_yx − a_xy.

    Args:
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        mask: A `gridax.Mask2D`, the wet cells; a dry cell is never read.

    Returns:
        ζ_c (s⁻¹), 0 at a calm point (u = v = 0), NaN in a dry cell and on a row
        at a pole.

    Raises:
        BackendError: If u or v is a DataArray or of another library.
        ValidationError: As `velocity_gradient`.

    Examples:
        A jet along x has none:

        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_coords(
        ...     y=[0.0, 1.0, 3.0], x=[0.0, 2.0, 3.0]
        ... )
        >>> y = np.broadcast_to(grid.coords("y")[:, None], grid.shape)
        >>> curvature_vorticity(1.0 + 2.0 * y, 0.0 * y, grid)[:, 0]
        array([0., 0., 0.])
    """
    u, v, a, speed_squared = _gradient_and_speed(u, v, grid, mask)
    curvature = u**2 * a.dvdx - v**2 * a.dudy - u * v * (a.dudx - a.dvdy)
    return curvature / speed_squared


def _gradient_and_speed(
    u: ArrayLike, v: ArrayLike, grid: Any, mask: Any
) -> tuple[Any, Any, VelocityGradient, Any]:
    """u, v, the velocity gradient, and u² + v², 1 at a calm point."""
    a = velocity_gradient(u, v, grid, mask=mask)
    xp, (u, v) = floating_inputs(u, v)
    speed_squared = u**2 + v**2
    # both numerators are 0 at a calm point: 0 / 1 there, with a finite gradient
    calm = speed_squared == 0
    return u, v, a, xp.where(calm, xp.ones_like(speed_squared), speed_squared)


def _turning(xp: ModuleType, field: Any, grid: Any, mask: Any) -> Any:
    """f tan φ / R, the metric term of a vector on a sphere; NaN at a pole."""
    # NumPy, or of the library of a traced grid
    latitude = grid.coords("y")[:, None]
    host = namespace_of(latitude)
    cosine = host.cos(latitude)
    at_pole = host.abs(cosine) < POLE
    # the unused branch divides by 1, so a traced grid's gradient is finite
    safe = host.where(at_pole, host.ones_like(cosine), cosine)
    # tan φ / R
    factor = host.where(at_pole, host.nan, host.sin(latitude) / (safe * grid.R))
    if mask is not None:
        # a dry cell is NaN already, and is not read
        field = xp.where(mask.wet(), field, xp.zeros_like(field))
    return field * xp.asarray(factor, dtype=field.dtype)
