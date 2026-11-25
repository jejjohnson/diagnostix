"""The flux of a quantity through the faces of a grid, and what it leaves behind."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gridax
from pipekit_array.arrays import constants_like, table_like

from diagnostix._src.budgets.faces import (
    across,
    face_lengths,
    face_mean,
    orientation,
)
from diagnostix._src.budgets.volume import ControlVolume
from diagnostix._src.grids import grid_inputs
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["boundary_flux", "face_fluxes", "flux_divergence", "require_volume"]


def face_fluxes(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    *,
    thickness: ArrayLike | None = None,
) -> tuple[Any, Any]:
    """The flux of a quantity through every face between two cells.

    Fᴱ = ½ ((q u H)_{j,i} + (q u H)_{j,i+1}) ℓᴱ and
    Fᴺ = ½ ((q v H)_{j,i} + (q v H)_{j+1,i}) ℓᴺ, with ℓ the length of the face:
    R Δφ and R cos(φ_{j+½}) Δλ on a sphere, Δy and Δx on a plane.

    Args:
        q: The quantity per unit volume, or per unit area without a thickness,
            broadcast against u.
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        thickness: H (m), broadcast against u; 1 if not given.

    Returns:
        Fᴱ, positive towards larger x, with one face fewer than cells along x
        (as many if x is periodic), and Fᴺ, positive towards larger y, the same
        along y. The units are those of q m² s⁻¹, or m³ s⁻¹ with a thickness.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or a velocity
            does not end in its shape.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_interior(3, 2, 3.0, 4.0)
        >>> u = np.asarray([[1.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
        >>> east, north = face_fluxes(10.0, u, np.ones((2, 3)), grid)
        >>> east
        array([[40., 80.],
               [ 0.,  0.]])
        >>> north
        array([[10., 10., 10.]])
    """
    _, east, north, _ = _fluxes(q, u, v, grid, thickness)
    return east, north


def boundary_flux(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    volume: ControlVolume,
    *,
    thickness: ArrayLike | None = None,
) -> Any:
    """The flux of a quantity out of a control volume, through its own faces.

    F = ∑ σᴱ Fᴱ + ∑ σᴺ Fᴺ over the boundary, with F the `face_fluxes` and σ the
    signs of the faces. It equals the area sum of `flux_divergence` over the
    region, because every inner face cancels.

    Args:
        q: The quantity per unit volume, or per unit area without a thickness,
            broadcast against u.
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: The grid ``volume`` was built on.
        volume: The region and its faces.
        thickness: H (m), broadcast against u; 1 if not given.

    Returns:
        F, positive outwards, u without (y, x), in the units of `face_fluxes`. A
        NaN on either side of a face of the boundary makes it NaN, because a
        budget with a missing face does not close; no other cell is read.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: As `face_fluxes`, or if ``volume`` is of another grid.

    Examples:
        A radial flow leaves a box at twice its area:

        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_interior(6, 6, 6.0, 6.0)
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> inside = (abs(x - 3.0) < 1.0) & (abs(y - 2.5) < 2.0)
        >>> box = ControlVolume.from_region(inside, grid)
        >>> float(boundary_flux(1.0, x, y, grid, box) / inside.sum())
        2.0
    """
    require_volume(volume, grid)
    xp, east, north, _ = _fluxes(q, u, v, grid, thickness)
    through_x = _signed_sum(xp, east, volume.east_sign)
    return through_x + _signed_sum(xp, north, volume.north_sign)


def flux_divergence(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    *,
    thickness: ArrayLike | None = None,
) -> Any:
    """The flux out of each cell through its four faces, per unit area.

    (Fᴱ_{j,i} − Fᴱ_{j,i−1} + Fᴺ_{j,i} − Fᴺ_{j−1,i}) / A, with F the
    `face_fluxes` and A the cell area. Its area sum over a region is the
    `boundary_flux` of the region to round-off, which that of
    `gridax.divergence` is not on a sphere, where its metric is at the centres.

    Args:
        q: The quantity per unit volume, or per unit area without a thickness,
            broadcast against u.
        u: Eastward velocity (m s⁻¹), ending in the grid's (y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        thickness: H (m), broadcast against u; 1 if not given.

    Returns:
        The divergence of the flux (units of q s⁻¹, or of q m s⁻¹ with a
        thickness), NaN on the edge cells of an axis that does not wrap.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: As `face_fluxes`.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_interior(4, 4, 4.0, 4.0)
        >>> x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        >>> flux_divergence(1.0, 3.0 * x, 2.0 * y, grid)[1]
        array([nan,  5.,  5., nan])
    """
    xp, east, north, ref = _fluxes(q, u, v, grid, thickness)
    wraps = [axis in grid.periodic for axis in "xy"]
    sign_x, sign_y = (constants_like(orientation(grid, a), ref, xp) for a in "xy")
    # Fᴱ_{j,i} − Fᴱ_{j,i−1}, towards larger x whichever way it is stored
    out_x = sign_x * across(xp, east, axis=-1, periodic=wraps[0])
    out_y = sign_y * across(xp, north, axis=-2, periodic=wraps[1])
    return (out_x + out_y) / constants_like(gridax.area_weights(grid), ref, xp)


def require_volume(volume: ControlVolume, grid: Any) -> None:
    """Raise unless a control volume has the shape of a grid.

    Raises:
        ValidationError: If it does not.
    """
    if volume.inside.shape != grid.shape:
        msg = (
            f"'volume' is of a grid of shape {volume.inside.shape}, and 'grid' "
            f"has {grid.shape}"
        )
        raise ValidationError(msg)


def _fluxes(
    q: ArrayLike, u: ArrayLike, v: ArrayLike, grid: Any, thickness: ArrayLike | None
) -> tuple[ModuleType, Any, Any, Any]:
    """The namespace, the fluxes through the faces, and a field of the call."""
    others = (q,) if thickness is None else (q, thickness)
    xp, (u, v, carried, *height) = grid_inputs(grid, u, v, broadcast=others)
    if height:
        carried = carried * height[0]
    east_length, north_length = face_lengths(xp, grid, u)
    # (q u H)ᶠ: the mean of the product, which is what makes the sums cancel
    east = face_mean(xp, carried * u, axis=-1, periodic="x" in grid.periodic)
    north = face_mean(xp, carried * v, axis=-2, periodic="y" in grid.periodic)
    return xp, east * east_length, north * north_length, u


def _signed_sum(xp: ModuleType, faces: Any, signs: Any) -> Any:
    """∑ σ F over the faces of the boundary; the others are not read."""
    on_boundary = table_like(signs != 0, faces, xp)
    sign = xp.astype(table_like(signs, faces, xp), faces.dtype)
    kept = xp.where(on_boundary, faces, xp.zeros_like(faces))
    return xp.sum(sign * kept, axis=(-2, -1), dtype=faces.dtype)
