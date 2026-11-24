"""The bulk Richardson number, and the boundary-layer height it gives."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from geophysix.constants import STANDARD
from pipekit_array.functions import like

from diagnostix._src.columns.crossing import crossing
from diagnostix._src.columns.levels import column_inputs
from diagnostix._typing import ArrayLike


__all__ = ["boundary_layer_height", "bulk_richardson"]


def bulk_richardson(
    virtual_potential_temperature: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    *,
    axis: int = -1,
    friction_velocity: ArrayLike | None = None,
    friction_coefficient: float = 100.0,
    gravity: float = STANDARD.standard_gravity,
) -> Any:
    """The bulk Richardson number of each level against the lowest.

    With s the lowest level, u∗ the friction velocity and b its coefficient,

    Ri_b(z) = g/θ_{v,s} · (θ_v − θ_{v,s}) (z − z_s)
              / ((u − u_s)² + (v − v_s)² + b u∗²).

    Args:
        virtual_potential_temperature: θ_v (K), with its levels along ``axis``,
            the surface first.
        u: Eastward wind (m s⁻¹), as θ_v.
        v: Northward wind (m s⁻¹), as θ_v.
        height: z (m), upward, 1-D or one column per point.
        axis: The axis of the levels.
        friction_velocity: u∗ (m s⁻¹), one number, or one per column (θ_v
            without ``axis``); without it the term b u∗² is left out.
        friction_coefficient: b.
        gravity: g (m s⁻²).

    Returns:
        Ri_b, the shape of θ_v, 0 at the lowest level. Where the denominator is
        0 (a calm level) it is +∞ if stable, −∞ if unstable and 0 if neutral.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or the inputs do
            not have the same ones.

    References:
        Vogelezang and Holtslag (1996), doi:10.1007/BF02430331.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([10.0, 110.0, 210.0])
        >>> theta_v = np.asarray([300.0, 300.0, 303.0])
        >>> u = np.asarray([2.0, 4.0, 2.0])
        >>> bulk_richardson(theta_v, u, np.zeros(3), z)
        array([ 0.,  0., inf])
        >>> ri = bulk_richardson(theta_v, u, np.zeros(3), z, friction_velocity=0.5)
        >>> np.round(ri, 3)
        array([0.   , 0.   , 0.785])
    """
    xp, richardson, _ = _richardson(
        virtual_potential_temperature,
        u,
        v,
        height,
        axis=axis,
        friction_velocity=friction_velocity,
        friction_coefficient=friction_coefficient,
        gravity=gravity,
    )
    if richardson.ndim == 1:
        return richardson
    return xp.moveaxis(richardson, -1, axis)


def boundary_layer_height(
    virtual_potential_temperature: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    *,
    axis: int = -1,
    critical: float = 0.25,
    friction_velocity: ArrayLike | None = None,
    friction_coefficient: float = 100.0,
    gravity: float = STANDARD.standard_gravity,
) -> Any:
    """The lowest height where the bulk Richardson number is critical.

    h is the first z above the lowest level where Ri_b(z) = Ri_c, with Ri_b
    linear between the levels (`bulk_richardson`, `first_crossing`).

    Args:
        virtual_potential_temperature: θ_v (K), with its levels along ``axis``,
            the surface first.
        u: Eastward wind (m s⁻¹), as θ_v.
        v: Northward wind (m s⁻¹), as θ_v.
        height: z (m), upward, 1-D or one column per point. h is in the same
            frame: above ground if z is.
        axis: The axis of the levels.
        critical: Ri_c > 0.
        friction_velocity: u∗ (m s⁻¹), one number, or one per column (θ_v
            without ``axis``); without it the term b u∗² is left out.
        friction_coefficient: b.
        gravity: g (m s⁻²).

    Returns:
        h (m), θ_v without ``axis``, NaN where no level reaches Ri_c: never the
        top level. A stable calm level gives the level below it. A NaN at or
        below h gives NaN.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or the inputs do
            not have the same ones.

    References:
        Vogelezang and Holtslag (1996), doi:10.1007/BF02430331.

    Examples:
        A neutral layer under an inversion at 400 m, with shear:

        >>> import numpy as np
        >>> z = np.linspace(25.0, 975.0, 20)
        >>> theta_v = 300.0 + 0.01 * np.maximum(z - 400.0, 0.0)
        >>> u = 2.0 + 0.005 * z
        >>> round(float(boundary_layer_height(theta_v, u, np.zeros(20), z)), 1)
        390.3

        A neutral column has no such height:

        >>> float(boundary_layer_height(np.full(20, 300.0), u, np.zeros(20), z))
        nan
    """
    xp, richardson, z = _richardson(
        virtual_potential_temperature,
        u,
        v,
        height,
        axis=axis,
        friction_velocity=friction_velocity,
        friction_coefficient=friction_coefficient,
        gravity=gravity,
        params=(critical,),
    )
    threshold = like(critical, richardson, xp)
    found, top = crossing(xp, richardson, z, threshold)
    return xp.where(found, top, xp.full_like(top, xp.nan))


def _richardson(
    virtual_potential_temperature: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    height: ArrayLike,
    *,
    axis: int,
    friction_velocity: ArrayLike | None,
    friction_coefficient: float,
    gravity: float,
    params: tuple[Any, ...] = (),
) -> tuple[ModuleType, Any, Any]:
    """The namespace, Ri_b with the levels last, and the height of each level."""
    per_column = () if friction_velocity is None else (friction_velocity,)
    xp, (theta, u_wind, v_wind), z, friction = column_inputs(
        (virtual_potential_temperature, u, v),
        height,
        axis=axis,
        per_column=per_column,
        params=(friction_coefficient, gravity, *params),
    )
    theta_s = theta[..., :1]
    g = like(gravity, theta, xp)
    # g (θ_v − θ_{v,s}) (z − z_s)
    buoyancy = g * (theta - theta_s) * (z - z[..., :1])
    # (u − u_s)² + (v − v_s)²
    shear = (u_wind - u_wind[..., :1]) ** 2 + (v_wind - v_wind[..., :1]) ** 2
    if friction_velocity is not None:
        b = like(friction_coefficient, theta, xp)
        # + b u∗²
        shear = shear + b * friction[0] ** 2

    calm = shear == 0
    # the unused branch divides by 1, so its gradient is finite too
    safe = xp.where(calm, xp.ones_like(shear), shear)
    richardson = buoyancy / (theta_s * safe)
    # the limit as the shear vanishes: ±∞ with the sign of the buoyancy, or 0
    infinite = xp.full_like(buoyancy, xp.inf)
    limit = xp.where(buoyancy > 0, infinite, xp.zeros_like(buoyancy))
    limit = xp.where(buoyancy < 0, -infinite, limit)
    limit = xp.where(xp.isnan(buoyancy), xp.full_like(buoyancy, xp.nan), limit)
    return xp, xp.where(calm, limit, richardson), z
