"""The budget of a quantity, term by term, with what is left over."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any, Literal

import gridax
import numpy as np
from jaxtyping import Float

from diagnostix._src.budgets.flux import (
    boundary_flux,
    flux_divergence,
    require_volume,
)
from diagnostix._src.budgets.volume import ControlVolume
from diagnostix._src.columns.derivatives import uneven_difference
from diagnostix._src.grids import grid_inputs, metric_like
from diagnostix._src.records import register_record
from diagnostix._src.reduce import on_device
from diagnostix._src.times import time_offsets
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["Budget", "budget", "cell_residual", "region_integral"]

type Tendency = Literal["centred", "forward"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class Budget:
    """d/dt ∫ q dA + ∮ q u·n dl = ∫ S dA over a control volume, term by term.

    Attributes:
        tendency: d/dt ∫ q H dA.
        boundary_flux: ∮ q H u·n dl, positive outwards.
        source: ∫ S dA.
        residual: ``tendency + boundary_flux − source``, 0 in a closed budget.
    """

    tendency: Float[Any, "time ..."]
    boundary_flux: Float[Any, "time ..."]
    source: Float[Any, "time ..."]
    residual: Float[Any, "time ..."]


def region_integral(
    q: ArrayLike,
    grid: Any,
    volume: ControlVolume,
    *,
    thickness: ArrayLike | None = None,
) -> Any:
    """The area integral of a quantity over a control volume.

    ∫ q H dA ≈ ∑ q H A over the cells inside, with A their areas
    (`gridax.area_weights`).

    Args:
        q: The quantity per unit volume, or per unit area without a thickness,
            broadcast against the grid's (y, x).
        grid: The grid ``volume`` was built on.
        volume: The region.
        thickness: H (m), broadcast against q; 1 if not given.

    Returns:
        The integral (units of q m², or of q m³ with a thickness), q without
        (y, x). A NaN inside the region makes it NaN; no cell outside is read.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, or ``volume`` is
            of another grid.

    Examples:
        >>> import gridax, numpy as np
        >>> from diagnostix.budgets import ControlVolume
        >>> grid = gridax.CartesianAGrid2D.from_interior(4, 4, 8.0, 8.0)
        >>> inside = np.zeros((4, 4), dtype=bool)
        >>> inside[1:3, 1:3] = True
        >>> box = ControlVolume.from_region(inside, grid)
        >>> float(region_integral(3.0, grid, box))
        48.0
    """
    require_volume(volume, grid)
    others = (q,) if thickness is None else (q, thickness)
    xp, (held, *height) = grid_inputs(grid, broadcast=others)
    if height:
        held = held * height[0]
    inside = on_device(xp, volume.inside, held)
    # 0 outside, also where q is not finite
    kept = xp.where(inside, held, xp.zeros_like(held))
    area = metric_like(xp, gridax.area_weights(grid), held)
    return xp.sum(kept * area, axis=(-2, -1), dtype=held.dtype)


def budget(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    volume: ControlVolume,
    *,
    time: ArrayLike,
    source: ArrayLike | None = None,
    thickness: ArrayLike | None = None,
    tendency: Tendency = "centred",
) -> Budget:
    """The budget of a quantity over a control volume, term by term.

    r(t) = d/dt ∫ q H dA + ∮ q H u·n dl − ∫ S dA.

    The volume budget has q = 1, heat q = θ and S = flux / (ρ₀ c_p), kinetic
    energy q = ½ (u² + v²), and a tracer its concentration and its emissions.
    For model output, the snapshots of q give the tendency and the time means of
    the fluxes the boundary term; the function cannot check which it is given.

    Args:
        q: The quantity per unit volume, or per unit area without a thickness,
            (T, …, y, x), or anything that broadcasts against u.
        u: Eastward velocity (m s⁻¹), (T, …, y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: The grid ``volume`` was built on.
        volume: The region and its faces.
        time: t (s), (T,), increasing.
        source: S, per unit area and second, broadcast against u; a sink is
            negative, and the thickness does not multiply it. 0 if not given.
        thickness: H (m), broadcast against u; 1 if not given.
        tendency: ``"centred"``, the second-order difference on the given times,
            one-sided at the first and last; or ``"forward"``,
            (q^{n+1} − q^n)/Δt beside the fluxes and the source at n, with T − 1
            times, which closes a forward Euler run to round-off.

    Returns:
        The terms (units of q m² s⁻¹, or of q m³ s⁻¹ with a thickness), (T, …).

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: If the grid is not collocated and 2-D, a velocity does
            not end in its shape, ``volume`` is of another grid, ``time`` is not
            the first axis of u or has fewer than two, or ``tendency`` is not
            one of the two.

    Examples:
        A tracer that grows by 2 a second everywhere, at rest:

        >>> import gridax, numpy as np
        >>> from diagnostix.budgets import ControlVolume
        >>> grid = gridax.CartesianAGrid2D.from_interior(4, 4, 8.0, 8.0)
        >>> inside = np.zeros((4, 4), dtype=bool)
        >>> inside[1:3, 1:3] = True
        >>> box = ControlVolume.from_region(inside, grid)
        >>> t = np.asarray([0.0, 1.0, 3.0])
        >>> rest = np.zeros((3, 4, 4))
        >>> q = 2.0 * t[:, None, None] + rest
        >>> terms = budget(q, rest, rest, grid, box, time=t, source=1.5)
        >>> terms.tendency, terms.source, terms.residual
        (array([32., 32., 32.]), array([24., 24., 24.]), array([8., 8., 8.]))
    """
    xp, fields = _budget_inputs(q, u, v, grid, time, source, thickness, tendency)
    q, u, v, added, height, t = fields
    held = region_integral(q, grid, volume, thickness=height)
    out = boundary_flux(q, u, v, grid, volume, thickness=height)
    gained = region_integral(added, grid, volume)
    change, now = _tendency(xp, held, t, tendency)
    out, gained = out[now, ...], gained[now, ...]
    return Budget(
        tendency=change,
        boundary_flux=out,
        source=gained,
        # r = d/dt ∫ q H dA + ∮ q H u·n dl − ∫ S dA
        residual=change + out - gained,
    )


def cell_residual(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    *,
    time: ArrayLike,
    source: ArrayLike | None = None,
    thickness: ArrayLike | None = None,
    tendency: Tendency = "centred",
) -> Any:
    """What is left of the budget of each cell, per unit area.

    r = ∂(q H)/∂t + ∇ · (q H u) − S, with the `flux_divergence`. Its area sum
    over a region is the residual of the `budget` there, to round-off.

    Args:
        q: The quantity, as in `budget`.
        u: Eastward velocity (m s⁻¹), (T, …, y, x).
        v: Northward velocity (m s⁻¹), as u.
        grid: A `gridax.CartesianAGrid2D` or `gridax.SphericalAGrid2D`.
        time: t (s), (T,), increasing.
        source: S, as in `budget`.
        thickness: H (m), as in `budget`.
        tendency: ``"centred"`` or ``"forward"``, as in `budget`.

    Returns:
        r (units of q s⁻¹, or of q m s⁻¹ with a thickness), (T, …, y, x), NaN on
        the edge cells of an axis that does not wrap.

    Raises:
        BackendError: If a field is a DataArray or of another library.
        ValidationError: As `budget`.

    Examples:
        >>> import gridax, numpy as np
        >>> grid = gridax.CartesianAGrid2D.from_interior(3, 3, 3.0, 3.0)
        >>> t = np.asarray([0.0, 1.0, 3.0])
        >>> rest = np.zeros((3, 3, 3))
        >>> q = 2.0 * t[:, None, None] + rest
        >>> cell_residual(q, rest, rest, grid, time=t, source=1.5)[:, 1, 1]
        array([0.5, 0.5, 0.5])
    """
    xp, fields = _budget_inputs(q, u, v, grid, time, source, thickness, tendency)
    q, u, v, added, height, t = fields
    held = q if height is None else q * height
    out = flux_divergence(q, u, v, grid, thickness=height)
    change, now = _tendency(xp, held, t, tendency)
    # r = ∂(q H)/∂t + ∇ · (q H u) − S
    return change + out[now, ...] - added[now, ...]


def _budget_inputs(
    q: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    grid: Any,
    time: ArrayLike,
    source: ArrayLike | None,
    thickness: ArrayLike | None,
    tendency: str,
) -> tuple[ModuleType, list[Any]]:
    """The namespace, and q, u, v, S, H and t, the fields all (T, …, y, x)."""
    if tendency not in ("centred", "forward"):
        msg = f"'tendency' must be 'centred' or 'forward', got {tendency!r}"
        raise ValidationError(msg)
    added = 0.0 if source is None else source
    others = (q, added) if thickness is None else (q, added, thickness)
    xp, (u, v, *rest) = grid_inputs(grid, u, v, broadcast=others, params=(time,))
    u, v, q, added, *height = xp.broadcast_arrays(u, v, *rest)
    shape = np.shape(time)
    if len(shape) != 1 or u.ndim < 3 or shape[0] != u.shape[0] or shape[0] < 2:
        msg = (
            "'time' must be 1-D, the first axis of the fields and at least two, got "
            f"shape {shape} for fields of shape {tuple(u.shape)}"
        )
        raise ValidationError(msg)
    t, _ = time_offsets(xp, time, u)
    return xp, [q, u, v, added, height[0] if height else None, t]


def _tendency(xp: ModuleType, held: Any, t: Any, tendency: str) -> tuple[Any, slice]:
    """d/dt of (T, …) values, and which of the T times it is at."""
    last = xp.moveaxis(held, 0, -1)
    if tendency == "centred":
        change, now = uneven_difference(xp, last, t), slice(None)
    else:
        # (q^{n+1} − q^n) / Δt
        change = (last[..., 1:] - last[..., :-1]) / (t[1:] - t[:-1])
        now = slice(None, -1)
    return xp.moveaxis(change, -1, 0), now
