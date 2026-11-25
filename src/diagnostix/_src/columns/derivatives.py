"""Derivatives along a column, and the stratification they measure."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gridax.analysis
import numpy as np
from geophysix.constants import STANDARD
from geophysix.seawater import LINEAR_EOS
from pipekit_array.arrays import floating_inputs, like

from diagnostix._src.columns.levels import column_inputs, levels_back
from diagnostix._typing import ArrayLike


__all__ = [
    "buoyancy_frequency_squared",
    "lapse_rate",
    "uneven_difference",
    "vertical_derivative",
]


def vertical_derivative(values: ArrayLike, z: ArrayLike, *, axis: int = -1) -> Any:
    """The derivative along a column, second-order accurate on uneven levels.

    With h₋ = z_k − z_{k−1} and h₊ = z_{k+1} − z_k, at an inner level

    f′(z_k) = (h₋ (f_{k+1} − f_k) / h₊ + h₊ (f_k − f_{k−1}) / h₋) / (h₋ + h₊),

    exact for a quadratic, and at an end level the one-sided difference, exact
    for a line.

    Args:
        values: f, with its levels along ``axis``.
        z: The position of each level, increasing or decreasing, 1-D or one
            column per point.
        axis: The axis of the levels.

    Returns:
        ∂f/∂z, the shape of f. A NaN gives NaN at its level and those beside it.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or ``z`` does not
            have those of ``values``.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([0.0, 1.0, 3.0, 7.0])
        >>> vertical_derivative(z**2, z)
        array([ 1.,  2.,  6., 10.])
    """
    xp, (f,), columns, _ = column_inputs((values,), z, axis=axis)
    # gridax's central stencil reads three points
    if np.ndim(z) == 1 and f.shape[-1] > 2:
        _, (_, levels) = floating_inputs(values, z)
        out = gridax.analysis.difference(f, axis=-1, coords=levels)
    else:
        out = uneven_difference(xp, f, columns)
    return levels_back(xp, out, axis)


def uneven_difference(xp: ModuleType, f: Any, z: Any) -> Any:
    """``gridax.analysis.difference`` along the last axis, any positions ``z``.

    It takes positions per column, two points or more, and any array library.
    """
    spacing = z[..., 1:] - z[..., :-1]
    # sₖ = (f_{k+1} − f_k) / (z_{k+1} − z_k), one per interval
    slope = (f[..., 1:] - f[..., :-1]) / spacing
    lower, upper = spacing[..., :-1], spacing[..., 1:]
    # (h₋ s₊ + h₊ s₋) / (h₋ + h₊)
    inner = (lower * slope[..., 1:] + upper * slope[..., :-1]) / (lower + upper)
    return xp.concat([slope[..., :1], inner, slope[..., -1:]], axis=-1)


def buoyancy_frequency_squared(
    density: ArrayLike,
    z: ArrayLike,
    *,
    axis: int = -1,
    reference_density: float = LINEAR_EOS.reference_density,
    gravity: float = STANDARD.standard_gravity,
) -> Any:
    """The squared buoyancy (Brunt–Väisälä) frequency of a Boussinesq column.

    N² = −(g / ρ₀) ∂ρ/∂z, positive where lighter water lies on denser.

    Args:
        density: ρ (kg m⁻³), usually potential density, with its levels along
            ``axis``.
        z: Height (m), upward, in either order, 1-D or one column per point.
            From a depth, positive down, pass ``-depth``.
        axis: The axis of the levels.
        reference_density: ρ₀ (kg m⁻³).
        gravity: g (m s⁻²).

    Returns:
        N² (s⁻²), the shape of ρ; the derivative is `vertical_derivative`'s.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or ``z`` does not
            have those of ``density``.

    Examples:
        >>> import numpy as np
        >>> depth = np.asarray([0.0, 10.0, 30.0])
        >>> rho = 1025.0 + 0.01 * depth
        >>> np.round(buoyancy_frequency_squared(rho, -depth), 9)
        array([9.5675e-05, 9.5675e-05, 9.5675e-05])
    """
    params = (reference_density, gravity)
    xp, (rho, height) = floating_inputs(density, z, params=params)
    slope = vertical_derivative(rho, height, axis=axis)
    g, rho_0 = like(gravity, slope, xp), like(reference_density, slope, xp)
    # N² = −(g / ρ₀) ∂ρ/∂z
    return -(g / rho_0) * slope


def lapse_rate(temperature: ArrayLike, z: ArrayLike, *, axis: int = -1) -> Any:
    """The rate at which temperature falls with height.

    Γ = −∂T/∂z.

    Args:
        temperature: T (K), with its levels along ``axis``. Potential
            temperature gives its own lapse rate, negative where stable.
        z: Height (m), upward, in either order, 1-D or one column per point.
        axis: The axis of the levels.

    Returns:
        Γ (K m⁻¹), the shape of T; the derivative is `vertical_derivative`'s.

    Raises:
        BackendError: If an argument is a DataArray or of another library.
        ValidationError: If there are fewer than two levels, or ``z`` does not
            have those of ``temperature``.

    Examples:
        >>> import numpy as np
        >>> z = np.asarray([0.0, 100.0, 300.0])
        >>> np.round(lapse_rate(288.15 - 0.0065 * z, z), 9)
        array([0.0065, 0.0065, 0.0065])
    """
    # Γ = −∂T/∂z
    return -vertical_derivative(temperature, z, axis=axis)
