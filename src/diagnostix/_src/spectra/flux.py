"""Spectral fluxes: what advection carries through each scale."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float
from pipekit_array.functions import floating_inputs

from diagnostix._src.records import register_record
from diagnostix._src.reduce import constants_like
from diagnostix._src.spectra.rings import Rings
from diagnostix._src.spectra.taper import Detrend, Window, checked_spacing, tapered
from diagnostix._typing import ArrayLike


__all__ = ["SpectralFlux", "enstrophy_flux", "kinetic_energy_flux"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class SpectralFlux:
    """What advection gives to each ring of wavenumbers, and carries through it.

    Attributes:
        wavenumber: k, from 0, the centres of the rings (cycles per unit
            length), those of `isotropic_spectrum`.
        transfer: T(kₙ), the gain of ring n per unit time, a sum over the ring,
            with the rings on the last axis.
        flux: Π(kₙ) = ∑ T(kₙ′) over n′ ≥ n, the gain of every scale smaller than
            the lower edge of ring n: positive towards small scales.
    """

    wavenumber: Float[Any, " k"]
    transfer: Float[Any, "*batch k"]
    flux: Float[Any, "*batch k"]


def kinetic_energy_flux(
    u: ArrayLike,
    v: ArrayLike,
    spacing: float | Sequence[float],
    *,
    window: Window = None,
    detrend: Detrend = None,
) -> SpectralFlux:
    """The flux of kinetic energy through each scale, positive towards small ones.

    T(k) = −Re(û(k)* · [(u·∇)u]^(k)) summed over each ring, and Π(kₙ) is the sum
    of T over the rings from n on, with ^ the coefficients of a periodic field.

    The advection is formed in physical space from spectral gradients on a grid
    3/2 as fine, so its modes are the exact sums over triads: nothing aliases,
    and T of a non-divergent velocity sums to 0. For even sizes the Nyquist
    modes, which have no odd derivative, are left out of the velocity.

    The box is periodic. A tapered velocity is divergent, so its transfer does
    not sum to 0, and a taper makes no periodic box out of another.

    Args:
        u: The x component (m s⁻¹), ending in (y, x), both growing with the
            index: flip a field whose y falls.
        v: The y component (m s⁻¹), broadcast against ``u``.
        spacing: (Δy, Δx) (m), or one number for both.
        window: As in `power_spectrum_2d`, but none if not given.
        detrend: As in `power_spectrum_2d`, but nothing if not given.

    Returns:
        The transfer and the flux (m² s⁻³), with the rings on the last axis. A
        NaN in a field makes both NaN.

    Raises:
        BackendError: If a component is a DataArray or of another library.
        ValidationError: As `power_spectrum_2d`.

    Examples:
        >>> import numpy as np
        >>> y, x = 2.0 * np.pi * np.mgrid[:16, :16] / 16.0
        >>> # one triad: ψ = cos y + cos(x + y) + cos(x + 2y + 1)
        >>> u = np.sin(y) + np.sin(x + y) + 2.0 * np.sin(x + 2.0 * y + 1.0)
        >>> v = -np.sin(x + y) - np.sin(x + 2.0 * y + 1.0)
        >>> out = kinetic_energy_flux(u, v, 2.0 * np.pi / 16.0)
        >>> np.round(out.transfer[:4], 5) + 0.0  # |k|: 1 and √2, then √5
        array([ 0.     ,  0.13508, -0.13508,  0.     ])
        >>> np.round(out.flux[:4], 5) + 0.0
        array([ 0.     ,  0.     , -0.13508,  0.     ])
    """
    xp, modes, (u_hat, v_hat) = _coefficients(u, v, spacing, window, detrend)
    return modes.transfer(xp, u_hat, v_hat, (u_hat, v_hat))


def enstrophy_flux(
    u: ArrayLike,
    v: ArrayLike,
    spacing: float | Sequence[float],
    *,
    window: Window = None,
    detrend: Detrend = None,
) -> SpectralFlux:
    """The flux of enstrophy through each scale, positive towards small ones.

    T(k) = −Re(ζ̂(k)* [u·∇ζ]^(k)) summed over each ring, with ζ = ∂v/∂x − ∂u/∂y
    the vorticity; otherwise as `kinetic_energy_flux`. Enstrophy is ½ζ².

    Args:
        u: The x component (m s⁻¹), as in `kinetic_energy_flux`.
        v: The y component (m s⁻¹), broadcast against ``u``.
        spacing: (Δy, Δx) (m), or one number for both.
        window: As in `kinetic_energy_flux`.
        detrend: As in `kinetic_energy_flux`.

    Returns:
        The transfer and the flux (s⁻³), with the rings on the last axis.

    Raises:
        BackendError: If a component is a DataArray or of another library.
        ValidationError: As `power_spectrum_2d`.

    Examples:
        >>> import numpy as np
        >>> y, x = 2.0 * np.pi * np.mgrid[:16, :16] / 16.0
        >>> u = np.sin(y) + np.sin(x + y) + 2.0 * np.sin(x + 2.0 * y + 1.0)
        >>> v = -np.sin(x + y) - np.sin(x + 2.0 * y + 1.0)
        >>> out = enstrophy_flux(u, v, 2.0 * np.pi / 16.0)
        >>> np.round(out.transfer[:4], 5) + 0.0
        array([ 0.     ,  0.67538, -0.67538,  0.     ])
    """
    xp, modes, (u_hat, v_hat) = _coefficients(u, v, spacing, window, detrend)
    # ζ̂ = i (k_x v̂ − k_y û)
    zeta_hat = modes.derivative_x(xp, v_hat) - modes.derivative_y(xp, u_hat)
    return modes.transfer(xp, u_hat, v_hat, (zeta_hat,))


def _coefficients(
    u: ArrayLike,
    v: ArrayLike,
    spacing: float | Sequence[float],
    window: Window,
    detrend: Detrend,
) -> tuple[ModuleType, _Modes, tuple[Any, Any]]:
    """The namespace, the modes of the grid, and the coefficients of u and v."""
    steps = checked_spacing(spacing, 2)
    xp, (u, v) = floating_inputs(u, v)
    u, v = xp.broadcast_arrays(u, v)
    u, _ = tapered(xp, u, 2, window, detrend)
    v, _ = tapered(xp, v, 2, window, detrend)
    modes = _Modes(tuple(u.shape[-2:]), (steps[0], steps[1]))
    return xp, modes, (modes.forward(xp, u), modes.forward(xp, v))


@dataclasses.dataclass(frozen=True)
class _Modes:
    """The modes of a real periodic field below its Nyquist wavenumbers.

    The coefficients of M_y = (Ny − 1)//2 and M_x likewise are (…, 2 M_y + 1,
    M_x + 1): rows 0 … M_y, −M_y … −1, and columns 0 … M_x.
    """

    shape: tuple[int, int]
    spacing: tuple[float, float]

    @property
    def largest(self) -> tuple[int, int]:
        """M_y and M_x."""
        return (self.shape[0] - 1) // 2, (self.shape[1] - 1) // 2

    @property
    def fine(self) -> tuple[int, int]:
        """The sizes of a grid on which a product of two fields does not alias."""
        # at least 3 M + 1
        return (3 * self.shape[0] + 1) // 2, (3 * self.shape[1] + 1) // 2

    def forward(self, xp: ModuleType, field: Any) -> Any:
        """The coefficients of a field on any grid that resolves these modes."""
        ny, nx = field.shape[-2:]
        rows, columns = self.largest
        z = xp.fft.rfftn(field, axes=(-2, -1)) / (ny * nx)
        z = z[..., : columns + 1]
        return xp.concat([z[..., : rows + 1, :], z[..., ny - rows :, :]], axis=-2)

    def half_plane(self, xp: ModuleType, z: Any, shape: tuple[int, int]) -> Any:
        """Coefficients in the half plane of a grid's ``rfftn``, 0 elsewhere."""
        rows, _ = self.largest
        above = xp.zeros_like(z[..., :1, :])
        gap = (*z.shape[:-2], shape[0] - z.shape[-2], z.shape[-1])
        z = xp.concat(
            [z[..., : rows + 1, :], xp.broadcast_to(above, gap), z[..., rows + 1 :, :]],
            axis=-2,
        )
        beyond = xp.zeros_like(z[..., :1])
        gap = (*z.shape[:-1], shape[1] // 2 + 1 - z.shape[-1])
        return xp.concat([z, xp.broadcast_to(beyond, gap)], axis=-1)

    def on_fine_grid(self, xp: ModuleType, z: Any) -> Any:
        """The field of some coefficients on the fine grid."""
        ny, nx = self.fine
        half = self.half_plane(xp, z, self.fine)
        return xp.fft.irfftn(half, s=self.fine, axes=(-2, -1)) * (ny * nx)

    def derivative_x(self, xp: ModuleType, z: Any) -> Any:
        """The coefficients i k_x z, with k_x in radians per unit length."""
        (_, nx), (_, dx) = self.shape, self.spacing
        k = 2.0 * np.pi * np.arange(self.largest[1] + 1) / (nx * dx)
        return z * constants_like(xp, k, xp.real(z)) * 1j

    def derivative_y(self, xp: ModuleType, z: Any) -> Any:
        """The coefficients i k_y z, with k_y in radians per unit length."""
        (ny, _), (dy, _) = self.shape, self.spacing
        rows = self.largest[0]
        index = np.concatenate([np.arange(rows + 1), np.arange(-rows, 0)])
        k = 2.0 * np.pi * index / (ny * dy)
        return z * constants_like(xp, k[:, None], xp.real(z)) * 1j

    def transfer(
        self, xp: ModuleType, u_hat: Any, v_hat: Any, advected: Sequence[Any]
    ) -> SpectralFlux:
        """The transfer and the flux of ½|q|², for the coefficients of q."""
        u, v = self.on_fine_grid(xp, u_hat), self.on_fine_grid(xp, v_hat)
        gained = []
        for q_hat in advected:
            q_x = self.on_fine_grid(xp, self.derivative_x(xp, q_hat))
            q_y = self.on_fine_grid(xp, self.derivative_y(xp, q_hat))
            advection_hat = self.forward(xp, u * q_x + v * q_y)
            # T = −Re(q̂* [u·∇q]^)
            gained.append(-xp.real(xp.conj(q_hat) * advection_hat))
        total = sum(gained[1:], gained[0])
        rings = Rings.of_grid(self.shape, self.spacing)
        transfer = rings.sum(xp, self.half_plane(xp, total, self.shape))
        # Π(kₙ) = ∑ T over n′ ≥ n
        flux = xp.flip(xp.cumulative_sum(xp.flip(transfer, axis=-1), axis=-1), axis=-1)
        return SpectralFlux(
            wavenumber=constants_like(xp, rings.wavenumber, u),
            transfer=transfer,
            flux=flux,
        )
