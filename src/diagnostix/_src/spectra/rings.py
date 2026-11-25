"""Rings of the wavenumber plane, and sums over them."""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Float

from diagnostix._src.groups import Groups
from diagnostix._src.reduce import constants_like
from diagnostix._src.spectra.taper import one_sided


__all__ = ["Rings"]


@dataclasses.dataclass(frozen=True, eq=False)
class Rings:
    """The modes of a real field's transform, grouped by the size of k.

    Ring n holds the modes with |k| within Δk/2 of n Δk, where Δk is the smaller
    spacing of the two wavenumbers. Binned by the floor of |k|/Δk instead, a mode
    of integer radius falls one ring short wherever the division rounds down.

    Only the half plane of ``rfftn`` is kept, so a mode between the mean and the
    Nyquist of the last axis stands for two.

    Attributes:
        groups: The ring of each mode of the half plane.
        multiplicity: 1 or 2, the modes that each of the last axis stands for.
        wavenumber: n Δk of each ring (cycles per unit length).
        step: Δk.
        cell: Δk_x Δk_y.
    """

    groups: Groups
    multiplicity: Float[np.ndarray, " kx"]
    wavenumber: Float[np.ndarray, " ring"]
    step: float
    cell: float

    @classmethod
    def of_grid(cls, shape: tuple[int, int], spacing: tuple[float, float]) -> Rings:
        """The rings of a grid of (Ny, Nx) points with spacings (Δy, Δx)."""
        (ny, nx), (dy, dx) = shape, spacing
        step_y, step_x = 1.0 / (ny * dy), 1.0 / (nx * dx)
        step = min(step_y, step_x)
        # |k|/Δk from integers, so that a radius that is one stays exact
        rows = np.fft.fftfreq(ny, 1.0 / ny)[:, None] * (step_y / step)
        columns = np.fft.rfftfreq(nx, 1.0 / nx) * (step_x / step)
        ring = np.rint(np.hypot(rows, columns)).astype(np.int64)
        n_rings = int(ring.max()) + 1
        return cls(
            groups=Groups.from_labels(ring.ravel(), n_rings),
            multiplicity=one_sided(nx),
            wavenumber=step * np.arange(n_rings),
            step=step,
            cell=step_y * step_x,
        )

    def sum(self, xp: ModuleType, values: Any) -> Any:
        """The sum over each ring of a value per mode of the half plane.

        Args:
            xp: The namespace.
            values: One per mode, even in k: (…, Ny, Nx//2 + 1).

        Returns:
            The sums over both halves of the plane, (…, ring).
        """
        both = values * constants_like(xp, self.multiplicity, values)
        flat = xp.reshape(both, (*both.shape[:-2], -1))
        return self.groups.sum(flat)
