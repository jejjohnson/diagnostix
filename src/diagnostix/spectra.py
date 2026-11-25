"""How the variance of a field is spread over scales.

Every spectrum is a density, normalised so that it sums to the variance.
"""

from __future__ import annotations

from diagnostix._src.spectra.power import (
    Spectrum,
    Spectrum2D,
    isotropic_spectrum,
    power_spectrum,
    power_spectrum_2d,
)


__all__ = [
    "Spectrum",
    "Spectrum2D",
    "isotropic_spectrum",
    "power_spectrum",
    "power_spectrum_2d",
]
