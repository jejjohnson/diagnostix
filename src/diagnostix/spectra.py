"""How the variance of a field is spread over scales.

Every spectrum is a density, normalised so that it sums to the variance.
"""

from __future__ import annotations

from diagnostix._src.spectra.cross import (
    Coherence,
    CrossSpectrum,
    coherence,
    cross_spectrum,
)
from diagnostix._src.spectra.flux import (
    SpectralFlux,
    enstrophy_flux,
    kinetic_energy_flux,
)
from diagnostix._src.spectra.power import (
    Spectrum,
    Spectrum2D,
    isotropic_spectrum,
    power_spectrum,
    power_spectrum_2d,
)
from diagnostix._src.spectra.rotary import RotarySpectrum, rotary_spectrum
from diagnostix._src.spectra.shape import integral_scale, spectral_slope


__all__ = [
    "Coherence",
    "CrossSpectrum",
    "RotarySpectrum",
    "SpectralFlux",
    "Spectrum",
    "Spectrum2D",
    "coherence",
    "cross_spectrum",
    "enstrophy_flux",
    "integral_scale",
    "isotropic_spectrum",
    "kinetic_energy_flux",
    "power_spectrum",
    "power_spectrum_2d",
    "rotary_spectrum",
    "spectral_slope",
]
