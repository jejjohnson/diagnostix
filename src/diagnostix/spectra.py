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
from diagnostix._src.spectra.wavelet import (
    WaveletTransform,
    inverse_wavelet_transform,
    wavelet_significance,
    wavelet_transform,
)
from diagnostix._src.spectra.wavelet_2d import LocalSpectrum, wavelet_power_2d


__all__ = [
    "Coherence",
    "CrossSpectrum",
    "LocalSpectrum",
    "RotarySpectrum",
    "SpectralFlux",
    "Spectrum",
    "Spectrum2D",
    "WaveletTransform",
    "coherence",
    "cross_spectrum",
    "enstrophy_flux",
    "integral_scale",
    "inverse_wavelet_transform",
    "isotropic_spectrum",
    "kinetic_energy_flux",
    "power_spectrum",
    "power_spectrum_2d",
    "rotary_spectrum",
    "spectral_slope",
    "wavelet_power_2d",
    "wavelet_significance",
    "wavelet_transform",
]
