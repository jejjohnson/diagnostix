# Spectra

How the variance of a field is spread over scales.

Every spectrum is a density in cycles per unit length, normalised so that its sum times the wavenumber step is the variance. Wavenumbers are not multiplied by 2π.

```python
from diagnostix.spectra import isotropic_spectrum, power_spectrum, power_spectrum_2d

# x: (…, T), one sample every 3600 s; frequencies in hertz
series = power_spectrum(x, 3600.0)
# ssh: (…, Ny, Nx) on a plane of 2 km by 3 km cells
rings = isotropic_spectrum(ssh, (2e3, 3e3))
# a section in (time, x): the wavenumber–frequency spectrum
section = power_spectrum_2d(hovmoller, (86400.0, 25e3))
```

The spectrum of a figure is also that of a loss: it runs on JAX, under `jit` and `grad`, with the spacing, the taper and the rings static.

```python
def loss(prediction):  # relative to the truth, ring by ring
    error = isotropic_spectrum(prediction - truth, 2e3).power
    return jnp.mean(error / isotropic_spectrum(truth, 2e3).power)
```

- **The grid is uniform.** A field on longitude and latitude goes onto a local plane first, or covers a small domain: nothing is projected here.
- **Before the transform** the least-squares line, or plane, is removed and a periodic Hann taper applied, and the density is divided by the taper's mean square. For a field that is periodic, pass `window=None, detrend=None`. Any other taper is a host array, as `scipy.signal.windows` makes.
- **Rings** of the isotropic spectrum are centred on whole multiples of Δk, and are sums of the modes in them, not means. Those beyond the smaller of the two Nyquist wavenumbers are incomplete.
- **Gaps.** A NaN makes the whole spectrum NaN. Filling a gap changes the spectrum, so it is the caller's.

## Two series

`coherence` is |⟨x̂ ŷ\*⟩|² over the product of the two powers, and the averages ⟨·⟩ are the whole of it: without one it is 1 for any two series, so `coherence` asks for segments, for realisations to average over, or both. Of M independent transforms of independent noise it is 1/M, not 0.

```python
from diagnostix.spectra import coherence, cross_spectrum, rotary_spectrum

# x, y: (…, T); Welch segments of 256 samples, half overlapping
c = coherence(x, y, 3600.0, segment=256)
# ssh_a, ssh_b: (member, Ny, Nx); over members and rows, whole rows as segments
c = coherence(ssh_a, ssh_b, 2e3, average=(0, 1))
# the lag at each frequency; positive where x leads y (SciPy's csd is the conjugate)
p = cross_spectrum(x, y, 3600.0, segment=256)
lag = jnp.angle(p.cross) / (2 * jnp.pi * p.wavenumber)
# a current's clockwise and counter-clockwise variance
r = rotary_spectrum(u, v, 3600.0)
```

`rotary_spectrum` splits the variance of u + iv by the way it turns. The mean is half in each part, and the one Nyquist bin of an even series is clockwise, so both parts sum to the variance of u plus that of v.

## Fluxes, slopes and scales

`kinetic_energy_flux` and `enstrophy_flux` say how much advection gives to each ring of wavenumbers (`transfer`), and how much it carries through the lower edge of the ring (`flux`), positive towards small scales.

```python
from diagnostix.spectra import integral_scale, kinetic_energy_flux, spectral_slope

# u, v: (time, Ny, Nx) of a doubly periodic box; x and y grow with the index
out = kinetic_energy_flux(u, v, (2e3, 2e3))
mean_flux = out.flux.mean(axis=0)
# s in P ∝ kˢ, between 200 km and 20 km
slope = spectral_slope(isotropic_spectrum(ssh, 2e3), 1 / 200e3, 1 / 20e3)
scale = integral_scale(isotropic_spectrum(u, 2e3))
```

The products are formed on a grid 3/2 as fine, so nothing aliases and the transfer of a non-divergent velocity sums to zero. That holds for a periodic box only: a taper makes the velocity divergent, so none is applied unless asked for.

A slope is `NaN` where its band holds fewer than two bins, or a bin without power. `integral_scale` is a wavelength: the inverse of a wavenumber in cycles per unit length.

## Wavelets

A spectrum of a whole record hides when its variance came. `wavelet_transform` is the continuous transform of Torrence and Compo (1998), with a Morlet, a Paul or a derivative-of-Gaussian mother.

```python
from diagnostix.spectra import (
    inverse_wavelet_transform,
    wavelet_power_2d,
    wavelet_significance,
    wavelet_transform,
)

# x: (…, T), monthly anomalies; scales from 2 months to 16 years, 8 to the octave
scales = 2.0 * 2.0 ** (np.arange(6.6 * 8) / 8)
out = wavelet_transform(x, 1.0, scales)
# where the power is above that of red noise, and away from the ends
threshold = wavelet_significance(out, x.var(-1), lag1=0.7)
shown = (out.power > threshold[..., None]) & out.inside_cone
# the part of the series with periods of 2 to 7 years
enso = inverse_wavelet_transform(out, band=(24.0, 84.0))
```

Under JAX only the coefficients are traced: the scales, the mother and the cone are static. A series that is not `periodic` is padded with zeros to twice its length, so nothing wraps, and `inside_cone` says which coefficients do not feel the ends. The mean is at no scale: it is removed first, and the inverse does not give it back.

`wavelet_power_2d` is an isotropic spectrum about each point: its integral over wavenumber is the variance there, and its mean over the plane is an `isotropic_spectrum` smoothed over about a quarter of k.

```python
# ssh: (…, Ny, Nx); wavelengths of 40 to 400 km on a 5 km grid
local = wavelet_power_2d(ssh, 5e3, 40e3 * 2.0 ** np.arange(0, 3.4, 0.25))
power = np.where(local.inside_cone, local.power, np.nan)
```

::: diagnostix.spectra
    options:
      show_root_heading: false
      show_source: false
