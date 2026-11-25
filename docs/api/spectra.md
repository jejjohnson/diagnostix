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

::: diagnostix.spectra
    options:
      show_root_heading: false
      show_source: false
