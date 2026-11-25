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

::: diagnostix.spectra
    options:
      show_root_heading: false
      show_source: false
