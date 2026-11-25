# Skill

How good a prediction is, as functions that can be losses.

## Pointwise scores

Each takes weights and a mask, and pools the axes it is told to, every one by default.

```python
import gridax
from diagnostix.skill import correlation, nrmse_score, rmse

# prediction, reference: (T, Ny, Nx)
area = gridax.area_weights(grid)
# one number per time
per_time = rmse(prediction, reference, weights=area, valid=ocean, axis=(-2, -1))
# one map
per_cell = correlation(prediction, reference, valid=ocean, axis=0)
# one number for everything
score = nrmse_score(prediction, reference, weights=area, valid=ocean)
```

- **The same cells in every moment.** `valid` says which cells of both fields count (True = use). A NaN in a valid cell gives NaN: a gap is said by the mask, not by the NaN.
- **Pooling.** The RMSE over space and time is not the mean over time of the RMSEs over space.
- **A perfect prediction** has a gradient of 0, also for `rmse`, whose square root has none there by itself.
- **A constant reference** has no correlation (NaN), and `r2_score` and `nrmse_score` of 1 for a perfect prediction and −∞ for any other.

## Ensembles

Each score has the value it should have for a reliable ensemble of any size, one whose members and reference are draws of one distribution.

```python
from diagnostix.skill import coverage, crps, rank_histogram, spread_skill
from pipekit_array.weighted import weighted_mean

# ensemble: (E, T, Ny, Nx), reference: (T, Ny, Nx)
score = weighted_mean(crps(ensemble, reference), weights=area, valid=ocean)
ratio = spread_skill(ensemble, reference, weights=area, valid=ocean).ratio  # 1
share = coverage(ensemble, reference, level=0.9, weights=area, valid=ocean)  # 0.9
counts = rank_histogram(ensemble, reference, valid=ocean)  # flat
```

- **`crps` and `energy_score` are fair**: they divide the sum over pairs by E (E − 1), so on average a small ensemble scores as well as a large one. `fair=False` divides by E², and a reliable 5-member ensemble then scores 1.2 times worse than it is.
- **`spread_skill`** has the factor ((E + 1)/E)^½ for the error of a mean of E members.
- **`rank_histogram`** shares ties evenly over the tied slots, so rain, which is mostly 0, does not pile up in slot 0.
- **`coverage`** works on the same slots, not on interpolated quantiles, so its expected value is the level for any number of members. The interval between interpolated 5 % and 95 % quantiles of 5 members holds 62 % of the references.
- **`energy_score`** is the CRPS of whole vectors or fields: it sees whether their elements vary together, which the mean of the CRPS does not.
- **`reliability`** bins probabilities against events: calibrated ones lie on the diagonal.

Counts are piecewise constant, so the gradients of ranks and coverage are 0.

## By scale

`psd_score` is 1 minus the power of the error over that of the reference: 1 where a scale is right, 0 where the error is as large as the signal. `resolved_scale` is the wavelength where it first falls to one half.

```python
from diagnostix.skill import band_rmse, psd_score, resolved_scale

# ssh maps: (T, Ny, Nx) on a 10 km grid; spectra averaged over time first
score = psd_score(maps, truth, 10e3, average=0)
wavelength = resolved_scale(score)  # m
# a section in (T, X): the spectrum along x
score = psd_score(section, truth_section, 10e3, isotropic=False, average=0)
# the error at scales above 200 km, from 200 to 50 km, and below
bands = [(0, 1 / 200e3), (1 / 200e3, 1 / 50e3), (1 / 50e3, np.inf)]
by_band = band_rmse(maps, truth, 10e3, bands)
```

The resolved wavelength is interpolated in log k, and is `NaN` where the score never falls to the level. The squares of `band_rmse` over bands that cover every wavenumber add up to the mean square error; it filters a periodic box and tapers nothing.

## Physical consistency

A low RMSE can hide a state that no ocean could be in.

```python
from diagnostix.skill import (
    density_inversion_fraction,
    geostrophic_imbalance,
    rms_divergence,
)

# 0 for a geostrophic flow, about 1 where the pressure has nothing to do with it
imbalance = geostrophic_imbalance(u, v, ssh, grid, f=f, mask=mask)
divergence = rms_divergence(u, v, grid, mask=mask)  # s⁻¹
# the share of each column's levels that is upside down
inversions = density_inversion_fraction(rho, -depth, axis=0, valid=wet)
```

`geostrophic_imbalance` and `rms_divergence` are means over the wet cells weighted by area, on a collocated grid, and their gradients are 0 at a balanced and at a non-divergent flow. `qg_balance_residual` compares a potential vorticity with that of its own streamfunction, on the Cartesian C-grid of `gridax.qg_potential_vorticity`. A resting state has no imbalance (`NaN`).

## Two methods

Whether one method's errors are smaller by more than chance, with the correlation of the errors of an h-step forecast counted:

```python
import scipy.stats
from diagnostix.skill import diebold_mariano

# errors: (method, T), of forecasts of 3 steps
out = diebold_mariano(errors[0], errors[1], horizon=3)
p_value = 2 * scipy.stats.t.sf(abs(out.statistic), out.degrees_of_freedom)
```

The statistic is positive where the first method is worse. SciPy is not a dependency, so the p-value is the caller's line.

::: diagnostix.skill
    options:
      show_root_heading: false
      show_source: false
