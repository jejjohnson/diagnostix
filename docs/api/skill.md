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

::: diagnostix.skill
    options:
      show_root_heading: false
      show_source: false
