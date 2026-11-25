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

::: diagnostix.skill
    options:
      show_root_heading: false
      show_source: false
