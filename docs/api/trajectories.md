# Trajectories

Scores of trajectories: how far a simulated path is from an observed one, the skill that follows, and how fast particles disperse. They take arrays of positions, such as the `trajectory` that `tracerax.advect.integrate` saves, and need no run.

## Separation and skill

`separation` is the distance between a simulated trajectory and an observed one at each time, `cumulative_separation` its sum over the observed path's length (Liu and Weisberg, 2011), and `trajectory_skill` is one minus that, and zero beyond a tolerance.

```python
from diagnostix.trajectories import cumulative_separation, separation, trajectory_skill

# (…, time, column): time before the columns, and any batch before both
apart = separation(simulated, observed)
score = cumulative_separation(simulated, observed)
skill = trajectory_skill(simulated, observed, tolerance=1.0)
```

- **They are losses too.** On JAX arrays the gradient is finite where the two trajectories coincide, also at a shared starting point. Beyond the tolerance the skill's gradient is zero, so fit with `cumulative_separation`.
- **A missing observation is NaN.** It is left out of both sums, and the length bridges the gap in a straight line or along a great circle. A simulated NaN is not hidden.
- **An observed path that does not move** gives an infinite separation, a skill of 0 and a gradient of 0; `min_length` adds a length to the denominator.
- **dask** has no `take_along_axis`, which the observed path's length needs: compute a dask array first.

## Dispersion

`absolute_dispersion` is the mean squared distance of particles from where they started, and `pair_dispersion` that between the partners of each pair. Both take positions of shape (time, particle, column), the starts first.

```python
import numpy as np
from diagnostix.trajectories import absolute_dispersion, pair_dispersion

spread = absolute_dispersion(trajectory)
gap = pair_dispersion(trajectory, np.array([[0, 1], [2, 3]]))
```

- **On the sphere** (`distance="sphere"`) positions are longitudes and latitudes in degrees, and the distances are great-circle distances (`geophysix.earth.great_circle_distance`), which are right across the date line. A difference of longitudes is not.
- **A position that is NaN is missing** and left out of the mean. A particle that is not released yet is at its start, and counts as one that stays: give it NaN.
- **Float32** resolves about 1e-7 rad, 0.6 m on the Earth, so a great-circle distance of a few kilometres is good to about 1e-3 of itself.

::: diagnostix.trajectories
