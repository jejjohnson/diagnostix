# EOFs

The patterns that explain most of an anomaly field's variance per unit area.

```python
import gridax
from diagnostix.climate import anomaly
from diagnostix.eofs import eofs, project, reconstruct

# x: (T, Ny, Nx), without its annual cycle
departure = anomaly(x, cycle, day, axis=0)
area = gridax.area_weights(grid)
out = eofs(departure, n_modes=10, weights=area, valid=ocean)
out.modes  # (mode, Ny, Nx), NaN on land
out.principal_components  # (T, mode)
out.explained_variance  # the share of each mode
```

Without weights a longitude–latitude grid counts the cells near the poles as much as those at the equator. The modes are orthonormal under the weights, ∑ w eₘ eₙ = δₘₙ, so a field's amplitudes are its weighted sums with them:

```python
# the amplitudes in new data, and the field those modes make of it
a = project(new, out, weights=area, valid=ocean)
smooth = reconstruct(out, a)
```

- **Nothing is centred.** The anomaly is the caller's: a mean or a trend left in is the first mode.
- **Signs.** The loading of each mode that is largest in size is positive.
- **Valid cells** are the same at every time, and concrete: `valid` may not be traced. A NaN in a valid cell makes every output NaN.
- **Weights are not normalised.** c times the weights gives modes over √c and principal components times it; the shares and the reconstruction do not change. `project` needs the weights and the mask the modes were made with.
- **Gradients** of the shares are finite for distinct singular values. Those of the modes grow without bound as two of them meet.
- **Large fields.** The cost is a full decomposition of T × C numbers. A randomised one, as gaussx has, is the caller's.

::: diagnostix.eofs
    options:
      show_root_heading: false
      show_source: false
