# Kinematics

What a horizontal velocity field does, on a collocated gridax grid: a plane (`gridax.CartesianAGrid2D`) or a sphere (`gridax.SphericalAGrid2D`). A field ends in the grid's (y, x), with any leading axes.

Functions here take plain arrays. A `mask` is a `gridax.Mask2D`: a dry cell is never read, and is NaN in the output.

The vorticity, the divergence, the strain, the Okubo–Weiss parameter and the geostrophic velocity are gridax's (`gridax.curl`, `divergence`, `shear_strain`, `tensor_strain`, `okubo_weiss_parameter`, `geostrophic_velocity`), on the same grids.

```python
import gridax
from geophysix.rotation import coriolis

from diagnostix.kinematics import frontogenesis, geostrophic_residual

grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon)
# sst, u, v: (T, Ny, Nx) → F: (T, Ny, Nx)
fronts = frontogenesis(sst, u, v, grid, mask=mask)
# f: (Ny, 1), one value per row
f = coriolis(lat)[:, None]
r_u, r_v = geostrophic_residual(u, v, ssh, grid, f=f, mask=mask)
```

::: diagnostix.kinematics
    options:
      show_root_heading: false
      show_source: false
