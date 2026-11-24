# Kinematics

What a horizontal velocity field does, on a collocated gridax grid: a plane (`gridax.CartesianAGrid2D`) or a sphere (`gridax.SphericalAGrid2D`). A field ends in the grid's (y, x), with any leading axes.

Functions here take plain arrays. A `mask` is a `gridax.Mask2D`: a dry cell is never read, and is NaN in the output.

The vorticity, the divergence, the strain, the Okubo–Weiss parameter and the geostrophic velocity are gridax's (`gridax.curl`, `divergence`, `shear_strain`, `tensor_strain`, `okubo_weiss_parameter`, `geostrophic_velocity`), on the same grids.

```python
import gridax

from diagnostix.kinematics import shear_vorticity

grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon)
# u, v: (T, Ny, Nx) → ζ_s: (T, Ny, Nx)
zeta_s = shear_vorticity(u, v, grid, mask=mask)
```

::: diagnostix.kinematics
    options:
      show_root_heading: false
      show_source: false
