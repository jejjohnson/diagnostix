# Budgets

How much of a quantity a region holds, gains and lets out, on a collocated gridax grid: a plane (`gridax.CartesianAGrid2D`) or a sphere (`gridax.SphericalAGrid2D`).

A `ControlVolume` is a region made of cells, with the faces that bound it. It is built once, from a concrete region, and is static under JAX. The region may have holes, have several parts and cross a periodic seam; it may not touch an edge that does not wrap, where one of its faces would lie outside the data.

The flux through a face is the mean of the product of the two cells beside it, times the length of the face. Every inner face of a region cancels, so the flux through its boundary equals the area sum of `flux_divergence` inside it, to round-off.

```python
import gridax

from diagnostix.budgets import ControlVolume, boundary_flux

grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon)
box = ControlVolume.from_region(around_source, grid)
# enhancement, u, v: (T, Ny, Nx) → Q: (T,)
rate = boundary_flux(enhancement, u, v, grid, box)
```

A missing value on either side of a face of the boundary makes the flux NaN, because a budget with a missing face does not close. Cells away from the boundary are not read.

::: diagnostix.budgets
    options:
      show_root_heading: false
      show_source: false
