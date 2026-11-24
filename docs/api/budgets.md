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

A `budget` puts the tendency, the boundary flux and the source side by side, with the residual that says which of them fails to close. One function serves every quantity: the volume budget has q = 1, the heat budget the temperature, the kinetic-energy budget ½ (u² + v²), and a tracer its concentration. For model output, pass snapshots for the tendency and time means for the fluxes; the function cannot tell them apart.

```python
from diagnostix.budgets import budget

# θ, u, v, H: (T, Ny, Nx); q_net: surface flux (W m⁻²)
heat = budget(
    theta,
    u,
    v,
    grid,
    box,
    time=seconds,
    thickness=layer,
    source=q_net / (rho_0 * c_p),
)
heat.residual  # (T,), 0 where the budget closes
```

A missing value on either side of a face of the boundary makes the flux NaN, because a budget with a missing face does not close. Cells away from the boundary are not read.

::: diagnostix.budgets
    options:
      show_root_heading: false
      show_source: false
