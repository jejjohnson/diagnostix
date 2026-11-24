# API Reference

The public API is the set of public modules below, plus the exceptions that `diagnostix.__all__` exports. Private code lives in `diagnostix._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| `diagnostix.kinematics` | Velocity gradient, shear and curvature vorticity, advection, frontogenesis, ageostrophic velocity, geostrophic residual | Wave 1 |
| [`diagnostix.columns`](columns.md) | First crossings, mixed-layer depth, bulk Richardson number, boundary-layer height; N², lapse rate and column integrals follow | Wave 1 |
| `diagnostix.budgets` | Control volumes, boundary fluxes, budgets and their residuals | Wave 2 |
| `diagnostix.climate` | Climatologies, anomalies, trends, running means | Wave 2 |
| `diagnostix.spectra` | Power, isotropic, cross, coherence, rotary and wavelet spectra; spectral fluxes and slopes | Wave 3 |
| `diagnostix.eofs` | Area-weighted EOFs and their reading | Wave 3 |
| `diagnostix.skill` | Pointwise, ensemble, spectral and physical scores | Wave 4 |
| [Type aliases](typing.md) | `ArrayLike` | available |
| [Exceptions](exceptions.md) | `DiagnostixError`, `BackendError`, `ValidationError` | available |

## Package overview

::: diagnostix
    options:
      members: false
      show_root_heading: false
      show_source: false
