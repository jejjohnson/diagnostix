# API Reference

The public API is the set of public modules below, plus the exceptions that `diagnostix.__all__` exports. Private code lives in `diagnostix._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| [`diagnostix.kinematics`](kinematics.md) | Velocity gradient, shear and curvature vorticity, advection, frontogenesis, ageostrophic velocity, geostrophic residual | available |
| [`diagnostix.columns`](columns.md) | Vertical derivative, N², lapse rate, column integrals, first crossings, mixed-layer depth, bulk Richardson number, boundary-layer height | available |
| [`diagnostix.budgets`](budgets.md) | Control volumes, face and boundary fluxes, the flux divergence, region integrals, budgets and their residuals | available |
| [`diagnostix.climate`](climate.md) | Calendar days, climatologies, harmonic fits and smoothing, anomalies, linear trends, running means | available |
| [`diagnostix.spectra`](spectra.md) | Power, isotropic, cross, rotary and wavelet spectra, coherence, spectral fluxes, slopes and scales | available |
| [`diagnostix.eofs`](eofs.md) | Area-weighted EOFs, projection and reconstruction | available |
| [`diagnostix.skill`](skill.md) | Pointwise, ensemble, spectral and physical scores, the Diebold–Mariano statistic | available |
| [`diagnostix.trajectories`](trajectories.md) | Separation of a trajectory from an observed one, its normalised cumulative form, skill, absolute and pair dispersion | available |
| [Type aliases](typing.md) | `ArrayLike` | available |
| [Exceptions](exceptions.md) | `DiagnostixError`, `BackendError`, `ValidationError` | available |

## Package overview

::: diagnostix
    options:
      members: false
      show_root_heading: false
      show_source: false
