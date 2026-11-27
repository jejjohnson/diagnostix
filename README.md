# diagnostix

[![Tests](https://github.com/jejjohnson/diagnostix/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/diagnostix/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/diagnostix/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/diagnostix/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/diagnostix/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/diagnostix/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/diagnostix/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/diagnostix/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/diagnostix/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/diagnostix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)


Physical and statistical diagnostics of geophysical fields on any array library.

diagnostix is the GeoStack's library of diagnostics: it takes a geophysical field and says something about it, physically or statistically. It holds velocity gradients and frontogenesis, mixed and boundary layers, budgets, climatologies and trends, spectra, EOFs, and the scores of a forecast.

It is written once, in the Python array API standard: NumPy in gives NumPy out, JAX in gives JAX out, and the same for any other array-API library. Neither JAX nor xarray is imported unless you pass one of their arrays.

diagnostix defines no second name for a quantity [gridax](https://github.com/jejjohnson/gridax) already has (curl, divergence, strain, Okubo–Weiss, geostrophic velocity) or for a fact at a point that [geophysix](https://github.com/jejjohnson/geophysix) has (density, potential temperature, Coriolis). Labels, CF and dask are in xrops, which will wrap diagnostix.

## Install

diagnostix is not on PyPI yet, and neither are the stack libraries it builds on, so it installs from a clone:

```bash
git clone https://github.com/jejjohnson/diagnostix.git
cd diagnostix
uv run poe install
```

## Status

Every module is built:

| Module | Holds | Wave |
|---|---|---|
| `kinematics` | velocity gradient, shear and curvature vorticity, advection, frontogenesis, ageostrophic velocity, geostrophic residual | [Wave 1](https://github.com/jejjohnson/diagnostix/issues/9), built |
| `columns` | vertical derivative, N², lapse rate, column integrals, first crossings, mixed-layer depth, bulk Richardson number, boundary-layer height | [Wave 1](https://github.com/jejjohnson/diagnostix/issues/9), built |
| `budgets` | control volumes, boundary fluxes, budgets and their residuals | [Wave 2](https://github.com/jejjohnson/diagnostix/issues/10), built |
| `climate` | climatologies, anomalies, trends, running means | [Wave 2](https://github.com/jejjohnson/diagnostix/issues/10), built |
| `spectra` | power, isotropic, cross, rotary and wavelet spectra, coherence; spectral fluxes, slopes and scales | [Wave 3](https://github.com/jejjohnson/diagnostix/issues/11), built |
| `eofs` | area-weighted EOFs, projection and reconstruction | [Wave 3](https://github.com/jejjohnson/diagnostix/issues/11), built |
| `skill` | pointwise, ensemble, spectral and physical scores; the Diebold–Mariano statistic | [Wave 4](https://github.com/jejjohnson/diagnostix/issues/12), built |
| `trajectories` | separation of a trajectory from an observed one, skill, absolute and pair dispersion | [#75](https://github.com/jejjohnson/diagnostix/issues/75), built |

## Design

The backlog is [geostack#309](https://github.com/jejjohnson/geostack/issues/309), the decisions are in [#15](https://github.com/jejjohnson/diagnostix/issues/15), and the design page is in [geostack](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/diagnostix.md).
