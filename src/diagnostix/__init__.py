"""Physical and statistical diagnostics of geophysical fields on any array library.

``diagnostix`` takes a field and says something about it, physically or
statistically: what the flow does, where the mixed layer ends, what a budget
leaves over, which patterns vary together, and how good a forecast is.

## Modules

Each is created by its first feature issue; so far there are
`diagnostix.kinematics`, `diagnostix.columns`, `diagnostix.budgets` and
`diagnostix.climate`.

| Module | What it gives you |
|---|---|
| `diagnostix.kinematics` | velocity gradients, vorticity, advection, frontogenesis |
| `diagnostix.columns` | stratification, mixed and boundary layers, integrals |
| `diagnostix.budgets` | control volumes, boundary fluxes, budgets |
| `diagnostix.climate` | climatologies, anomalies, trends |
| `diagnostix.spectra` | power, cross, rotary and wavelet spectra |
| `diagnostix.eofs` | area-weighted EOFs |
| `diagnostix.skill` | pointwise, ensemble, spectral and physical scores |
"""

from __future__ import annotations

from diagnostix.exceptions import BackendError, DiagnostixError, ValidationError


__version__ = "0.0.2"

__all__ = [
    "BackendError",
    "DiagnostixError",
    "ValidationError",
    "__version__",
]
