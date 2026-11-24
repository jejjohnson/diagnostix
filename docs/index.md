# diagnostix

> Physical and statistical diagnostics of geophysical fields on any array library.

diagnostix is the GeoStack's library of diagnostics: it takes a geophysical field and says something about it, physically or statistically. It will hold velocity gradients and frontogenesis, mixed and boundary layers, budgets, climatologies and trends, spectra, EOFs, and the scores of a forecast.

It is written once, in the Python array API standard: NumPy in gives NumPy out, JAX in gives JAX out, and the same for any other array-API library. Neither JAX nor xarray is imported unless you pass one of their arrays.

## Installation

diagnostix is not on PyPI yet, and neither are [gridax](https://github.com/jejjohnson/gridax), [geophysix](https://github.com/jejjohnson/geophysix) and [pipekit-array](https://github.com/jejjohnson/pipekit), which it builds on, so it installs from a clone.

```bash
git clone https://github.com/jejjohnson/diagnostix.git
cd diagnostix
uv run poe install
```

## Status

The package has its scaffold, and no public module is built yet. The [API overview](https://jejjohnson.github.io/diagnostix/reference/) lists the modules and the wave that builds each.

## Design rules

1. **One name per quantity.** What gridax or geophysix already names stays theirs.
2. **Any array library.** The result is the same kind of array as the input. Elementwise functions keep a DataArray's labels, and functions over a grid or an axis refuse one, so labels are never lost silently.
3. **True means use.** Masks are boolean, and a NaN is not a mask.
4. **Constants enter in the input's dtype**, so float32 stays float32.
5. **Fail with a typed exception.** Everything raised derives from [`DiagnostixError`](xref:api#diagnostix.DiagnostixError), which is a `ValueError`. An undefined result is NaN, not an error.
