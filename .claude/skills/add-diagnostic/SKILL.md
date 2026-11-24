---
name: add-diagnostic
description: Add a diagnostic to a public module of the package — whether gridax or geophysix already names it, where it goes in _src/<module>/, its units and signs, array-API code with masks and weights, a Google-style docstring, the closed form, backend, transform, gradient, precision and mask tests, the API page and the capability index. Use when asked to add, port or implement a diagnostic, a budget, a climatology, a spectrum, a score or a new module in src/diagnostix.
---

# Add a diagnostic

Read "1. Public functions" and "The public API" under "The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` for the quantity and its synonyms, and the "Reuse before you write" table in `AGENTS.md`.
- Does gridax name it (curl, divergence, kinetic energy, strain, Okubo–Weiss, geostrophic velocity, shallow-water potential vorticity)? Then it stays gridax's, on every grid: a missing grid is a gridax issue, not a function here.
- Is it a fact at a point (density, θ, wind speed, Coriolis)? Then it is geophysix's.
- A one-line combination of existing names needs no new function: show it in the docs.

## 2. Pick its home

Private code goes in `src/diagnostix/_src/<module>/<topic>.py`; the public module (`kinematics`, `columns`, `budgets`, `climate`, `spectra`, `eofs`, `skill`) re-exports it. A new public module is created by its first feature issue. A helper that two modules need goes in `src/diagnostix/_src/`. A new exception goes in `exceptions.py`, derives from `DiagnostixError` and joins `tests/test_exceptions.py`.

## 3. Write it

- Open with `xp, (x, y) = floating_inputs(x, y)` from `pipekit_array.functions`, and call only functions of the array API standard on `xp`. Never `import jax` or `import xarray` at module level, and never call `jax.jit`.
- Elementwise: decorate with `@elementwise()`, so a DataArray in gives a DataArray out. Over a grid or an axis: plain arrays only; `floating_inputs` raises `BackendError` for a DataArray.
- Axes: a field on a grid ends in (y, x); a function along one axis takes `axis=-1`.
- Masks: `mask=` on a grid, `valid=` in statistics, boolean, True = use. Replace an invalid value by 0 through `where` before any product. An undefined result is NaN through a double `where`.
- Weights broadcast, are non-negative, and need not be normalised.
- Units: SI, stated in `Args:` and `Returns:`, with the sign convention in words. Constants are geophysix records, keywords with a default; f is an argument.
- Constants enter in the input's dtype: `c = like(C, x, xp)`; float32 in gives float32 out.
- Comments above the line with the Unicode equation above each mathematical line; descriptive names; one operation per line.
- Google-style docstring in the `python-style` layout: a one-line summary (it becomes the capability-index entry), the defining equation in Unicode (never `$…$`), `Args:` with meaning, symbol and unit, `Returns:`, `Raises:` and an `Examples:` block of rounded values.

## 4. Export and document

- Add the name to the public module's `__all__` (sorted); a new module joins `SUBMODULES` in `tests/test_public_api.py`.
- API page `docs/api/<module>.md` with a `::: diagnostix.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/test_<module>.py`, one class per function:

- **Closed form** or stored reference, with the tolerance and its source in a comment. Stored xrops values only where xrops is right (the table in #16).
- **Backends.** NumPy, JAX and `array_api_strict` give the same values; a function that calls gridax runs on NumPy and JAX only.
- **Transforms.** Eager, `jit` and `vmap` over a batch of one agree.
- **Gradients.** Autodiff against central differences, and finite on the valid cells when others are invalid.
- **Precision.** float32 in gives float32 out.
- **Mask.** A NaN under an invalid cell changes nothing; a NaN in a valid cell gives NaN.
- Each invalid argument (`match=`), and an oracle of the `oracles` group where one exists (`pytest.importorskip`, marked `oracle`).

## 6. Verify

```bash
uv run pytest --no-cov tests/test_<module>.py src/diagnostix/_src/<module> tests/test_public_api.py tests/test_capabilities.py -v
uv run poe capabilities
uv run poe test-cov
```

then the `pre-pr-check` skill.
