# AGENTS.md

Standing instructions for **every** coding agent working in this repository (Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth: `CLAUDE.md` and `.github/copilot-instructions.md` point here.

The house style (Python, maths notation, prose, pseudocode, reports) comes from the maintainer's personal skills in [jejagents](https://github.com/jejjohnson/jejagents); "House style" and "Personal skills" below say which skill covers what and carry the essentials, so an agent without those skills installed can still follow them.


## What this repo is

diagnostix is the GeoStack's library of diagnostics: it takes a geophysical field and says something about it, physically or statistically. It is a GeoNumerics library, above geophysix and gridax, written once in the Python array API standard, so it runs on NumPy, JAX and any other array-API library.

People build on a package like this, so **the package is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

The public API is the set of public modules below; private code lives in `src/diagnostix/_src/<module>/`, and code that several modules share in `src/diagnostix/_src/`. `src/diagnostix/__init__.py` exports only `__version__` and the exceptions. [jejjohnson/geostack#309](https://github.com/jejjohnson/geostack/issues/309) is the backlog.

| Module | Owns | Status |
|---|---|---|
| `_typing.py` | The `ArrayLike` type alias | now |
| `exceptions.py` | `DiagnostixError` (a `ValueError`) and its subclass `ValidationError`; pipekit's `BackendError` | now |
| `kinematics` | Velocity gradient, shear and curvature vorticity, advection, frontogenesis, ageostrophic velocity, geostrophic residual | now |
| `columns` | Vertical derivative, N², lapse rate, column integrals, first crossings, mixed-layer depth, bulk Richardson number, boundary-layer height | now |
| `budgets` | Control volumes, face and boundary fluxes, the flux divergence, region integrals, budgets and their residuals | now |
| `climate` | Calendar days, climatologies, harmonic fits and smoothing, anomalies, linear trends, running means | now |
| `spectra` | Power, isotropic, cross, rotary and wavelet spectra, coherence, spectral fluxes, slopes and scales | now |
| `eofs` | Area-weighted EOFs, projection and reconstruction | now |
| `skill` | Pointwise, ensemble, spectral and physical scores, the Diebold–Mariano statistic | now |

### Boundaries

- **Layer rule.** diagnostix imports gridax, geophysix and the array backend `pipekit_array.functions`, and nothing else from the GeoStack at runtime. It never imports functax, integrix, gaussx, geotoolz or any GeoModels library; xrops, plumax and sesmax-ocean import it. functax is a reference library in the tests. `tests/test_import_isolation.py` checks this.
- **Runtime dependencies**: `numpy>=2.0`, `jaxtyping`, `gridax`, `geophysix` and `pipekit-array[numpy,compat]`, with the `pipekit-ops` it brings. None of the stack's packages is on PyPI, so they are pinned to git tags, and every pipekit package to one tag, the one that gridax and geophysix pin. JAX and xarray are extras.
- **Ownership test.** Does it take a field and say something about it, and does no library below already name it? Then it belongs here. A quantity gridax already names (curl, divergence, kinetic energy, strain, Okubo–Weiss, geostrophic velocity, shallow-water potential vorticity) stays gridax's, on every grid. A fact at a point (density, θ, wind speed, Coriolis) is geophysix's. Labels, CF and dask are xrops'. Benchmarks are pipekit-evaluate's. Extremes are xtremax's, and Lagrangian analysis is tracerax's.
- **No convenience aliases.** A one-line combination of functions the stack has (absolute vorticity, eddy kinetic energy, f from latitude) is written by the caller, and the docs show it.

When other projects build on the package, list them here and keep their imports working through the change that moves them.

## What it is built on

Each foundation brings a rule. Breaking one runs fine on your machine and fails in someone else's pipeline or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **NumPy ≥ 2.0** | The default namespace; tables built once on the host | NumPy in gives NumPy out; no function needs JAX or xarray installed. |
| **pipekit-array** (`pipekit_array.functions`, `.weighted`, `.records`) | The namespace, the dtype and the device of a call; masked sums, means and groups; records as pytrees | A change to them is made there. Only functions of the array API standard are called on the namespace; `array_api_strict` in the tests refuses the rest. |
| **gridax** | Grids, masks, derivatives, `area_weights`, and the diagnostics it names | No stencil, metric term or cell area is written here when gridax has it. gridax does not run on `array_api_strict`, so a function that calls it is tested on NumPy and JAX. |
| **geophysix** | Constants and facts at a point | No constant or Earth radius is written here. |
| **JAX** (extra `jax`) | The same functions under `jit`, `vmap` and `grad` | Never imported at module level, and nothing inside the library calls `jax.jit`: the caller jits once, at the top. |
| **xarray** (extra `xarray`) | A DataArray in, a DataArray out, for elementwise functions only | Imported only when a DataArray arrives. |
| **`dataclasses`** | Results that hold several arrays | `frozen=True`, registered as JAX pytrees once JAX loads. |
| **scipy, hypothesis, array-api-strict** (the `test` group) | Reference results and property tests | Never a runtime import. |
| **MetPy, xrft, xskillscore, scikit-learn, functax** (the `oracles` group) | Reference results | Only the job "reference libraries" installs them; an oracle test skips when its package is missing. xrops is never one of them, because it imports diagnostix. |
| **pytest + `--doctest-modules`** | The suite and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a constant, a conversion or a formula, find out whether it exists:

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public name once, at its public home, grouped by the module that defines it, with a one-line summary. It is generated (`uv run poe capabilities`) and checked by `tests/test_capabilities.py`, so it is current.
2. **Search the private helpers** in the table below: several invariants live in exactly one place each.
3. **If it is missing, add it to the module that owns the concept**, at the lowest layer every caller can reach, not inline in the one caller you have today; a helper two modules need goes into `utils.py`.
4. **One object, one name.** A name means one thing across the package and its submodules (`tests/test_capabilities.py`); a deliberate exception is listed with its reason in `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| A hand-written Earth radius, g, or a gas constant | `geophysix.earth`, `geophysix.constants.STANDARD` |
| A cell area, a cosine-latitude weight | `gridax.area_weights` |
| A curl, a divergence, a strain, an Okubo–Weiss, a geostrophic velocity, a kinetic energy | the gridax function of that name |
| A density, a potential temperature, a Coriolis parameter, a wind speed | geophysix |
| `xp = np if isinstance(x, np.ndarray) else jnp`, or `x.astype(float)` on an input | `namespace_of`, `floating_inputs` from `pipekit_array.functions` |
| A constant as an array of the input's dtype, on its device | `pipekit_array.functions.like` |
| `isinstance(x, xr.DataArray)`, or a call to `xr.apply_ufunc` | `@elementwise` and `is_dataarray` from `pipekit_array.functions` |
| `nansum`, `nanmean`, a sum or mean with a mask or weights | `weighted_sum`, `weighted_mean` from `pipekit_array.weighted` |
| `bincount`, `segment_sum`, a `groupby`, an average over rings, days or bins | `Groups` from `pipekit_array.weighted`, built once on the host |
| A ratio that is NaN where its denominator is 0, a host table on an array's device, a mask as an array | `divide_or_nan`, `table_like`, `boolean_like` from `pipekit_array.functions` |
| A host table or a grid's metric in an array's dtype too | `constants_like` from `pipekit_array.functions` |
| The checks of another reduction along an axis | `reduction_inputs`, `used_weights` from `pipekit_array.weighted`, `require_axis` from `diagnostix._src.reduce` |
| A frozen result that must be a JAX pytree | `@register_record` from `pipekit_array.records` |
| The checks of a call on a grid | `grid_inputs` from `diagnostix._src.grids` |
| Times in a field's precision | `time_offsets` from `diagnostix._src.times` |
| A derivative along an axis of uneven positions | `vertical_derivative`; inside, `uneven_difference` of `_src/columns/derivatives.py` |
| A sum or a mean over a centred window | `window_sum` from `diagnostix._src.climate.window` |
| A detrended, tapered field before a transform, \|z\|² | `tapered` from `diagnostix._src.spectra.taper`, `squared_magnitude` from `pipekit_array.functions` |
| A sum over rings of the wavenumber plane | `Rings` from `diagnostix._src.spectra.rings` |
| The cells of a score, a root with a gradient at 0, a ratio that is ∞ over 0 | `paired_inputs`, `root`, `ratio_or_inf` from `diagnostix._src.skill.moments` |
| The members and cases of an ensemble score | `ensemble_inputs` from `diagnostix._src.skill.members` |
| The face lengths of a grid, a mean onto its faces | `face_lengths`, `face_mean` from `diagnostix._src.budgets.faces` |
| A new exception for a bad argument | `ValidationError` |

## The contracts

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### 1. Public functions

These are the decisions of [#15](https://github.com/jejjohnson/diagnostix/issues/15), where the measurements behind them are.

- **Scope.** diagnostix owns a function that takes a field and says something about it, physically or statistically. A quantity gridax already names (curl, divergence, kinetic energy, strain, Okubo–Weiss, geostrophic velocity, shallow-water potential vorticity) stays gridax's, on every grid. A fact at a point (density, θ, wind speed, Coriolis) is geophysix's. Labels, CF and dask are xrops'. Benchmarks are pipekit-evaluate's.
- **Arrays.** One implementation in functions of the Python array API standard, through `pipekit_array.functions`. JAX and xarray are never imported at module level. Nothing inside the library calls `jax.jit`; the caller jits once, at the top.
- **Dependencies.** NumPy, gridax, geophysix and `pipekit-array[numpy,compat]`. Nothing else from the stack at runtime.
- **Labels.** Elementwise functions carry pipekit's `@elementwise()` and take DataArrays. Functions over a grid or an axis take plain arrays, and xrops wraps them; a DataArray there raises `BackendError`, so labels are never lost silently.
- **Axes.** A field on a grid ends in (y, x), on a collocated gridax grid, with any leading axes. A function along one axis takes `axis=-1`.
- **Missing data.** Boolean, True = use. Functions on a grid take gridax's `mask=`; statistics take `valid=`. A NaN in a valid cell gives NaN. An undefined result is NaN through a double `where`, so that gradients of the other cells stay finite.
- **Weights** are non-negative and need not be normalised.
- **Precision.** The output dtype follows the inputs: float32 stays float32. Nothing enables x64.
- **Units.** SI. Physical constants are geophysix records, passed as keywords with a default; f is an argument, never read from a global.
- **Time.** Float seconds or days for trends and tendencies; integer calendar days and months for climatologies; never `datetime`.
- **Results.** An array, or a frozen dataclass registered as a JAX pytree when JAX is imported.
- **Pure.** Return a new array and never mutate the input.
- **Documented.** Google-style docstring in the `python-style` layout: a one-line summary, the defining equation in Unicode (`eman-style-math`, no `$…$` in docstrings), `Args:` (meaning, symbol and unit, no type), `Returns:`, `Raises:` and an executable `Examples:` block.

### 2. Exceptions

- Every error derives from `DiagnostixError`, which is a `ValueError`: `ValidationError` is a bad argument (its shape, a dimension name, its value). The one exception is the array backend, which is shared: pipekit's `BackendError` (a `TypeError` and a `ValueError`) for an array of an unknown library or of several in one call, and a plain `ValueError` for an input that is not real, a mask that is not boolean, an axis that does not hold the samples of a `Groups`, or an unlabelled array next to a DataArray. A new error goes into `exceptions.py`, is exported, and joins the list in `tests/test_exceptions.py`.

### The public API

- Private code lives in `src/diagnostix/_src/<module>/`; the public modules re-export it. Export a public name from its module's `__all__`; both stay in plain `sorted()` order, which `tests/test_public_api.py` checks (ruff's `RUF022` is off: its isort-style order differs). `src/diagnostix/__init__.py` exports `__version__` and the exceptions only.
- A new module declares a sorted `__all__`, joins `SUBMODULES` in `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: diagnostix.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; a name on an existing module's page is picked up by its `:::` block.
- Run `uv run poe capabilities` after any public API change.
- Link prose to the API by the public name (``[`DiagnostixError`](xref:api#diagnostix.DiagnostixError)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` is exactly the set of modules; every submodule declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `--doctest-modules` on `src/diagnostix` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_exceptions.py` | The hierarchy: every error is a `DiagnostixError`, which is a `ValueError`; `BackendError` is pipekit's |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `tests/conftest.py` | `jax_enable_x64` is on whenever JAX is importable; tests marked `jax` or `xarray` are skipped when it is not |
| `tests/test_import_isolation.py` | Importing every module loads neither JAX nor xarray, and no stack library but gridax, geophysix, pipekit-array and pipekit-ops |
| `tests/test_harness.py` | The checks of `pipekit_array.testing` pass on toy functions, each closed form and library in `tests/_oracles.py` agrees with an independent evaluation, and every stored xrops value is what its name says |
| CI job "without JAX and xarray" | The base install imports and passes the tests with neither installed |
| CI job "reference libraries" | The `oracles` group resolves, and the comparisons with it pass |
| `uv sync --locked` in every test job | `uv.lock` matches the pins in `pyproject.toml` |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/diagnostix` in every coverage-enabled run (CI, `uv run poe test-cov`) |
| `strict_markers = true` | Only the registered markers (`slow`, `integration`, `jax`, `xarray`, `oracle`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo), format, types on `src/diagnostix` and `scripts` |

Each function brings five tests, as `add-diagnostic` lays out, with the helpers of `tests/`:

1. **Closed form** or stored reference, at a stated tolerance, in float64 (`assert_matches_oracle`, `tests/_oracles.py`). A closed form comes first, and `xrops_values` only where xrops is right ([#16](https://github.com/jejjohnson/diagnostix/issues/16)).
2. **Backends**: NumPy, JAX, `array_api_strict` and dask give the same values. A function that calls gridax runs on NumPy and JAX only (`assert_backends_agree`, with `kinds=GRID_KINDS`), and one that dask cannot run says so in its docstring (`kinds=EAGER_KINDS`; `labelled=True` for an elementwise function, `assert_refuses_dataarrays` for the others).
3. **Transforms**: under JAX, eager, `jit` and `vmap` over a batch of one agree (`assert_traces`).
4. **Gradients**: autodiff against central differences where the function is smooth, and finite on the valid cells when others are invalid (`assert_grad_matches_fd`, `assert_finite_gradient`).
5. **Precision**: float32 in gives float32 out (`assert_backends_agree` checks it).

A test on random draws uses `assert_mean` of `pipekit_array.testing`. xrops will import diagnostix, so its values are stored in `tests/data/`, with the geotoolz commit in each file name, by `write_xrops.py`, which refuses another commit; add an array there rather than a test dependency, and check it against its formula in `tests/test_harness.py`.

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

| Job | Recipe |
|---|---|
| Add a function to a public module | `add-diagnostic` |
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `diagnostix-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `correctness-reviewer.md`, the diagnostics reviewer) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `uv run poe gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |


## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`, `ls` and other non-Python commands need no `uv run`.

```bash
uv run poe                  # list every task (they live in [tool.poe.tasks] in pyproject.toml)
uv run poe install          # uv sync --all-groups + pre-commit hooks
uv run poe test             # every test + doctests, no coverage (fast)
uv run poe test-cov         # the same with coverage and the 90 % gate (what CI runs)
uv run poe lint             # ruff check .   (entire repo)
uv run poe format           # ruff format . && ruff check --fix .
uv run poe typecheck        # ty check src/diagnostix scripts (what CI runs)
uv run poe check            # lint + format-check + typecheck + test-cov (everything CI checks)
uv run poe capabilities     # regenerate docs/api/capabilities.md
uv run poe docs             # build both docs halves into public/ and check every link
uv run poe docs-check       # strict validation of both halves, no themed render
uv run poe docs-api         # Zensical API reference only (zensical build --strict)
uv run poe docs-serve       # build, then serve public/ at :8000
uv run poe precommit        # every pre-commit hook on every file
uv run poe build            # wheel + sdist into dist/
```

Extra arguments after a task name are passed to its command (`uv run poe test tests/test_exceptions.py -k base`).

`addopts` turns coverage on, and a partial run never reaches the 90 % gate, so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_exceptions.py -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For a parametrised test, mark only the expensive cases (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13 (`uv run pytest -v`), and again with no extra and with the `oracles` group. Select the fast tier with `uv run pytest -m "not slow and not integration"`; when the slow tiers grow, give them a CI job of their own rather than dropping them. `jax`, `xarray` and `oracle` are not tiers: they say what a test needs.

### Tests that assert on random draws

diagnostix draws no random numbers; its tests do, for ensembles and property tests. A failing `hypothesis` example is pinned with an explicit `@example` so it keeps being run.

- **Incidental randomness** (any draw would do): seed it (`random.Random(0)`, `numpy.random.default_rng(0)`, a fixed key) so a failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical property): bound the statistic by its own sampling distribution (a number of standard errors, a test's p-value floor), not by a tolerance tuned on one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `uv run poe test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`. Never lint a subdirectory: `ruff check src/diagnostix/` misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `uv run poe typecheck` (`ty check src/diagnostix scripts`, as CI runs it).
5. After changing a public API: `uv run poe capabilities`, and the `docs/api` page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`: `uv run poe docs` (or at least `uv run poe docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several readings, name them instead of picking one silently; if something is unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no speculative features, no single-use abstractions, no configurability nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing style; don't refactor or add docstrings to code you didn't change; remove only what your change made unused, and mention (don't delete) unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced bug, a closed form to match) and loop until it passes; for multi-step work, state the plan with a check per step.

These are the `karpathy-coding-principles` skill in four lines. When you suggest a change, propose the code; when you fix a bug or add a feature, add the test that checks it.

## House style

### Python

`python-style` is the house style. Its general rules hold in every package built from this template:

- Python 3.12+, `from __future__ import annotations` in every module, type hints on every public function, `X | None` over `Optional`, specific exceptions with `raise … from …`, pure functions with side effects isolated and explicit.
- Every module opens with a docstring saying what it holds, in one or two sentences.
- Docstrings follow the layout in "The contracts" above; `Examples:` gives two or three cases (typical, edge, composition), each building its own small data, and `References:` names the paper when one defines the method.
- Comments go above the line, never beside it. A mathematical line has its Unicode equation above it (`# x̄ = (1/N) ∑ᵢ xᵢ`), a long function is split into stages with step headers (`# --- whiten ---`), and every other comment says why, not what.
- Names are descriptive (`posterior_precision`, not `Lam`); the symbol lives only in the equation comment.
- One operation per line, no nested calls that hide an intermediate result, 88 columns.
- One package base error; subclasses are named by what failed.
- Extract only real duplicates (code that must change together); code that merely looks alike stays separate.

Its JAX/Equinox rules apply when the package uses JAX: logic under `pkg/_src/` with public modules that only re-export, jaxtyping with the shared dim vocabulary, einx instead of `axis=` / `.T` / `reshape` / `[:, None]`, lineax for solves, arrays in `eqx.Module`s with `__check_init__` and `eqx.error_if`, `key` keyword-only and last, no silent dtype casts, `get_config()` / `from_config()`, and the four tests per public forward function (closed-form oracle, hypothesis properties, gradient check, transform invariance).

diagnostix computes through plain functions and frozen `dataclasses` where `python-style` uses `eqx.Module` and einx, because NumPy in must give NumPy out and JAX stays optional. Reductions use the array API's `axis=` and `xp.matmul`. Errors derive from `DiagnostixError`, which is itself a `ValueError`, so `except ValueError` keeps working. Doctests print exact floats only when the value is exactly representable; otherwise they round.

**CLIs and scripts** use `cyclopts` for arguments and `loguru` for output; never argparse, click, typer, progress `print`s or stdlib `logging`. For a repo script they are in the `dev` group already; diagnostix has no CLI. `scripts/capabilities.py` still uses `argparse` and `print`; move it over when you next change it (`scripts/build_docs.py` shows the pattern: a `cyclopts.App` with `@app.default`, and `loguru` for every message).

### Maths notation

Maths follows `eman-style-math` in every format: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol is defined, with its shape, before it is used.

- **Code and docstrings:** Unicode only (x̄, σ², Σ⁻¹, ∑, 𝔼, ⊙), never `$…$`; identifiers stay ASCII.
- **MyST prose and MkDocs pages:** `$…$` and `$$…$$` with the kind-named macros (`\vect{x}`, `\gvect{\mu}`, `\mat{K}`, `\gmat{\Sigma}`). The macros are installed in `docs/myst.yml` (`math:`) and `docs/api/javascripts/mathjax.js`; regenerate both with the skill's `scripts/emit_macros.py` when the macros change.
- **GitHub issues and PRs:** expand the macros first (`scripts/expand_macros.py`), display maths in a `math` fence.
- **Everywhere:** each `$…$` expression on one source line; no spacing or sizing macros (`\,`, `\;`, `\quad`, `\qquad`, `\big`, `\tfrac`, `\boxed`, `\cdots`, `\operatorname`). Check a page with `uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py <file> --target myst` (the skill's scripts need those three packages, which this repo does not install).

### Prose and diagrams

- Markdown prose is never hard-wrapped: one line per paragraph or list item in docs, skills and agent instructions; issue bodies follow `report` (one sentence per line); PR text, one line per paragraph. Don't rewrap untouched text in a file you edit.
- Docs, READMEs and release notes written in the maintainer's voice follow `eman-prose-tech` (the docs register): plain words, "we" for the maintainers, parentheses rather than em dashes.
- Pseudocode in design pages, issues and READMEs follows `code-sketch`: a shapes header, the equation above each line, an aligned shape column and a status marker (`today (pkg)`, `proposed (X)`, `user`) on every call into the stack.
- Never draw ASCII box-and-arrow diagrams; use Mermaid (both docs halves render it). Directory trees stay plain text.

## Personal skills

The maintainer's skills live in [jejagents](https://github.com/jejjohnson/jejagents) and install into `~/.claude/skills/` with its `install.sh`. Use them when they are installed; the sections above carry what an agent without them needs. Where a skill and this file disagree, this file wins: it records the template's deliberate departures.

| Job | Skill |
|---|---|
| Write, review or refactor Python | `python-style` |
| Maths in docstrings, docs pages, issues or PRs | `eman-style-math` |
| Pseudocode in a design page, issue or README | `code-sketch` |
| Prose in the maintainer's voice (docs, README, release notes, PR descriptions) | `eman-prose-tech` |
| Write up a research, feature, bug, design, epic or chore report | `report`, then file it here with `create-gh-issue` |
| File or link issues in another of the maintainer's repos | `gh-issue` (in this repo, `create-gh-issue` and `link-gh-issues` follow it) |
| A derivation walkthrough for the docs | `derivation`, with `derivation-myst` for a page in `docs/` |
| A learning explainer page | `explainer` |
| General coding discipline | `karpathy-coding-principles` ("Coding principles" above) |

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push to main", "merge to main"). Work on a feature branch and commit locally; never run `git push` unless asked. "Merge the branch" means push the feature branch, not merge into `main`. Confirm before any action that affects a shared branch.
- Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) with a lowercase subject (`feat(columns): add …`); CI validates PR titles (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- **No AI attribution.** No `Co-Authored-By:` trailer for an AI on any commit and no "Generated with …" line in a PR description, unless the maintainer asks for one; keep human co-authors' trailers.
- **PR descriptions are for a human who did not watch the work:** why first, a small before/after table, what stays the same, one or two bullets on how it was tested. Label behaviour as "Today (bug): …" and "After this PR: …". Link review comments rather than naming reviewers.
- Write one `Closes #N` line per issue.
- Releases are cut by release-please (`release-please-config.json`): it bumps the version and writes `CHANGELOG.md` from the `feat`, `fix`, `perf` and `revert` commits; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent called for a small follow-up tends to write a fresh description scoped to its own work and silently discard the rest: read the existing description first and only append checklist items or update their status. Change the title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

Fix a review comment in code, push the fix to the PR branch, then resolve its thread; don't post a reply comment. Don't resolve threads you didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

GraphQL is blocked in Claude sessions; there, and whenever the `gh` CLI is unavailable, use the GitHub MCP tools for the same operations (they can resolve review threads).

## Documentation

The docs are built by **two tools** and deployed as one site (`docs/README.md` has the rationale); `docs.yml` builds them on every PR and push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from `main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | Zensical + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`uv run poe docs` runs `scripts/build_docs.py`: it builds the API with `zensical build --strict` (Zensical reads `mkdocs.yml`), serves `site/` on port 8910 so mystmd can read its `objects.inv` (mystmd loads inventories only over http), runs `myst build --html --strict`, rewrites the localhost API URLs to `/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and fails if any internal link in the assembled `public/` site does not resolve. mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its theme download is blocked, `docs/README.md` says how to build against a local clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`, `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level** exported name: ``[`DiagnostixError`](xref:api#diagnostix.DiagnostixError)``, not a deeper path (mystmd lowercases `$`-abbreviated anchors; see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-` (`guide/quickstart.md` → `/quickstart/`, `notebooks/deep_dive.ipynb` → `/deep-dive/`), so keep basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`; frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`): the theme prefixes root-relative ones with `BASE_URL` a second time (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs-format pages, built by Zensical (which runs only the plugins it supports and ignores the rest silently; check its plugin list before adding one): prose plus `:::` blocks, `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are Google style (`docstring_style: google`) with Unicode maths; maths in the API pages themselves goes through MathJax (`pymdownx.arithmatex`), with the `eman-style-math` macros loaded by `docs/api/javascripts/mathjax.js`.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb` files, authored as jupytext percent `.py` and deleted after conversion; mystmd renders the stored outputs without re-executing, so each must be listed in the `docs/myst.yml` toc. Figures render inline with `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root holds scratch notebooks, outside the docs. Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed); track work in GitHub issues. Draft a whole wave of issues from `docs/templates/wave-backlog.md` (see `docs/contributing.md`).
