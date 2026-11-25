# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/diagnostix/`: the GeoStack's library of physical and statistical diagnostics. It imports gridax, geophysix and `pipekit_array`, and nothing else from the GeoStack at runtime (never functax, integrix, gaussx, geotoolz or a model). Public modules, each created by its first feature issue: `kinematics`, `columns`, `budgets`, `climate`, `spectra`, `eofs`, `skill`; private code in `_src/<module>/`. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Ownership test: does it take a field and say something about it, physically or statistically, and does no library below already name it? A quantity gridax names stays gridax's on every grid, a fact at a point is geophysix's, and labels are xrops'.
- Keep the contracts in `AGENTS.md`:
  - **arrays**: one implementation in array API functions; the result is the same kind of array as the input; JAX and xarray are never imported at module level; nothing calls `jax.jit`; constants enter in the input's dtype;
  - **labels**: elementwise functions take DataArrays, functions over a grid or an axis raise `BackendError` for one;
  - **axes** are (…, y, x) on a grid and `axis=-1` along one axis; **masks** (`mask=`, `valid=`) are boolean, True = use; weights need not be normalised;
  - **units** are SI and f is an argument; **time** is float seconds or days, or integer calendar days and months;
  - **undefined results** are NaN through a double `where`; every exception derives from `DiagnostixError`, which is a `ValueError`, except those of the array backend (`AGENTS.md`, "Exceptions").

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (add a diagnostic, guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).
