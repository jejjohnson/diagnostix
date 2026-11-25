# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a missing metric term, a mask read the wrong way round, a spectrum that is not normalised, a constant that promotes float32, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented stencil, cell area or kinematic quantity gridax names, or a constant or point fact geophysix has, is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the library imports gridax, geophysix and `pipekit_array`, and never functax, integrix, gaussx, geotoolz or any model at runtime.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Arrays.** A function computes in the namespace of its inputs, with array API functions only; NumPy in gives NumPy out; JAX and xarray are never imported at module level; nothing calls `jax.jit`.
- **Labels.** An elementwise function takes a DataArray and gives one back; a function over a grid or an axis raises `BackendError` for one.
- **Axes.** (…, y, x) on a grid, `axis=-1` along one axis; no position is assumed for time or for the vertical.
- **Missing data.** `mask=` on a grid, `valid=` in statistics, boolean, True = use. A NaN in a valid cell gives NaN; an invalid value is replaced before any product; an undefined result is NaN through a double `where`.
- **Weights** are non-negative and the result does not depend on their scale.
- **Constants** enter in the input's dtype (`like(c, x, xp)`); float32 in gives float32 out.
- **Units.** SI; constants are geophysix records passed as keywords with defaults; f is an argument. The sign convention is in the docstring.
- **Time.** Float seconds or days, or integer calendar days and months; never `datetime`.
- **Results** are arrays, or frozen dataclasses registered as JAX pytrees.
- **Pure.** Return a new array and never mutate the input.
- **Exceptions** derive from `DiagnostixError` (a `ValueError`), except those of the array backend (`AGENTS.md`); errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (rounded values, not 17-digit floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#diagnostix.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each function has its five tests ("What enforces them" in `AGENTS.md`): a closed form or a stored reference, backends, transforms, gradients and precision; and a mask test. The edges (an empty group, an all-masked column, a constant field, the periodic seam) have their own.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Each example below runs on NumPy alone.

### A NaN under an invalid cell must not reach the sum

```python
import numpy as np

values = np.asarray([1.0, np.nan, 3.0])
valid = np.asarray([True, False, True])

# ❌ NaN × 0 is NaN: the mask comes after the product
assert np.isnan(np.sum(values * valid))

# ✅ The invalid value is replaced first, and the mean divides by what was used
kept = np.where(valid, values, 0.0)
assert np.sum(kept) / np.sum(valid) == 2.0
```

### An undefined result is NaN, and its gradient does not spread

```python
import numpy as np


# ✅ The inner where keeps the unused branch finite, so the gradient of the
#    other cells is too; a single outer where would differentiate 1/0
def mean_or_nan(total, weight):
    defined = weight > 0
    safe = np.where(defined, weight, 1.0)
    return np.where(defined, total / safe, np.nan)


assert np.isnan(mean_or_nan(np.asarray(0.0), np.asarray(0.0)))
assert mean_or_nan(np.asarray(6.0), np.asarray(3.0)) == 2.0
```

### Weights need not be normalised

```python
import numpy as np

values = np.asarray([1.0, 4.0])
weights = np.asarray([3.0, 1.0])

# ❌ Right only when the weights sum to 1
assert np.sum(weights * values) == 7.0

# ✅ The scale of the weights cancels
assert np.sum(weights * values) / np.sum(weights) == 1.75
```

### Constants take the input's dtype

```python
import numpy as np

THRESHOLD = 0.03


# ❌ A float64 array constant promotes a float32 input to float64
def excess_bad(density):
    return density - np.asarray(THRESHOLD, dtype=np.float64)


# ✅ The constant enters in the input's dtype
def excess(density):
    return density - np.asarray(THRESHOLD, dtype=density.dtype)


density = np.asarray([0.0, 0.5], dtype=np.float32)
assert excess_bad(density).dtype == np.float64
assert excess(density).dtype == np.float32
```

### Doctest output the code really prints

```python
import math

# ❌ Expected output copied from a calculator: the doctest fails on the
#    17th digit, or passes only on one platform
#     >>> buoyancy_frequency_of(slope=0.02)
#     0.0138

# ✅ Round in the example, and paste what it prints
#     >>> round(buoyancy_frequency_of(slope=0.02), 6)
#     0.013833
# N = √(g Δρ / (ρ₀ Δz))
assert round(math.sqrt(9.80665 * 0.02 / 1025.0), 6) == 0.013833
```


---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
