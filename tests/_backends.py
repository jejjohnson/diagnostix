"""The kinds of array a function must accept, and what a test needs to run.

The kinds, and the check that they agree, are pipekit-array's
(`pipekit_array.testing`). Each library but NumPy is optional: a kind whose library
is missing is left out, so the job without JAX and xarray runs the rest.
"""

from __future__ import annotations

import importlib.util

import pytest
from pipekit_array.testing import assert_backends_agree, assert_refuses_dataarrays


__all__ = [
    "EAGER_KINDS",
    "GRID_KINDS",
    "HAS_JAX",
    "assert_backends_agree",
    "assert_refuses_dataarrays",
    "needs_jax",
]

HAS_JAX = importlib.util.find_spec("jax") is not None

needs_jax = pytest.mark.jax

#: The kinds of a function that calls gridax, which does not run on
#: ``array_api_strict`` (jejjohnson/diagnostix#15).
GRID_KINDS = ("numpy", "jax")

#: The kinds of a function that does not run on dask, as its docstring says: dask
#: has no ``take_along_axis``, no FFT of a chunked axis and no batched solve.
EAGER_KINDS = ("numpy", "strict", "second device", "jax", "torch")
