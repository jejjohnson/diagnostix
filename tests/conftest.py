"""Shared pytest configuration.

JAX and xarray are optional. When JAX is importable, 64-bit mode is switched on
so that float64 reference comparisons are meaningful; tests marked ``jax`` or
``xarray`` are skipped when that library is missing.
"""

from __future__ import annotations

import importlib.util

import pytest


try:
    import jax
except ImportError:  # pragma: no cover - exercised in the job without JAX
    pass
else:
    jax.config.update("jax_enable_x64", True)

_OPTIONAL = ("jax", "xarray")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests whose marker names a library that is not installed."""
    for name in _OPTIONAL:
        if importlib.util.find_spec(name) is not None:
            continue
        missing = pytest.mark.skip(reason=f"{name} is not installed")
        for item in items:
            if item.get_closest_marker(name) is not None:
                item.add_marker(missing)
