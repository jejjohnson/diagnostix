"""What importing the package loads: the layer rule, as a test.

Each check runs in a fresh interpreter, because this one has already imported
whatever the other tests needed.
"""

from __future__ import annotations

import json
import subprocess
import sys


#: The stack libraries diagnostix may import at runtime (AGENTS.md, "Boundaries").
ALLOWED = {"geophysix", "gridax", "pipekit_array", "pipekit_ops"}

#: Every import package of the stack, so that a new one is noticed by name.
STACK = ALLOWED | {
    "argminax",
    "filterax",
    "functax",
    "gaussx",
    "geocatalog",
    "geocloud",
    "geonnax",
    "geoops",
    "geopatcher",
    "geoproducts",
    "geotoolz",
    "integrix",
    "kernellib",
    "pipekit",
    "pipekit_cycle",
    "pipekit_jax",
    "plumax",
    "pyrox",
    "radiomax",
    "radtrax",
    "sesmax",
    "somax",
    "spectraldiffx",
    "tracerax",
    "vardax",
    "xr_toolz",
    "xrops",
    "xtremax",
}

_IMPORT_EVERYTHING = """
import importlib, json, pkgutil, sys
import diagnostix
for info in pkgutil.walk_packages(diagnostix.__path__, "diagnostix."):
    importlib.import_module(info.name)
print(json.dumps(sorted({name.split(".")[0] for name in sys.modules})))
"""


def _loaded_by_importing_everything() -> set[str]:
    """The top-level packages loaded once every module of the package is."""
    run = subprocess.run(
        [sys.executable, "-c", _IMPORT_EVERYTHING],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(run.stdout))


def test_importing_the_package_loads_neither_jax_nor_xarray() -> None:
    assert not _loaded_by_importing_everything() & {"jax", "xarray"}


def test_importing_the_package_loads_only_the_allowed_stack_libraries() -> None:
    loaded = _loaded_by_importing_everything()
    assert loaded & STACK <= ALLOWED
    # the backend is always loaded, so an empty result cannot pass for a right one
    assert "pipekit_array" in loaded
