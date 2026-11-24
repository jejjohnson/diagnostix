"""The kinds of array a function must accept, and the check that they agree.

JAX, xarray and `array_api_strict` are each optional here: a kind whose
library is missing is left out, so the job without JAX and xarray runs the rest.
From radiomax (`tests/_backends.py` at v0.0.8, MIT).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Callable, Collection
from typing import Any

import numpy as np
import pytest
from _harness import assert_matches_oracle
from pipekit_array.functions import BackendError


__all__ = [
    "GRID_KINDS",
    "HAS_JAX",
    "HAS_STRICT",
    "HAS_XARRAY",
    "KINDS",
    "ArrayKind",
    "assert_backends_agree",
    "assert_refuses_dataarrays",
    "needs_jax",
    "needs_xarray",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_XARRAY = importlib.util.find_spec("xarray") is not None
HAS_STRICT = importlib.util.find_spec("array_api_strict") is not None

needs_jax = pytest.mark.jax
needs_xarray = pytest.mark.xarray

#: The kinds of a function that calls gridax, which does not run on
#: ``array_api_strict`` (jejjohnson/diagnostix#15).
GRID_KINDS = ("numpy", "jax")


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayKind:
    """One kind of array: how to make it, what comes back, and how to read it.

    ``host`` takes an array back to NumPy, and checks where it was.
    """

    name: str
    make: Callable[[np.ndarray], Any]
    type: Any
    host: Callable[[Any], np.ndarray] = np.asarray

    def read(self, array: Any) -> np.ndarray:
        """The array as NumPy, after checking that it is of this kind."""
        assert isinstance(array, self.type), (
            f"{self.name} in must give {self.name} out, got {type(array).__name__}"
        )
        return self.host(array)


def _dims(array: np.ndarray) -> tuple[str, ...]:
    """Dimension names from the right, so that arrays broadcast as NumPy does."""
    return tuple(f"dim{array.ndim - axis}" for axis in range(array.ndim))


def _labelled(array: np.ndarray) -> Any:
    """``array`` as a DataArray."""
    import xarray as xr

    return xr.DataArray(array, dims=_dims(np.asarray(array)))


def _kinds() -> list[ArrayKind]:
    kinds = [ArrayKind("numpy", np.asarray, (np.ndarray, np.generic))]
    if HAS_STRICT:
        import array_api_strict

        strict_type = type(array_api_strict.asarray(0.0))
        kinds.append(ArrayKind("strict", array_api_strict.asarray, strict_type))
        second = array_api_strict.Device("device1")

        def on_second(array: np.ndarray) -> Any:
            return array_api_strict.asarray(array, device=second)

        def from_second(array: Any) -> np.ndarray:
            assert array.device == second, f"came back on {array.device}"
            return np.asarray(array.to_device(array_api_strict.Device("CPU_DEVICE")))

        kinds.append(ArrayKind("second device", on_second, strict_type, from_second))
    if HAS_JAX:
        import jax
        import jax.numpy as jnp

        kinds.append(ArrayKind("jax", jnp.asarray, jax.Array))
    if HAS_XARRAY:
        import xarray as xr

        kinds.append(ArrayKind("xarray", _labelled, xr.DataArray))
    return kinds


KINDS = _kinds()


def assert_backends_agree(
    fn: Callable[..., Any],
    *args: Any,
    rtol: float = 1e-12,
    rtol32: float = 1e-5,
    kinds: Collection[str] | None = None,
    labelled: bool = False,
) -> None:
    """Check that every array kind gives NumPy's numbers, kind and dtype.

    For each kind: the same kind comes back, float64 values agree with NumPy's
    to ``rtol``, float32 in gives float32 out and agrees to ``rtol32``. JAX is
    also checked under ``jit``. Boolean and integer arguments keep their dtype,
    and boolean and integer outputs must match exactly.

    Args:
        fn: The function under test; it returns an array or a tuple of them.
        *args: NumPy arrays to call it with; the floating ones are float64.
        rtol: Relative tolerance in float64.
        rtol32: Relative tolerance of float32 against float64.
        kinds: The names of the kinds to check, `GRID_KINDS` for a function that
            calls gridax; ``None`` checks every kind.
        labelled: Also check DataArrays, for an elementwise function.

    Raises:
        AssertionError: If a check fails; the message names the kind.
    """
    arrays = [np.asarray(arg) for arg in args]
    expected = _outputs(fn(*arrays))
    for kind in KINDS:
        if kinds is not None and kind.name not in kinds:
            continue
        if kind.name == "xarray" and not labelled:
            continue
        variants = {kind.name: fn}
        if kind.name == "jax":
            import jax

            variants["jax.jit"] = jax.jit(fn)
        for what, variant in variants.items():
            for dtype, tolerance in ((np.float64, rtol), (np.float32, rtol32)):
                inputs = [kind.make(_cast(array, dtype)) for array in arrays]
                outputs = _outputs(variant(*inputs))
                for got, want in zip(outputs, expected, strict=True):
                    label = f"{what} {np.dtype(dtype).name}"
                    values = kind.read(got)
                    if np.issubdtype(want.dtype, np.floating):
                        assert values.dtype == dtype, (
                            f"{label}: {values.dtype} came back"
                        )
                    try:
                        assert_matches_oracle(values, want, rtol=tolerance)
                    except AssertionError as err:
                        raise AssertionError(f"{label}: {err}") from err


def assert_refuses_dataarrays(fn: Callable[..., Any], *args: Any) -> None:
    """Check that a DataArray in any argument raises `BackendError`.

    A function over a grid or an axis takes plain arrays, so labels are never
    lost silently. Skips without xarray.

    Args:
        fn: The function under test.
        *args: NumPy arrays to call it with; each is labelled in turn.

    Raises:
        AssertionError: If a call is accepted; the message names the argument.
    """
    pytest.importorskip("xarray")
    for position in range(len(args)):
        inputs = list(args)
        inputs[position] = _labelled(np.asarray(args[position]))
        try:
            fn(*inputs)
        except BackendError:
            continue
        raise AssertionError(f"argument {position} accepted a DataArray")


def _cast(array: np.ndarray, dtype: Any) -> np.ndarray:
    """A floating array in ``dtype``; a boolean or integer one as it is."""
    return array.astype(dtype) if np.issubdtype(array.dtype, np.floating) else array


def _outputs(result: Any) -> tuple[Any, ...]:
    return result if isinstance(result, tuple) else (result,)
