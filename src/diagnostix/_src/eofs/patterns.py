"""Area-weighted empirical orthogonal functions, and fields in terms of them."""

from __future__ import annotations

import dataclasses
import math
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Bool, Float, Int
from pipekit_array.functions import (
    boolean_like,
    divide_or_nan,
    floating_inputs,
    table_like,
)
from pipekit_array.records import register_record

from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["EOFs", "eofs", "project", "reconstruct"]


@register_record
@dataclasses.dataclass(frozen=True, eq=False)
class EOFs:
    """The patterns that explain most of an anomaly field's variance per area.

    Attributes:
        modes: e, orthonormal under the weights: ∑ w eₘ eₙ is 1 for m = n and 0
            otherwise. NaN in a cell that is not valid, or has no weight.
        principal_components: a, the amplitude of each mode at each time, in the
            unit of the anomaly times that of √w.
        explained_variance: λ, the share of the weighted variance of each mode,
            descending. The shares of all modes sum to 1.
    """

    modes: Float[Any, "mode ..."]
    principal_components: Float[Any, "time mode"]
    explained_variance: Float[Any, " mode"]


def eofs(
    anomaly: ArrayLike,
    *,
    n_modes: int,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> EOFs:
    """The leading area-weighted EOFs of an anomaly field.

    X diag(√w) = U Σ Vᵀ, the thin singular value decomposition; the modes are V
    with √w divided out, the principal components U Σ, and mode m explains the
    share σₘ² / ∑ⱼ σⱼ².

    Nothing is centred: ``anomaly`` and ``detrend`` of ``diagnostix.climate`` come
    first. The weights are not normalised; c times the weights gives modes over
    √c and principal components times it. Of each mode, the loading that is
    largest in size is positive.

    The cost is that of a decomposition of T × C numbers. The derivative of
    ``explained_variance`` is finite for distinct singular values, and that of
    the modes grows without bound as two of them meet.

    Args:
        anomaly: X, with the T times first and then the grid.
        n_modes: The number of modes, at most the smaller of T and the number of
            valid cells.
        weights: w ≥ 0, broadcast against the grid; 1 if not given. Cell areas
            come from ``gridax.area_weights``.
        valid: Which cells to use (True = use), concrete, broadcast against the
            grid; all if not given. A NaN in a valid cell makes every output NaN.

    Returns:
        The modes, their principal components and the shares they explain.

    Raises:
        BackendError: If an array is a DataArray or of another library.
        ValidationError: If ``anomaly`` has no grid, ``n_modes`` is out of range,
            ``weights`` or ``valid`` does not broadcast to the grid, or ``valid``
            leaves no cell.
        ValueError: If ``valid`` is not boolean.

    Examples:
        >>> import numpy as np
        >>> time = np.asarray([1.0, -1.0, 1.0, -1.0])
        >>> pattern = np.asarray([[2.0, 0.0], [-1.0, np.nan]])
        >>> out = eofs(
        ...     time[:, None, None] * pattern,
        ...     n_modes=1,
        ...     weights=[[1.0], [4.0]],
        ...     valid=~np.isnan(pattern),
        ... )
        >>> np.round(out.modes[0] * np.sqrt(8.0), 9) + 0.0
        array([[ 2.,  0.],
               [-1., nan]])
        >>> np.round(out.principal_components[:, 0] / np.sqrt(8.0), 9)
        array([ 1., -1.,  1., -1.])
        >>> np.round(out.explained_variance, 9)
        array([1.])
    """
    xp, (x, *given) = floating_inputs(anomaly, *_present(weights))
    if x.ndim < 2:
        msg = f"'anomaly' must have a time axis and a grid, got shape {x.shape}"
        raise ValidationError(msg)
    grid = tuple(x.shape[1:])
    cells = _Cells.of_grid(grid, valid)
    n_times = x.shape[0]
    if not 1 <= n_modes <= min(n_times, cells.index.size):
        msg = (
            f"'n_modes' must be between 1 and the smaller of {n_times} times and "
            f"{cells.index.size} cells, got {n_modes}"
        )
        raise ValidationError(msg)
    x = cells.take(xp, x)
    w = cells.weights(xp, given, x)
    positive = w > 0
    # √w and its inverse, both with a finite gradient where w is 0
    root = xp.sqrt(xp.where(positive, w, xp.ones_like(w)))
    used = xp.where(positive, x * root, xp.zeros_like(x))
    # NumPy's decomposition raises for a NaN and JAX's does not
    finite = xp.all(xp.isfinite(used))
    used = xp.where(finite, used, xp.zeros_like(used))
    # X diag(√w) = U Σ Vᵀ
    u, sigma, v_transposed = xp.linalg.svd(used, full_matrices=False)
    modes = v_transposed[:n_modes, :] / root
    modes = xp.where(positive, modes, xp.zeros_like(modes))
    sign = _sign_of_largest(xp, modes)
    amplitudes = u[:, :n_modes] * sigma[:n_modes] * sign
    # λₘ = σₘ² / ∑ⱼ σⱼ²
    share = divide_or_nan(sigma[:n_modes] ** 2, xp.sum(sigma**2), xp)
    modes = xp.where(positive, modes * sign[:, None], xp.full_like(modes, xp.nan))
    modes, amplitudes, share = (
        xp.where(finite, part, xp.full_like(part, xp.nan))
        for part in (modes, amplitudes, share)
    )
    return EOFs(
        modes=cells.spread(xp, modes),
        principal_components=amplitudes,
        explained_variance=share,
    )


def project(
    field: ArrayLike,
    eofs: EOFs,
    *,
    weights: ArrayLike | None = None,
    valid: ArrayLike | None = None,
) -> Any:
    """The amplitude of each mode in a field.

    aₘ = ∑ w x eₘ over the valid cells. For the field that the modes came from,
    these are its principal components.

    Args:
        field: x, anomalies as those the modes came from, that end in the grid.
        eofs: The modes.
        weights: w, the weights the modes were made with.
        valid: The cells the modes were made with, concrete. A NaN in a valid
            cell gives NaN.

    Returns:
        a, with the modes on the last axis in place of the grid.

    Raises:
        BackendError: If an array is a DataArray or of another library.
        ValidationError: If ``field`` does not end in the grid of the modes, or
            as `eofs`.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0], [2.0, 0.0, 1.0]])
        >>> out = eofs(x, n_modes=3)
        >>> np.allclose(project(x, out), out.principal_components)
        True
        >>> np.allclose(reconstruct(out, project(x[1], out)), x[1])
        True
    """
    xp, (x, modes, *given) = floating_inputs(field, eofs.modes, *_present(weights))
    grid = tuple(modes.shape[1:])
    if tuple(x.shape[-len(grid) :]) != grid:
        msg = f"'field' must end in the grid of the modes, {grid}, got {x.shape}"
        raise ValidationError(msg)
    cells = _Cells.of_grid(grid, valid)
    x, modes = cells.take(xp, x), cells.take(xp, modes)
    w = cells.weights(xp, given, x)
    positive = w > 0
    # a mode is NaN where it has no weight, and so counts for nothing
    modes = xp.where(positive, modes, xp.zeros_like(modes))
    x = xp.where(positive, x * w, xp.zeros_like(x))
    return xp.matmul(x, xp.matrix_transpose(modes))


def reconstruct(eofs: EOFs, principal_components: ArrayLike | None = None) -> Any:
    """The field that some amplitudes of the modes add up to.

    x = ∑ₘ aₘ eₘ.

    Args:
        eofs: The modes.
        principal_components: a, with the modes on the last axis; those of
            ``eofs`` if not given. A mode set to 0 is left out.

    Returns:
        x, with the grid in place of the modes, NaN where the modes are.

    Raises:
        BackendError: If an array is a DataArray or of another library.
        ValidationError: If there is not one amplitude per mode.

    Examples:
        >>> import numpy as np
        >>> x = np.asarray([[1.0, 2.0], [-1.0, 0.5], [0.0, -2.5]])
        >>> np.allclose(reconstruct(eofs(x, n_modes=2)), x)
        True
    """
    given = principal_components
    amplitudes = eofs.principal_components if given is None else given
    xp, (a, modes) = floating_inputs(amplitudes, eofs.modes)
    n_modes, *grid = modes.shape
    if a.ndim < 1 or a.shape[-1] != n_modes:
        msg = (
            f"'principal_components' must hold {n_modes} modes on the last axis, "
            f"got shape {a.shape}"
        )
        raise ValidationError(msg)
    # x = ∑ₘ aₘ eₘ
    x = xp.matmul(a, xp.reshape(modes, (n_modes, -1)))
    return xp.reshape(x, (*a.shape[:-1], *grid))


def _present(value: ArrayLike | None) -> tuple[ArrayLike, ...]:
    """The value, in a tuple, or no value."""
    return () if value is None else (value,)


def _sign_of_largest(xp: ModuleType, modes: Any) -> Any:
    """The sign of the loading of each mode that is largest in size."""
    largest = xp.max(modes, axis=-1) + xp.min(modes, axis=-1)
    one = xp.ones_like(largest)
    return xp.where(largest < 0, -one, one)


@dataclasses.dataclass(frozen=True)
class _Cells:
    """The valid cells of a grid, fixed on the host.

    Attributes:
        grid: The shape of the grid.
        valid: Which cells are used.
        index: Their positions in the flat grid.
    """

    grid: tuple[int, ...]
    valid: Bool[np.ndarray, "*grid"]
    index: Int[np.ndarray, " cell"]

    @classmethod
    def of_grid(cls, grid: tuple[int, ...], valid: ArrayLike | None) -> _Cells:
        """The valid cells of a grid.

        Raises:
            ValidationError: As `eofs`.
        """
        mask = np.ones(grid, dtype=bool)
        if valid is not None:
            given = boolean_like(np.asarray(valid), mask, np)
            try:
                mask = np.broadcast_to(given, grid)
            except ValueError as err:
                msg = f"'valid' of shape {given.shape} is not of the grid {grid}"
                raise ValidationError(msg) from err
        if not mask.any():
            msg = "'valid' leaves no cell"
            raise ValidationError(msg)
        return cls(grid=grid, valid=mask, index=np.flatnonzero(mask))

    @property
    def size(self) -> int:
        """The number of cells of the grid."""
        return math.prod(self.grid)

    def take(self, xp: ModuleType, x: Any) -> Any:
        """The valid cells of an array that ends in the grid, (…, cell)."""
        flat = xp.reshape(x, (*x.shape[: x.ndim - len(self.grid)], self.size))
        if self.index.size == self.size:
            return flat
        return xp.take(flat, table_like(self.index, x, xp), axis=-1)

    def weights(self, xp: ModuleType, given: list[Any], x: Any) -> Any:
        """The weight of each valid cell, 1 if none is given.

        Raises:
            ValidationError: If the weights do not broadcast to the grid.
        """
        if not given:
            return xp.ones_like(x[(0,) * (x.ndim - 1)])
        (w,) = given
        try:
            fits = np.broadcast_shapes(tuple(w.shape), self.grid) == self.grid
        except ValueError:
            fits = False
        if not fits:
            msg = f"'weights' of shape {tuple(w.shape)} is not of the grid {self.grid}"
            raise ValidationError(msg)
        return self.take(xp, xp.broadcast_to(w, self.grid))

    def spread(self, xp: ModuleType, values: Any) -> Any:
        """(…, cell) values back on the grid, NaN in the other cells."""
        shape = (*values.shape[:-1], *self.grid)
        if self.index.size == self.size:
            return xp.reshape(values, shape)
        position = np.zeros(self.size, dtype=np.int64)
        position[self.index] = np.arange(self.index.size)
        full = xp.take(values, table_like(position, values, xp), axis=-1)
        inside = table_like(self.valid.reshape(-1), values, xp)
        full = xp.where(inside, full, xp.full_like(full, xp.nan))
        return xp.reshape(full, shape)
