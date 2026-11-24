"""Sums and means over groups of samples, on any array library.

The array API standard has no ``bincount`` and no ``segment_sum``. A padded
gather needs neither: each group's samples are read into one row of a table, and
the rows are summed. It keeps float32's precision, which a difference of
cumulative sums does not (jejjohnson/diagnostix#15).
"""

from __future__ import annotations

import dataclasses
from types import ModuleType
from typing import Any

import numpy as np
from jaxtyping import Bool, Int
from pipekit_array.functions import floating_inputs

from diagnostix._src.reduce import (
    divide_or_nan,
    on_device,
    reduction_inputs,
    used_weights,
)
from diagnostix._typing import ArrayLike
from diagnostix.exceptions import ValidationError


__all__ = ["Groups"]

#: The label of a sample that is in no group.
NO_GROUP = -1


@dataclasses.dataclass(frozen=True, eq=False)
class Groups:
    """Which samples belong to which group; built once, on the host.

    The tables are NumPy arrays and static: under JAX a `Groups` is a closure or
    a static argument, and never a traced one. It is compared by identity, so
    build it once; a new one compiles again.

    G groups of at most M samples cost G × M reads, so one very large group
    among many small ones costs memory.

    Attributes:
        index: Positions of each group's samples, padded with 0.
        member: Which entries of ``index`` are real.
        labels: The group of each sample, −1 for none.

    Examples:
        >>> import numpy as np
        >>> groups = Groups.from_labels([0, 1, 0, -1], n_groups=3)
        >>> x = np.asarray([1.0, 2.0, 3.0, 100.0])
        >>> groups.sum(x)
        array([4., 2., 0.])
        >>> groups.mean(x)
        array([ 2.,  2., nan])
        >>> groups.count(x)
        array([2, 1, 0], dtype=int32)
        >>> x - groups.spread(groups.mean(x))
        array([-1.,  0.,  1., nan])
    """

    index: Int[np.ndarray, "group member"]
    member: Bool[np.ndarray, "group member"]
    labels: Int[np.ndarray, " sample"]

    @classmethod
    def from_labels(cls, labels: ArrayLike, n_groups: int) -> Groups:
        """The groups of samples with the same label.

        Args:
            labels: The group of each sample, concrete integers in [0, G), or
                −1 for a sample in no group.
            n_groups: G, at least 1; a group may be empty.

        Returns:
            The groups.

        Raises:
            ValidationError: If ``labels`` is not a 1-D array of integers, if
                one is neither −1 nor in [0, G), or if ``n_groups`` is below 1.
        """
        labels = np.asarray(labels)
        if n_groups < 1:
            msg = f"'n_groups' must be at least 1, got {n_groups}"
            raise ValidationError(msg)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            msg = (
                "'labels' must be a 1-D array of integers, got shape "
                f"{labels.shape} and dtype {labels.dtype}"
            )
            raise ValidationError(msg)
        # a copy: the caller may write to its array later
        labels = labels.astype(np.int64)
        outside = (labels < NO_GROUP) | (labels >= n_groups)
        if outside.any():
            # segment_sum would drop it silently
            msg = (
                f"'labels' must be -1 or in [0, {n_groups}), got "
                f"{labels[outside][0]} at position {int(np.argmax(outside))}"
            )
            raise ValidationError(msg)

        # --- the samples in a group, in group order ---
        position = np.flatnonzero(labels != NO_GROUP)
        order = position[np.argsort(labels[position], kind="stable")]
        sizes = np.bincount(labels[position], minlength=n_groups)

        # --- one padded row per group ---
        member = np.arange(int(sizes.max())) < sizes[:, None]
        index = np.zeros(member.shape, dtype=np.int64)
        # row by row, as `order` is
        index[member] = order
        for table in (index, member, labels):
            table.setflags(write=False)
        return cls(index=index, member=member, labels=labels)

    @property
    def n_groups(self) -> int:
        """G, the number of groups, empty ones included."""
        return self.index.shape[0]

    @property
    def n_samples(self) -> int:
        """N, the number of samples, those in no group included."""
        return self.labels.shape[0]

    def sum(
        self, values: ArrayLike, *, axis: int = -1, valid: ArrayLike | None = None
    ) -> Any:
        """The sum of each group's valid samples.

        Sⱼ = ∑_{n : g(n) = j} mₙ xₙ, with g(n) the group of sample n and m the
        validity, 1 or 0.

        Args:
            values: x, with the N samples on ``axis``.
            axis: The axis of the samples.
            valid: Which samples to use (True = use), broadcast against
                ``values``. A NaN in a valid sample gives NaN.

        Returns:
            S, with the G groups on ``axis``. An empty group has sum 0.

        Raises:
            BackendError: If an argument is a DataArray or of another library.
            ValidationError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, _, m = self._inputs(values, None, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        kept = xp.where(used, x_rows, xp.zeros_like(x_rows))
        # Sⱼ = ∑ mₙ xₙ over the row of group j
        return xp.moveaxis(xp.sum(kept, axis=-1, dtype=x.dtype), -1, axis)

    def count(
        self, values: ArrayLike, *, axis: int = -1, valid: ArrayLike | None = None
    ) -> Any:
        """The number of valid samples in each group.

        Args:
            values: x, with the N samples on ``axis``; only its shape, its
                library and its device matter, so a NaN counts.
            axis: The axis of the samples.
            valid: Which samples to count (True = count), broadcast against
                ``values``.

        Returns:
            The counts, int32, with the G groups on ``axis``.

        Raises:
            BackendError: If an argument is a DataArray or of another library.
            ValidationError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, _, m = self._inputs(values, None, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        ones = xp.astype(xp.broadcast_to(used, x_rows.shape), xp.int32)
        return xp.moveaxis(xp.sum(ones, axis=-1, dtype=xp.int32), -1, axis)

    def mean(
        self,
        values: ArrayLike,
        *,
        axis: int = -1,
        valid: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> Any:
        """The mean of each group's valid samples, weighted by w.

        x̄ⱼ = Sⱼ / ∑_{n : g(n) = j} mₙ wₙ, with Sⱼ = ∑_{n : g(n) = j} mₙ wₙ xₙ.

        Args:
            values: x, with the N samples on ``axis``.
            axis: The axis of the samples.
            valid: Which samples to use (True = use), broadcast against
                ``values``. A NaN in a valid sample gives NaN.
            weights: w ≥ 0, broadcast against ``values``; they need not sum to 1.

        Returns:
            x̄, with the G groups on ``axis``, NaN where a group is empty or its
            weights sum to 0.

        Raises:
            BackendError: If an argument is a DataArray or of another library.
            ValidationError: If ``axis`` does not hold N samples, or ``valid``
                is not boolean.
        """
        xp, x, w, m = self._inputs(values, weights, valid, axis)
        x_rows, used = self._rows(xp, x, m)
        used = xp.broadcast_to(used, x_rows.shape)
        w_rows = None if w is None else self._gather(xp, w)
        # mₙ wₙ, 0 on the padding too
        weight_rows = used_weights(xp, x_rows, w_rows, used)
        kept = xp.where(used, x_rows, xp.zeros_like(x_rows))
        # Sⱼ = ∑ mₙ wₙ xₙ over the row of group j
        total = xp.sum(kept * weight_rows, axis=-1, dtype=x.dtype)
        weight = xp.sum(weight_rows, axis=-1, dtype=x.dtype)
        return xp.moveaxis(divide_or_nan(xp, total, weight), -1, axis)

    def spread(self, grouped: ArrayLike, *, axis: int = -1) -> Any:
        """Each sample's group value: the inverse lookup.

        yₙ = x_{g(n)}, so that ``x - groups.spread(groups.mean(x))`` is the
        anomaly from the group's mean.

        Args:
            grouped: One value per group, on ``axis``.
            axis: The axis of the groups.

        Returns:
            y, with the N samples on ``axis``, NaN for a sample in no group.

        Raises:
            BackendError: If ``grouped`` is a DataArray.
            ValidationError: If ``axis`` does not hold G groups.
        """
        xp, (x,) = floating_inputs(grouped)
        self._check_axis(x, axis, self.n_groups, "groups")
        grouped_last = xp.moveaxis(x, axis, -1)
        labelled = self.labels != NO_GROUP
        # yₙ = x_{g(n)}; a sample in no group reads group 0, and is NaN below
        lookup = on_device(xp, np.where(labelled, self.labels, 0), x)
        values = xp.take(grouped_last, lookup, axis=-1)
        in_a_group = on_device(xp, labelled, x)
        spread = xp.where(in_a_group, values, xp.full_like(values, xp.nan))
        return xp.moveaxis(spread, -1, axis)

    # --- the gather -----------------------------------------------------------

    def _inputs(
        self,
        values: ArrayLike,
        weights: ArrayLike | None,
        valid: ArrayLike | None,
        axis: int,
    ) -> tuple[ModuleType, Any, Any, Any]:
        """The namespace, and x, w and m with the samples on the last axis."""
        xp, *arrays = reduction_inputs(values, weights, valid)
        self._check_axis(arrays[0], axis, self.n_samples, "samples")
        x, w, m = (
            None if array is None else xp.moveaxis(array, axis, -1) for array in arrays
        )
        return xp, x, w, m

    @staticmethod
    def _check_axis(x: Any, axis: int, size: int, what: str) -> None:
        """Refuse an axis of another size: a gather would not notice it."""
        if not -x.ndim <= axis < x.ndim or x.shape[axis] != size:
            msg = f"axis {axis} of shape {x.shape} must hold the {size} {what}"
            raise ValidationError(msg)

    def _gather(self, xp: ModuleType, x: Any) -> Any:
        """(…, N) samples as (…, G, M) rows, padding included."""
        flat = on_device(xp, self.index.reshape(-1), x)
        rows = xp.take(x, flat, axis=-1)
        return xp.reshape(rows, (*x.shape[:-1], *self.index.shape))

    def _rows(self, xp: ModuleType, x: Any, m: Any) -> tuple[Any, Any]:
        """The rows of x, and which of their entries are real and valid."""
        x_rows = self._gather(xp, x)
        used = on_device(xp, self.member, x)
        if m is not None:
            used = xp.logical_and(used, self._gather(xp, m))
        return x_rows, used
