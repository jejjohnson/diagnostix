"""Times as offsets that keep their differences in a field's precision."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from diagnostix._src.reduce import on_device
from diagnostix._typing import ArrayLike


__all__ = ["time_offsets"]


def time_offsets(xp: ModuleType, time: ArrayLike, ref: Any) -> tuple[Any, Any]:
    """Times from their first one, in ``ref``'s dtype, and the first one.

    Seconds since 1970 have no seconds left in float32, and their differences
    do. Times on the host are subtracted in float64 before they take ``ref``'s
    dtype, and others in their own dtype; they never decide the dtype of a call.

    Args:
        xp: The namespace of the call.
        time: t, 1-D, in any unit.
        ref: An array of the namespace ``xp``.

    Returns:
        t − t₀, and t₀, both in ``ref``'s dtype, on its device.
    """
    if isinstance(time, (list, tuple, np.ndarray, np.generic)):
        host = np.asarray(time, dtype=np.float64)
        shifted = on_device(xp, host - host[:1], ref)
        origin = on_device(xp, host[:1], ref)
    else:
        shifted, origin = time - time[:1], time[:1]
    return xp.astype(shifted, ref.dtype), xp.astype(origin, ref.dtype)[0]
