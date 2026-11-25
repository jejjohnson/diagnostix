"""Exception hierarchy for `diagnostix`.

Every error raised by this package derives from
[`DiagnostixError`][diagnostix.DiagnostixError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The exception is the array backend, which is pipekit-array's: an input that is
no array of one known library raises pipekit's `BackendError`, and an input
that is not real, or an unlabelled array next to a labelled one, raises a plain
`ValueError`.

Both are a `ValueError`, as the base class is, so code written against the
standard library keeps working without importing anything from this package.
"""

from __future__ import annotations

from pipekit_array.arrays import BackendError


__all__ = [
    "BackendError",
    "DiagnostixError",
    "ValidationError",
]


class DiagnostixError(ValueError):
    """Base class for every exception raised by ``diagnostix``.

    Examples:
        >>> from diagnostix import DiagnostixError, ValidationError
        >>> issubclass(ValidationError, DiagnostixError)
        True
        >>> issubclass(DiagnostixError, ValueError)
        True
    """


class ValidationError(DiagnostixError):
    """Raised for a bad argument: its shape, a dimension name or its value."""
