"""Checks on random samples, bounded by the sample's own standard error."""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ["assert_mean"]


def assert_mean(sample: Any, expected: float, *, n_sigma: float = 4.0) -> None:
    """Assert that a sample's mean is the expected one, within its sampling error.

    The bound is ``n_sigma`` standard errors s/√n of the sample itself, so no
    tolerance is tuned to a seed: at 4 a right mean fails about once in 16,000
    draws. The entries must be independent, so average over anything that shares
    a draw first.

    Args:
        sample: Independent draws, of any shape.
        expected: The mean they should have.
        n_sigma: The bound, in standard errors.

    Raises:
        AssertionError: If the mean is further away, if a draw is not finite, or
            if the sample is too small or has no spread to bound it by.
    """
    draws = np.asarray(sample, dtype=np.float64).ravel()
    if draws.size < 30:
        raise AssertionError(f"{draws.size} draws are too few for a standard error")
    if not np.isfinite(draws).all():
        raise AssertionError("the sample holds a draw that is not finite")
    # s/√n
    standard_error = float(np.std(draws, ddof=1)) / np.sqrt(draws.size)
    if standard_error == 0.0:
        raise AssertionError("the sample has no spread, so compare its values exactly")
    distance = abs(float(np.mean(draws)) - expected) / standard_error
    if distance > n_sigma:
        raise AssertionError(
            f"mean {np.mean(draws):.6g} is {distance:.1f} standard errors "
            f"({standard_error:.3g} each) from {expected:.6g}, more than {n_sigma}"
        )
