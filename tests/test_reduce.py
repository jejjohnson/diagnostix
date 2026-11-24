"""Masked, weighted sums and means."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _backends import assert_backends_agree, assert_refuses_dataarrays
from _harness import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from diagnostix import BackendError, DiagnostixError
from diagnostix._src.reduce import weighted_mean, weighted_sum


RNG = np.random.default_rng(0)
X = RNG.normal(size=(3, 4, 5))
W = RNG.uniform(0.1, 2.0, size=(4, 5))
M = RNG.random(X.shape) > 0.3
# some sample is valid along every axis
M[0] = True
# no cancellation, so that float32 keeps its relative precision
POSITIVE = np.abs(X)


def sum_of(x: Any, w: Any, m: Any) -> Any:
    return weighted_sum(x, weights=w, valid=m, axis=(-2, -1))


def mean_of(x: Any, w: Any, m: Any) -> Any:
    return weighted_mean(x, weights=w, valid=m, axis=(-2, -1))


class TestClosedForm:
    @pytest.mark.parametrize("axis", [None, 0, -1, (0, 2), (-2, -1)])
    def test_against_numpy(self, axis: Any) -> None:
        weight = np.where(M, W, 0.0)
        total = np.sum(weight * X, axis=axis)
        # a dozen additions: a few units in the last place, relative to ∑ |w x|
        scale = np.sum(weight * np.abs(X), axis=axis)
        assert_matches_oracle(
            weighted_sum(X, weights=W, valid=M, axis=axis) / scale,
            total / scale,
            rtol=0.0,
            atol=1e-14,
        )
        assert_matches_oracle(
            weighted_mean(X, weights=W, valid=M, axis=axis) / scale,
            total / np.sum(weight, axis=axis) / scale,
            rtol=0.0,
            atol=1e-14,
        )

    def test_without_weights_or_a_mask(self) -> None:
        assert_matches_oracle(weighted_sum(X, axis=-1), X.sum(axis=-1), rtol=0.0)
        assert_matches_oracle(weighted_mean(X, axis=-1), X.mean(axis=-1), rtol=1e-15)
        assert_matches_oracle(weighted_mean(X), X.mean(), rtol=1e-14)

    def test_constant_weights_give_the_plain_mean(self) -> None:
        plain = weighted_mean(X, valid=M, axis=-1)
        for constant in (1.0, 7.3):
            weighted = weighted_mean(X, weights=constant, valid=M, axis=-1)
            assert_matches_oracle(weighted, plain, rtol=1e-14, atol=1e-15)

    def test_the_scale_of_the_weights_cancels(self) -> None:
        assert_matches_oracle(mean_of(X, 1e6 * W, M), mean_of(X, W, M), rtol=1e-13)

    def test_weights_and_masks_broadcast_past_the_values(self) -> None:
        series = X[0, 0]
        out = weighted_mean(series, weights=W, valid=M[0], axis=-1)
        assert out.shape == (4,)
        assert_matches_oracle(out, mean_of_rows(series, W, M[0]), rtol=1e-13)


def mean_of_rows(x: np.ndarray, w: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The weighted mean of each row, written as a loop."""
    return np.asarray(
        [
            np.average(x[valid], weights=weight[valid])
            for weight, valid in zip(w, m, strict=True)
        ]
    )


class TestMissing:
    def test_a_nan_in_an_invalid_sample_changes_nothing(self) -> None:
        spoiled = np.where(M, X, np.nan)
        assert_matches_oracle(sum_of(spoiled, W, M), sum_of(X, W, M), rtol=0.0)
        assert_matches_oracle(mean_of(spoiled, W, M), mean_of(X, W, M), rtol=0.0)

    def test_a_nan_weight_of_an_invalid_sample_changes_nothing(self) -> None:
        weights = np.where(M, W, np.nan)
        assert_matches_oracle(mean_of(X, weights, M), mean_of(X, W, M), rtol=0.0)
        assert_matches_oracle(sum_of(X, weights, M), sum_of(X, W, M), rtol=0.0)

    def test_a_nan_in_a_valid_sample_gives_nan(self) -> None:
        spoiled = X.copy()
        spoiled[1, 2, 3] = np.nan
        for out in (mean_of(spoiled, W, np.ones_like(M)), weighted_sum(spoiled)):
            assert np.isnan(out).any()
        assert not np.isnan(mean_of(spoiled, W, np.ones_like(M))[[0, 2]]).any()

    def test_no_weight_is_nan_and_an_empty_sum_is_zero(self) -> None:
        none = np.zeros_like(M)
        assert np.isnan(mean_of(X, W, none)).all()
        assert np.isnan(mean_of(X, np.zeros_like(W), M)).all()
        assert (sum_of(X, W, none) == 0.0).all()

    def test_the_gradient_is_finite_next_to_a_nan_and_an_empty_mean(self) -> None:
        valid = M.copy()
        valid[1] = False
        spoiled = np.where(valid, X, np.nan)
        for function in (sum_of, mean_of):
            assert_finite_gradient(lambda x, f=function: f(x, W, valid), spoiled)
        weights = np.where(valid[0], W, 0.0)
        assert_finite_gradient(lambda w: mean_of(X, w, valid), weights)


class TestBackends:
    @pytest.mark.parametrize("function", [sum_of, mean_of])
    def test_agree(self, function: Any) -> None:
        assert_backends_agree(function, POSITIVE, W, M)
        assert_traces(function, X, RNG.uniform(0.1, 2.0, size=X.shape), M)
        assert_refuses_dataarrays(function, X, W, M)

    @pytest.mark.parametrize("function", [sum_of, mean_of])
    def test_gradients(self, function: Any) -> None:
        assert_grad_matches_fd(lambda x: function(x, W, M), X)
        assert_grad_matches_fd(lambda w: function(X, w, M), W)

    @pytest.mark.jax
    def test_a_jax_mask_on_numpy_values_selects_jax(self) -> None:
        import jax

        out = jax.jit(lambda m: weighted_mean(X, valid=m, axis=-1))(M)
        assert_matches_oracle(out, weighted_mean(X, valid=M, axis=-1), rtol=1e-13)

    def test_a_mask_may_be_a_list_or_a_python_bool(self) -> None:
        assert weighted_sum([1.0, 2.0], valid=[True, False]) == 1.0
        assert weighted_sum([1.0, 2.0], valid=False) == 0.0

    def test_integers_become_floats(self) -> None:
        assert weighted_mean(np.asarray([1, 2])) == 1.5

    @pytest.mark.parametrize("valid", [[1.0, 0.0], [1, 0], np.asarray([0, 1])])
    def test_a_mask_that_is_not_boolean(self, valid: Any) -> None:
        with pytest.raises(DiagnostixError, match="'valid' must be boolean"):
            weighted_mean([1.0, 2.0], valid=valid)

    def test_a_traced_mask_that_is_not_boolean(self) -> None:
        jnp = pytest.importorskip("jax.numpy")
        with pytest.raises(DiagnostixError, match="'valid' must be boolean"):
            weighted_mean(X, valid=jnp.ones(5))

    def test_an_unknown_library(self) -> None:
        with pytest.raises(BackendError):
            weighted_mean(object())
