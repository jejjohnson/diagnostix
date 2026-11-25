"""Scores of a prediction against its reference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _backends import assert_backends_agree, assert_refuses_dataarrays, needs_jax
from _harness import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import xrops_values
from hypothesis import given, strategies as st

from diagnostix import ValidationError
from diagnostix.skill import bias, correlation, mae, nrmse_score, r2_score, rmse


def noise(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def nrms_score(*args: Any, **options: Any) -> Any:
    return nrmse_score(*args, normalise="rms", **options)


def numpy_moments(p: Any, r: Any, w: Any) -> dict[str, float]:
    """Every score of two 1-D samples, with `numpy.average`."""

    def mean(x: Any) -> Any:
        return np.average(x, weights=w)

    mean_p, mean_r = mean(p), mean(r)
    var_p, var_r = mean((p - mean_p) ** 2), mean((r - mean_r) ** 2)
    mse = mean((p - r) ** 2)
    return {
        "rmse": np.sqrt(mse),
        "mae": mean(np.abs(p - r)),
        "bias": mean(p - r),
        "correlation": mean((p - mean_p) * (r - mean_r)) / np.sqrt(var_p * var_r),
        "r2": 1.0 - mse / var_r,
        "nrmse std": 1.0 - np.sqrt(mse / var_r),
        "nrmse rms": 1.0 - np.sqrt(mse / mean(r**2)),
    }


POINTWISE: dict[str, Callable[..., Any]] = {
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
    "correlation": correlation,
    "r2": r2_score,
    "nrmse std": nrmse_score,
    "nrmse rms": nrms_score,
}


@pytest.mark.parametrize("name", POINTWISE)
class TestPointwise:
    def test_numpy(self, name: str) -> None:
        p, r = noise(0, 2, 3, 20)
        r = 2.0 + r + 0.5 * p
        w = np.random.default_rng(1).random((3, 20))
        score = POINTWISE[name]
        expected = [numpy_moments(*one)[name] for one in zip(p, r, w, strict=True)]
        assert_matches_oracle(score(p, r, weights=w, axis=-1), expected, rtol=1e-11)
        # pooled, and without weights
        pooled = numpy_moments(p.ravel(), r.ravel(), np.ones(60))[name]
        assert float(score(p, r)) == pytest.approx(pooled, rel=1e-11)
        assert float(score(p, r, axis=(0, 1))) == pytest.approx(pooled, rel=1e-11)

    def test_gaps_are_the_score_of_what_is_left(self, name: str) -> None:
        # jejjohnson/geotoolz#777: each moment took its own cells
        p, r, w = noise(2, 3, 50)
        w = np.abs(w)
        p[:25] += 20.0
        valid = np.arange(50) % 3 != 0
        valid[:25] = False
        score = POINTWISE[name]
        expected = score(p[valid], r[valid], weights=w[valid])
        for hole in (np.nan, np.inf, 1e6):
            q = np.where(valid, p, hole), np.where(valid, r, hole)
            got = score(*q, weights=w, valid=valid)
            assert_matches_oracle(got, expected, rtol=1e-12)
            # a weight that is missing where the cell is
            got = score(*q, weights=np.where(valid, w, np.nan), valid=valid)
            assert_matches_oracle(got, expected, rtol=1e-12)

    def test_a_nan_in_a_valid_cell_is_nan(self, name: str) -> None:
        for field in (0, 1):
            pair = noise(3, 2, 4, 6)
            pair[field, 1, 2] = np.nan
            out = POINTWISE[name](*pair, axis=-1)
            assert np.isnan(out[1])
            assert np.isfinite(np.delete(out, 1)).all()

    def test_no_valid_cell_is_nan(self, name: str) -> None:
        p, r = noise(4, 2, 3, 5)
        valid = np.arange(3)[:, None] != 1
        out = POINTWISE[name](p, r, valid=valid, axis=-1)
        assert_matches_oracle(np.isnan(out), [False, True, False], rtol=0.0)
        assert np.isnan(POINTWISE[name](p, r, weights=0.0))

    def test_broadcasting(self, name: str) -> None:
        p, r = noise(5, 4, 3, 6), noise(6, 6)
        w = np.abs(noise(7, 3, 1))
        out = POINTWISE[name](p, r, weights=w, axis=(1, 2))
        one = POINTWISE[name](
            p[2], np.broadcast_to(r, (3, 6)), weights=np.broadcast_to(w, (3, 6))
        )
        assert out.shape == (4,)
        assert_matches_oracle(out[2], one, rtol=1e-12)

    @pytest.mark.parametrize("value", [0.0, 0.1, 1e6])
    def test_a_constant_reference(self, name: str, value: float) -> None:
        # 0.1 is not a float, so its mean over 7 cells is not 0.1
        r = np.full(7, value)
        score = POINTWISE[name]
        perfect, off = score(r, r), score(r + 1.0, r)
        if name == "correlation":
            assert np.isnan(perfect)
            assert np.isnan(score(noise(8, 7), r))
            assert np.isnan(score(r, noise(8, 7)))
        elif name in ("r2", "nrmse std") or (name == "nrmse rms" and value == 0.0):
            assert perfect == 1.0
            assert off == -np.inf
        else:
            assert perfect == (1.0 if name == "nrmse rms" else 0.0)

    @pytest.mark.slow
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_backends_and_precision(self, name: str, with_valid: bool) -> None:
        def of(p: Any, r: Any, w: Any, m: Any) -> Any:
            mask = m if with_valid else None
            return 9.0 + POINTWISE[name](p, r, weights=w, valid=mask, axis=-1)

        p, r, w = 1.0 + np.random.default_rng(9).random((3, 2, 12))
        m = np.random.default_rng(10).random((2, 12)) > 0.2
        assert_backends_agree(of, p, r, w, m, rtol32=2e-5)
        assert_traces(of, p[None], r[None], w[None], m[None])
        assert_refuses_dataarrays(lambda p, r, w: of(p, r, w, m), p, r, w)
        assert_grad_matches_fd(lambda p: of(p, r, w, m), p, rtol=1e-5)
        assert_grad_matches_fd(lambda r: of(p, r, w, m), r, rtol=1e-5)
        assert_grad_matches_fd(lambda w: of(p, r, w, m), w, rtol=1e-5)

    @needs_jax
    def test_gradient_at_a_perfect_prediction(self, name: str) -> None:
        r = noise(11, 8)
        score = POINTWISE[name]
        assert_finite_gradient(lambda p: score(p, r), r)
        # and where the reference, or everything, is constant
        constant = np.ones(8)
        assert_finite_gradient(lambda p: score(p, constant), constant)
        assert_finite_gradient(lambda r: score(constant, r), constant)

    @needs_jax
    def test_gradient_beside_a_gap(self, name: str) -> None:
        p, r = noise(12, 2, 3, 6)
        valid = np.arange(6) != 2
        p, r = (np.where(valid, a, np.nan) for a in (p, r))
        score = POINTWISE[name]

        def of(p: Any) -> Any:
            return score(p, r, valid=valid, axis=-1)

        assert_finite_gradient(of, p)

    def test_errors(self, name: str) -> None:
        p, r = noise(13, 2, 5)
        with pytest.raises(ValidationError, match="no axis 1"):
            POINTWISE[name](p, r, axis=1)
        with pytest.raises(ValueError, match="'valid' must be boolean"):
            POINTWISE[name](p, r, valid=np.ones(5))


class TestStored:
    def test_xrops(self) -> None:
        stored = xrops_values("skill")
        pair = stored["prediction"], stored["reference"]
        for name, score in (
            ("rmse", rmse),
            ("mae", mae),
            ("bias", bias),
            ("nrmse_rms", nrms_score),
            ("nrmse_std", nrmse_score),
        ):
            assert_matches_oracle(score(*pair, axis=-1), stored[name], rtol=1e-12)

    @pytest.mark.oracle
    def test_scikit_learn(self) -> None:
        metrics = pytest.importorskip("sklearn.metrics")
        p, r, w = noise(14, 3, 40)
        w = np.abs(w)
        expected = metrics.r2_score(r, p, sample_weight=w)
        assert float(r2_score(p, r, weights=w)) == pytest.approx(expected, rel=1e-12)

    @given(st.integers(0, 1000), st.floats(0.1, 10.0), st.floats(-5.0, 5.0))
    def test_correlation_is_that_of_a_line(
        self, seed: int, slope: float, offset: float
    ) -> None:
        x = noise(seed, 20)
        assert float(correlation(x, offset + slope * x)) == pytest.approx(1.0, abs=1e-9)
        assert float(correlation(x, offset - slope * x)) == pytest.approx(
            -1.0, abs=1e-9
        )

    def test_small_fields_in_float32(self) -> None:
        p, r = noise(15, 2, 30)
        expected = float(correlation(p, r))
        small = [(1e-15 * a).astype(np.float32) for a in (p, r)]
        assert float(correlation(*small)) == pytest.approx(expected, rel=1e-4)

    def test_area_weights(self) -> None:
        # an error at the pole counts for less than one at the equator
        latitude = np.linspace(-85.0, 85.0, 18)
        w = np.cos(np.deg2rad(latitude))[:, None]
        r = noise(16, 18, 36)
        at_pole, at_equator = r.copy(), r.copy()
        at_pole[0] += 1.0
        at_equator[9] += 1.0
        assert rmse(at_pole, r, weights=w) < 0.3 * rmse(at_equator, r, weights=w)
        assert rmse(at_pole, r) == pytest.approx(rmse(at_equator, r), rel=1e-12)

    def test_normalise(self) -> None:
        with pytest.raises(ValidationError, match="'normalise' must be"):
            nrmse_score([1.0], [1.0], normalise="range")
