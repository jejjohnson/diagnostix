"""Scores of a prediction against its reference."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import gridax
import numpy as np
import pytest
from _backends import (
    EAGER_KINDS,
    GRID_KINDS,
    assert_backends_agree,
    assert_refuses_dataarrays,
    needs_jax,
)
from _oracles import reliable_ensemble, xrops_values, xskillscore_crps
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_mean,
    assert_traces,
)

from diagnostix import ValidationError
from diagnostix.kinematics import geostrophic_residual
from diagnostix.skill import (
    PSDScore,
    band_rmse,
    bias,
    correlation,
    coverage,
    crps,
    density_inversion_fraction,
    diebold_mariano,
    energy_score,
    geostrophic_imbalance,
    mae,
    nrmse_score,
    psd_score,
    qg_balance_residual,
    r2_score,
    rank_histogram,
    reliability,
    resolved_scale,
    rms_divergence,
    rmse,
    spread_skill,
)
from diagnostix.spectra import isotropic_spectrum, power_spectrum


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


def pairwise(x: Any, y: Any, fair: bool) -> np.ndarray:
    """The CRPS from its definition, members first."""
    n = x.shape[0]
    error = np.mean(np.abs(x - y), axis=0)
    pairs = np.sum(np.abs(x[:, None] - x[None]), axis=(0, 1))
    return error - pairs / (2.0 * n * (n - 1 if fair else n))


SIZES = [2, 3, 5, 20]


class TestCrps:
    @pytest.mark.parametrize("n_members", SIZES)
    @pytest.mark.parametrize("fair", [True, False])
    def test_the_sum_over_pairs(self, n_members: int, fair: bool) -> None:
        x, y = noise(20, n_members, 6, 4), noise(21, 6, 4)
        expected = pairwise(x, y, fair)
        assert_matches_oracle(crps(x, y, fair=fair), expected, rtol=0.0, atol=1e-13)
        moved = np.moveaxis(x, 0, 1)
        out = crps(moved, y, member_axis=1, fair=fair)
        assert_matches_oracle(out, expected, rtol=0.0, atol=1e-13)

    @pytest.mark.oracle
    @pytest.mark.parametrize("n_members", [1, 5])
    def test_xskillscore(self, n_members: int) -> None:
        ensemble, truth, _ = reliable_ensemble(n_members, 60)
        out = crps(ensemble, truth, member_axis=-1, fair=False)
        assert_matches_oracle(
            out, xskillscore_crps(truth, ensemble), rtol=0.0, atol=1e-13
        )

    @pytest.mark.parametrize("n_members", SIZES)
    def test_a_fair_score_does_not_depend_on_the_members(self, n_members: int) -> None:
        # of a standard Gaussian: 1/√π
        x, y = noise(22, n_members, 20_000), noise(23, 20_000)
        assert_mean(crps(x, y), 1.0 / np.sqrt(np.pi))
        # and the other one by (E + 1)/E of it
        expected = (n_members + 1) / n_members / np.sqrt(np.pi)
        assert_mean(crps(x, y, fair=False), expected)

    def test_one_member(self) -> None:
        x, y = noise(24, 2, 1, 5)
        out = crps(x, y[0], fair=False)
        assert_matches_oracle(out, np.abs(x[0] - y[0]), rtol=1e-14)
        with pytest.raises(ValidationError, match="a fair score needs at least 2"):
            crps(x, y[0])

    def test_a_perfect_ensemble(self) -> None:
        y = noise(25, 5)
        x = np.broadcast_to(y, (4, 5))
        for fair in (True, False):
            out = crps(x, y, fair=fair)
            assert_matches_oracle(out, np.zeros(5), rtol=0.0, atol=1e-15)
            assert_finite_gradient(lambda x, fair=fair: crps(x, y, fair=fair), x)

    def test_a_nan_is_in_its_case_only(self) -> None:
        x, y = noise(26, 5, 4), noise(27, 4)
        for hole in range(5):
            broken = x.copy()
            broken[hole, 2] = np.nan
            assert_matches_oracle(
                np.isnan(crps(broken, y)), [False, False, True, False], rtol=0.0
            )
        y[1] = np.nan
        assert np.isnan(crps(x, y)[1])

    def test_float32_of_warm_temperatures(self) -> None:
        x, y = 300.0 + 0.1 * noise(28, 8, 50), 300.0 + 0.1 * noise(29, 50)
        expected = crps(x, y)
        out = crps(x.astype(np.float32), y.astype(np.float32))
        # a rounding of 3e-5 K in the inputs themselves
        assert_matches_oracle(out, expected, rtol=0.0, atol=5e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("fair", [True, False])
    def test_backends_and_precision(self, fair: bool) -> None:
        def of(x: Any, y: Any) -> Any:
            return 1.0 + crps(x, y, fair=fair)

        x, y = noise(30, 4, 2, 3), noise(31, 2, 3)
        assert_backends_agree(of, x, y)
        assert_traces(of, x[None], y[None])
        assert_refuses_dataarrays(of, x, y)
        assert_grad_matches_fd(lambda x: of(x, y), x)
        assert_grad_matches_fd(lambda y: of(x, y), y)

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="no axis 2"):
            crps(np.ones((3, 4)), np.ones(4), member_axis=2)
        with pytest.raises(ValidationError, match="at least 1 member"):
            crps(np.ones((0, 4)), np.ones(4))


class TestEnergyScore:
    @pytest.mark.parametrize("n_members", SIZES)
    @pytest.mark.parametrize("fair", [True, False])
    def test_one_element_is_the_crps(self, n_members: int, fair: bool) -> None:
        x, y = noise(32, n_members, 6, 1), noise(33, 6, 1)
        out = energy_score(x, y, axis=-1, fair=fair)
        expected = crps(x, y, fair=fair)[:, 0]
        assert_matches_oracle(out, expected, rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("n_members", [4, 5])
    def test_the_sum_over_pairs(self, n_members: int) -> None:
        x, y, w = (
            noise(34, n_members, 3, 4, 2),
            noise(35, 3, 4, 2),
            noise(36, 4, 1) ** 2,
        )

        def norm(d: Any) -> Any:
            return np.sqrt(np.sum(w * d**2, axis=(-2, -1)))

        error = np.mean(norm(x - y), axis=0)
        pairs = np.sum(norm(x[:, None] - x[None]), axis=(0, 1))
        expected = error - pairs / (2.0 * n_members * (n_members - 1))
        for axis in ((1, 2), (-1, -2)):
            out = energy_score(x, y, axis=axis, weights=w)
            assert_matches_oracle(out, expected, rtol=1e-12)

    @pytest.mark.parametrize("n_members", [2, 5])
    def test_a_fair_score_does_not_depend_on_the_members(self, n_members: int) -> None:
        x, y = noise(37, n_members, 20_000, 4), noise(38, 20_000, 4)
        # E‖x − y‖ − ½ E‖x − x′‖ = ½ √2 E χ₄
        exact = 0.5 * np.sqrt(2.0) * np.sqrt(2.0) * math.gamma(2.5) / math.gamma(2.0)
        assert_mean(energy_score(x, y, axis=-1), exact)
        biased = energy_score(x, y, axis=-1, fair=False)
        assert np.mean(biased) > 1.15 * exact

    def test_it_sees_dependence(self) -> None:
        # members of the right elements that vary together the wrong way
        shared = noise(39, 4000, 1)
        y = np.concatenate([shared, shared], axis=-1)
        right = noise(40, 10, 4000, 1) * np.ones(2)
        wrong = right * np.asarray([1.0, -1.0])
        score = [np.mean(energy_score(x, y, axis=-1)) for x in (right, wrong)]
        apart = [np.mean(crps(x, y)) for x in (right, wrong)]
        assert score[0] < 0.9 * score[1]
        assert apart[0] == pytest.approx(apart[1], rel=0.05)

    def test_a_perfect_ensemble_has_a_finite_gradient(self) -> None:
        y = noise(41, 2, 3)
        x = np.broadcast_to(y, (3, 2, 3))
        assert_matches_oracle(energy_score(x, y, axis=-1), np.zeros(2), rtol=0.0)
        assert_finite_gradient(lambda x: energy_score(x, y, axis=-1), x)

    @pytest.mark.slow
    @pytest.mark.parametrize("fair", [True, False])
    def test_backends_and_precision(self, fair: bool) -> None:
        def of(x: Any, y: Any, w: Any) -> Any:
            return energy_score(x, y, axis=-1, weights=w, fair=fair)

        x, y, w = noise(42, 4, 2, 3), noise(43, 2, 3), 0.5 + noise(44, 3) ** 2
        assert_backends_agree(of, x, y, w, rtol32=1e-4)
        assert_traces(of, x[None], y[None], w[None])
        assert_refuses_dataarrays(of, x, y, w)
        assert_grad_matches_fd(lambda x: of(x, y, w), x)

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="no axis 1"):
            energy_score(np.ones((3, 4)), np.ones(4), axis=1)
        with pytest.raises(ValidationError, match="a fair score needs"):
            energy_score(np.ones((1, 4)), np.ones(4), axis=0)


class TestSpreadSkill:
    @pytest.mark.parametrize("n_members", SIZES)
    def test_a_reliable_ensemble(self, n_members: int) -> None:
        x, y = noise(45, n_members, 50, 5000), noise(46, 50, 5000)
        out = spread_skill(x, y, axis=-1)
        assert_mean(out.spread, 1.0)
        assert_mean(out.ratio, 1.0)
        assert_mean(out.skill, np.sqrt((n_members + 1) / n_members))

    def test_numpy(self) -> None:
        x, y, w = noise(47, 6, 3, 8), noise(48, 3, 8), noise(49, 3, 8) ** 2
        valid = np.arange(8) != 4
        out = spread_skill(x, y, weights=w, valid=valid, axis=-1)
        spread = np.average(np.var(x, axis=0, ddof=1)[:, valid], -1, w[:, valid]) ** 0.5
        error = (np.mean(x, axis=0) - y) ** 2
        skill = np.average(error[:, valid], -1, w[:, valid]) ** 0.5
        assert_matches_oracle(out.spread, spread, rtol=1e-12)
        assert_matches_oracle(out.skill, skill, rtol=1e-12)
        ratio = np.sqrt(7.0 / 6.0) * spread / skill
        assert_matches_oracle(out.ratio, ratio, rtol=1e-12)

    def test_gaps(self) -> None:
        x, y = noise(50, 5, 12), noise(51, 12)
        valid = np.arange(12) % 4 != 1
        expected = spread_skill(x[:, valid], y[valid])
        x[:, ~valid] = np.nan
        out = spread_skill(x, np.where(valid, y, np.nan), valid=valid)
        for name in ("spread", "skill", "ratio"):
            got = getattr(out, name)
            assert_matches_oracle(got, getattr(expected, name), rtol=1e-12)

        def ratio(x: Any) -> Any:
            return spread_skill(x, y, valid=valid).ratio

        assert_finite_gradient(ratio, x)
        x[2, 0] = np.nan
        assert np.isnan(spread_skill(x, y, valid=valid).spread)

    def test_a_perfect_ensemble(self) -> None:
        y = noise(52, 5)
        x = np.broadcast_to(y, (4, 5))
        out = spread_skill(x, y)
        assert out.spread == 0.0
        assert out.skill == 0.0
        assert np.isnan(out.ratio)
        assert_finite_gradient(lambda x: spread_skill(x, y).spread, x)
        assert_finite_gradient(lambda x: spread_skill(x, y).skill, x)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any, y: Any, w: Any, m: Any) -> Any:
            out = spread_skill(x, y, weights=w, valid=m, axis=-1)
            return out.spread, out.skill, out.ratio

        x, y, w = noise(53, 4, 2, 6), noise(54, 2, 6), 0.5 + noise(55, 6) ** 2
        m = np.arange(6) != 1
        assert_backends_agree(of, x, y, w, m)
        assert_traces(of, x[None], y[None], w[None], m[None])
        assert_refuses_dataarrays(lambda x, y, w: of(x, y, w, m), x, y, w)
        assert_grad_matches_fd(lambda x: of(x, y, w, m)[2], x)

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="a spread needs at least 2"):
            spread_skill(np.ones((1, 4)), np.ones(4))
        with pytest.raises(ValidationError, match="no axis 1"):
            spread_skill(np.ones((3, 4)), np.ones(4), axis=1)


class TestRankHistogram:
    @pytest.mark.parametrize("n_members", [1, 2, 9])
    def test_a_reliable_ensemble_is_flat(self, n_members: int) -> None:
        stats = pytest.importorskip("scipy.stats")
        x, y = noise(56, n_members, 100, 200), noise(57, 100, 200)
        counts = rank_histogram(x, y)
        assert counts.shape == (n_members + 1,)
        assert np.sum(counts) == 20_000
        assert stats.chisquare(counts).pvalue > 1e-4
        # and one that is biased; one member alone cannot be too narrow
        assert stats.chisquare(rank_histogram(x - 0.2, y)).pvalue < 1e-4

    def test_searchsorted(self) -> None:
        x, y = noise(58, 7, 40), noise(59, 40)
        order = np.sort(x, axis=0)
        ranks = [np.searchsorted(order[:, i], y[i]) for i in range(40)]
        expected = np.bincount(ranks, minlength=8).astype(float)
        assert_matches_oracle(rank_histogram(x, y), expected, rtol=0.0)
        out = rank_histogram(x.T, y, member_axis=1)
        assert_matches_oracle(out, expected, rtol=0.0)

    def test_ties_share(self) -> None:
        x = np.asarray([0.0, 1.0, 1.0, 1.0, 2.0])[:, None]
        out = rank_histogram(x, [1.0])
        assert_matches_oracle(out, [0.0, 0.25, 0.25, 0.25, 0.25, 0.0], rtol=0.0)
        # rain: most members and references are 0
        dry = rank_histogram(np.zeros((3, 8)), np.zeros(8))
        assert_matches_oracle(dry, np.full(4, 2.0), rtol=0.0)

    def test_missing_cases(self) -> None:
        # jejjohnson/geotoolz#776: not as rank 0
        x, y = noise(60, 5, 30), noise(61, 30)
        valid = np.arange(30) % 3 != 0
        expected = rank_histogram(x[:, valid], y[valid])
        x[2, ~valid] = np.nan
        out = rank_histogram(x, np.where(valid, y, np.nan), valid=valid)
        assert_matches_oracle(out, expected, rtol=0.0)
        assert np.isnan(rank_histogram(x, y)).all()
        y[1] = np.nan
        assert np.isnan(rank_histogram(x, y, valid=valid)).all()

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any, y: Any, m: Any) -> Any:
            return rank_histogram(x, y, valid=m)

        x, y = np.round(noise(62, 4, 2, 6), 1), np.round(noise(63, 2, 6), 1)
        m = np.arange(6) != 1
        assert_backends_agree(of, x, y, m)
        assert_traces(of, x[None], y[None], m[None])
        assert_refuses_dataarrays(lambda x, y: of(x, y, m), x, y)
        assert_finite_gradient(lambda x: of(x, y, m), x)


class TestCoverage:
    @pytest.mark.parametrize("n_members", [1, 2, 5, 20])
    @pytest.mark.parametrize("level", [0.3, 0.6, 0.9, 1.0])
    def test_a_reliable_ensemble(self, n_members: int, level: float) -> None:
        # not Gaussian, and the same for each member and the reference
        rng = np.random.default_rng(64)
        x, y = rng.exponential(size=(n_members, 20_000)), rng.exponential(size=20_000)
        per_case = coverage(x[..., None], y[..., None], level=level, axis=-1)
        if level == 1.0:
            assert_matches_oracle(per_case, np.ones(20_000), rtol=1e-12)
        elif n_members == 1:
            # both slots have the same share of the interval
            assert_matches_oracle(per_case, np.full(20_000, level), rtol=1e-12)
        else:
            assert_mean(per_case, level)

    def test_from_the_histogram(self) -> None:
        x, y = np.round(noise(65, 6, 50), 1), np.round(noise(66, 50), 1)
        counts = rank_histogram(x, y) / 50.0
        edges = np.arange(8) / 7.0
        share = np.clip(
            np.minimum(edges[1:], 0.85) - np.maximum(edges[:-1], 0.15), 0, 1
        )
        expected = np.sum(counts * share * 7.0)
        assert float(coverage(x, y, level=0.7)) == pytest.approx(expected, rel=1e-12)

    def test_too_narrow_and_too_wide(self) -> None:
        x, y = noise(67, 20, 5000), noise(68, 5000)
        assert coverage(0.5 * x, y) < 0.7
        assert coverage(3.0 * x, y) > 0.97

    def test_weights_and_gaps(self) -> None:
        x, y, w = noise(69, 5, 3, 20), noise(70, 3, 20), noise(71, 3, 20) ** 2
        valid = np.arange(20) % 4 != 0
        out = coverage(x, y, weights=w, valid=valid, axis=-1)
        for row in range(3):
            one = [coverage(x[:, row, i : i + 1], y[row, i : i + 1]) for i in range(20)]
            expected = np.average(np.asarray(one)[valid], weights=w[row, valid])
            assert out[row] == pytest.approx(expected, rel=1e-12)
        x[:, :, ~valid] = np.nan
        again = coverage(x, y, weights=w, valid=valid, axis=-1)
        assert_matches_oracle(again, out, rtol=1e-12)
        assert np.isnan(coverage(x, y))

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any, y: Any, w: Any) -> Any:
            return coverage(x, y, level=0.8, weights=w, axis=-1)

        x, y, w = noise(72, 4, 2, 6), noise(73, 2, 6), 0.5 + noise(74, 6) ** 2
        assert_backends_agree(of, x, y, w)
        assert_traces(of, x[None], y[None], w[None])
        assert_refuses_dataarrays(of, x, y, w)
        assert_finite_gradient(lambda x: of(x, y, w), x)

    def test_errors(self) -> None:
        for level in (0.0, 1.1, np.nan):
            with pytest.raises(ValidationError, match="'level' must be"):
                coverage(np.ones((3, 4)), np.ones(4), level=level)


class TestReliability:
    EDGES = np.linspace(0.0, 1.0, 6)

    def test_calibrated_probabilities(self) -> None:
        rng = np.random.default_rng(75)
        p = rng.random(50_000)
        event = rng.random(50_000) < p
        out = reliability(p, event, edges=self.EDGES)
        assert_matches_oracle(out.weight, np.histogram(p, self.EDGES)[0], rtol=0.0)
        for b in range(5):
            inside = (p >= self.EDGES[b]) & (p < self.EDGES[b + 1])
            assert out.mean_probability[b] == pytest.approx(p[inside].mean(), rel=1e-12)
            assert_mean(event[inside] - p[inside], 0.0)
            expected = event[inside].mean()
            assert out.observed_frequency[b] == pytest.approx(expected, rel=1e-12)

    def test_the_edges(self) -> None:
        # a bin holds its lower edge, the last both; a case outside is in none
        p = [0.0, 0.5, 1.0, 1.5, -0.5]
        out = reliability(p, [1, 0, 1, 1, 1], edges=[0.0, 0.5, 1.0])
        assert_matches_oracle(out.weight, [1.0, 2.0], rtol=0.0)
        assert_matches_oracle(out.observed_frequency, [1.0, 0.5], rtol=0.0)

    def test_an_empty_bin(self) -> None:
        out = reliability([0.1], [1.0], edges=[0.0, 0.5, 1.0])
        assert_matches_oracle(out.weight, [1.0, 0.0], rtol=0.0)
        assert np.isnan(out.mean_probability[1])
        assert np.isnan(out.observed_frequency[1])

    def test_weights_and_gaps(self) -> None:
        rng = np.random.default_rng(76)
        p, w = rng.random((2, 3, 20))
        event = rng.random((3, 20)) < 0.5
        valid = rng.random((3, 20)) > 0.3
        out = reliability(p, event, edges=self.EDGES, weights=w, valid=valid)
        bare = reliability(p[valid], event[valid], edges=self.EDGES, weights=w[valid])
        spoiled = np.where(valid, p, np.nan)
        again = reliability(spoiled, event, edges=self.EDGES, weights=w, valid=valid)
        for name in ("mean_probability", "observed_frequency", "weight"):
            expected = getattr(bare, name)
            assert_matches_oracle(getattr(out, name), expected, rtol=1e-12)
            assert_matches_oracle(getattr(again, name), expected, rtol=1e-12)

    def test_a_nan_in_a_valid_case(self) -> None:
        out = reliability([0.1, np.nan], [1.0, 0.0], edges=[0.0, 0.5, 1.0])
        assert np.isnan(out.weight).all()
        assert np.isnan(out.observed_frequency).all()
        out = reliability([0.1, 0.7], [np.nan, 0.0], edges=[0.0, 0.5, 1.0])
        assert_matches_oracle(out.observed_frequency, [np.nan, 0.0], rtol=0.0)
        assert_matches_oracle(out.weight, [1.0, 1.0], rtol=0.0)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(p: Any, event: Any, w: Any) -> Any:
            out = reliability(p, event, edges=[0.0, 0.4, 1.0], weights=w)
            return out.mean_probability, out.observed_frequency, out.weight

        rng = np.random.default_rng(77)
        p, w = 0.05 + 0.9 * rng.random((2, 2, 8))
        event = rng.random((2, 8)) < 0.5
        assert_backends_agree(of, p, event, w)
        assert_traces(of, p[None], event[None], w[None])
        assert_refuses_dataarrays(of, p, event, w)
        assert_grad_matches_fd(lambda p: of(p, event, w)[0], p)

    def test_errors(self) -> None:
        for edges in ([0.5], [0.5, 0.5], [1.0, 0.0], [[0.0, 1.0]], [0.0, np.nan]):
            with pytest.raises(ValidationError, match="'edges' must be"):
                reliability([0.5], [1.0], edges=edges)


class TestPsdScore:
    def test_stored_xrops(self) -> None:
        stored = xrops_values("skill")
        out = psd_score(
            stored["spectral_prediction"],
            stored["spectral_truth"],
            float(stored["spectral_spacing"]),
            isotropic=False,
            window=None,
            detrend="constant",
        )
        # two-sided, from −Nyquist: its positive half is ours without the ends
        half = stored["spectral_frequency"] > 0.0
        assert_matches_oracle(
            out.wavenumber[1:-1], stored["spectral_frequency"][half], rtol=1e-12
        )
        assert_matches_oracle(
            out.score[1:-1], stored["spectral_psd_score"][half], rtol=1e-10
        )

    @pytest.mark.parametrize("isotropic", [True, False])
    def test_the_spectra(self, isotropic: bool) -> None:
        p, r = noise(80, 2, 3, 4, 12, 10)
        spacing = (2.0, 0.5) if isotropic else 0.5
        spectrum: Any = isotropic_spectrum if isotropic else power_spectrum
        out = psd_score(p, r, spacing, isotropic=isotropic)
        error, truth = spectrum(p - r, spacing), spectrum(r, spacing)
        assert_matches_oracle(out.wavenumber, truth.wavenumber, rtol=0.0)
        # a ring that holds no mode, between those of a long grid, is NaN
        with np.errstate(invalid="ignore"):
            expected = 1.0 - error.power / truth.power
        assert_matches_oracle(out.score, expected, rtol=1e-12)
        # the mean of the spectra, then the ratio
        out = psd_score(p, r, spacing, isotropic=isotropic, average=(0, 1))
        error, truth = (np.mean(s.power, axis=(0, 1)) for s in (error, truth))
        with np.errstate(invalid="ignore"):
            expected = 1.0 - error / truth
        assert_matches_oracle(out.score, expected, rtol=1e-12)
        negative = psd_score(p, r, spacing, isotropic=isotropic, average=-3)
        assert negative.score.shape[0] == 3

    def test_perfect_and_hopeless(self) -> None:
        r = noise(81, 12, 12)
        assert_matches_oracle(psd_score(r, r, 1.0).score, np.ones(9), rtol=0.0)
        out = psd_score(0.0, r, 1.0)
        assert_matches_oracle(out.score, np.zeros(9), rtol=0.0, atol=1e-12)
        out = psd_score(-r, r, 1.0)
        assert_matches_oracle(out.score, np.full(9, -3.0), rtol=1e-12)

    def test_no_power_is_nan(self) -> None:
        out = psd_score(noise(82, 8, 8), 0.0, 1.0)
        assert np.isnan(out.score).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("isotropic", [True, False])
    def test_backends_and_precision(self, isotropic: bool) -> None:
        def of(p: Any, r: Any) -> Any:
            out = psd_score(p, r, 0.5, isotropic=isotropic, average=0)
            # k = 0 is a ratio of two small numbers
            return out.wavenumber, 1.0 + out.score[..., 1:]

        p, r = noise(83, 2, 3, 6, 7)
        assert_backends_agree(of, p, r, rtol=1e-10, rtol32=2e-3)
        assert_traces(of, p[None], r[None], rtol=1e-10)
        assert_refuses_dataarrays(of, p, r)
        assert_grad_matches_fd(lambda p: of(p, r[:2])[1], p[:2], rtol=1e-5)

    @needs_jax
    def test_gradient_at_a_perfect_prediction(self) -> None:
        r = noise(84, 8, 8)
        assert_finite_gradient(lambda p: psd_score(p, r, 1.0).score, r)

    def test_errors(self) -> None:
        r = noise(85, 3, 8, 8)
        for average in (1, -1, (0, 2)):
            with pytest.raises(ValidationError, match="'average' must not name"):
                psd_score(r, r, 1.0, average=average)
        with pytest.raises(ValidationError, match="no axis 3"):
            psd_score(r, r, 1.0, average=3)
        with pytest.raises(ValidationError, match="at least 2 samples"):
            psd_score(r[0, 0], r[0, 0], 1.0)
        with pytest.raises(ValidationError, match="spacing"):
            psd_score(r, r, (1.0, 1.0), isotropic=False)


class TestResolvedScale:
    def test_a_low_pass_filter(self) -> None:
        # every mode below the cut-off is kept, and none above
        n, step = 128, 5.0
        r = noise(86, 40, n)
        z = np.fft.rfft(r, axis=-1)
        k = np.fft.rfftfreq(n, step)
        cut = 0.5 * (k[20] + k[21])
        p = np.fft.irfft(np.where(k < cut, z, 0.0), n, axis=-1)
        score = psd_score(
            p, r, step, isotropic=False, average=0, window=None, detrend=None
        )
        out = float(resolved_scale(score))
        # between the last bin that is kept and the first that is not
        assert 1.0 / k[21] < out < 1.0 / k[20]
        assert out == pytest.approx(1.0 / np.sqrt(k[20] * k[21]), rel=1e-9)

    def test_an_error_that_grows_with_wavenumber(self) -> None:
        # S = 1 − (k/k_N)², 0.5 at k_N/√2
        stored = xrops_values("skill")
        score = psd_score(
            stored["spectral_prediction"],
            stored["spectral_truth"],
            float(stored["spectral_spacing"]),
            isotropic=False,
            window=None,
            detrend=None,
        )
        out = float(resolved_scale(score))
        exact = np.sqrt(2.0) * 2.0 * float(stored["spectral_spacing"])
        assert out == pytest.approx(exact, rel=1e-3)
        # xrops interpolates in the wavelength
        stored_scale = float(stored["spectral_resolved_scale"])
        assert out == pytest.approx(stored_scale, rel=3e-3)

    def test_the_first_crossing(self) -> None:
        k = 0.1 * np.arange(8)
        s = np.asarray([0.0, 1.0, 0.9, 0.4, 0.8, 0.3, 0.2, 0.1])
        out = float(resolved_scale(PSDScore(k, s)))
        weight = (0.9 - 0.5) / (0.9 - 0.4)
        log_k = (1.0 - weight) * np.log(0.2) + weight * np.log(0.3)
        assert out == pytest.approx(np.exp(-log_k), rel=1e-12)
        # another level, and one that is never reached
        assert 1.0 / 0.7 < float(resolved_scale(PSDScore(k, s), level=0.15)) < 1.0 / 0.6
        assert np.isnan(resolved_scale(PSDScore(k, s), level=0.05))
        # below it from the start
        assert float(resolved_scale(PSDScore(k, s), level=1.5)) == pytest.approx(10.0)

    def test_a_nan_before_the_crossing(self) -> None:
        k = 0.1 * np.arange(6)
        s = np.asarray(
            [[1.0, 1.0, np.nan, 0.2, 0.1, 0.0], [1.0, 1, 0.8, 0.2, 0, np.nan]]
        )
        out = resolved_scale(PSDScore(k, s))
        assert np.isnan(out[0])
        assert np.isfinite(out[1])

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(k: Any, s: Any) -> Any:
            return resolved_scale(PSDScore(k, s))

        k = 0.1 * np.arange(6)
        s = np.asarray([[1.0, 1.0, 0.8, 0.2, 0.1, 0.0], [1.0, 0.9, 0.7, 0.6, 0.4, 0.2]])
        assert_backends_agree(of, k, s, rtol32=1e-4, kinds=EAGER_KINDS)
        assert_traces(of, k[None], s[None])
        assert_refuses_dataarrays(of, k, s)
        assert_grad_matches_fd(lambda s: of(k, s), s)

    @needs_jax
    def test_inside_a_loss(self) -> None:
        import jax

        r = noise(87, 4, 16, 16)
        p = r + 0.8 * noise(88, 4, 16, 16)

        def of(p: Any) -> Any:
            return resolved_scale(psd_score(p, r, 1.0, average=0), level=0.4)

        assert_matches_oracle(jax.jit(of)(p), of(p), rtol=1e-9)
        assert np.isfinite(jax.grad(of)(jax.numpy.asarray(p))).all()


class TestBandRmse:
    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, np.inf))

    @pytest.mark.parametrize("shape", [(12, 12), (9, 14), (8, 7)])
    def test_bands_that_cover_the_plane(self, shape: Any) -> None:
        p, r = noise(89, 2, 3, *shape)
        out = band_rmse(p, r, (1.0, 1.3), self.BANDS)
        total = np.mean((p - r) ** 2, axis=(-2, -1))
        assert out.shape == (3, 3)
        assert_matches_oracle(np.sum(out**2, axis=-1), total, rtol=1e-12)
        # one band is the whole
        whole = band_rmse(p, r, 1.0, [(0.0, np.inf)])
        assert_matches_oracle(whole[..., 0], rmse(p, r, axis=(-2, -1)), rtol=1e-12)

    def test_a_wave_is_in_its_band(self) -> None:
        y, x = np.mgrid[:20, :20]
        # |k| = 5/20 exactly, a lower edge, which is inside
        wave = 3.0 * np.cos(2.0 * np.pi * (3 * y + 4 * x) / 20.0)
        out = band_rmse(wave, 0.0, 1.0, self.BANDS)
        expected = [0.0, 0.0, 3.0 / np.sqrt(2.0)]
        assert_matches_oracle(out, expected, rtol=0.0, atol=1e-13)

    def test_weights(self) -> None:
        p, r, w = noise(90, 3, 10, 11)
        w = w**2
        k = np.hypot(np.fft.fftfreq(10)[:, None], np.fft.fftfreq(11))
        z = np.fft.fft2(p - r)
        expected = []
        for low, high in self.BANDS:
            inside = (k >= low) & (k < high)
            filtered = np.fft.ifft2(np.where(inside, z, 0.0)).real
            expected.append(np.sqrt(np.average(filtered**2, weights=w)))
        out = band_rmse(p, r, 1.0, self.BANDS, weights=w)
        assert_matches_oracle(out, expected, rtol=1e-12)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(p: Any, r: Any, w: Any) -> Any:
            return band_rmse(p, r, 1.0, self.BANDS, weights=w)

        p, r, w = noise(91, 2, 6, 7), noise(92, 6, 7), 0.5 + noise(93, 6, 7) ** 2
        assert_backends_agree(of, p, r, w, rtol32=1e-4)
        assert_traces(of, p[None], r[None], w[None])
        assert_refuses_dataarrays(of, p, r, w)
        assert_grad_matches_fd(lambda p: of(p, r, w), p[0])
        assert_finite_gradient(lambda p: of(p, r, w), r)

    def test_errors(self) -> None:
        x = noise(94, 8, 8)
        for bands in ([], [0.1, 0.2], [(0.1, 0.2, 0.3)]):
            with pytest.raises(ValidationError, match="pairs of wavenumbers"):
                band_rmse(x, x, 1.0, bands)
        for band in ((0.2, 0.1), (0.1, 0.1), (-0.1, 0.1), (np.nan, 0.1)):
            with pytest.raises(ValidationError, match="a band must be"):
                band_rmse(x, x, 1.0, [band])
        with pytest.raises(ValidationError, match="no wavenumber of this grid"):
            band_rmse(x, x, 1.0, [(0.01, 0.02)])
        with pytest.raises(ValidationError, match="at least 2 samples"):
            band_rmse(x[0], x[0], 1.0, self.BANDS)


PLANE = gridax.CartesianAGrid2D.from_coords(
    y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
)
SPHERE = gridax.SphericalAGrid2D.from_coords(
    lat=[-70.0, -35.0, -10.0, 15.0, 40.0, 65.0],
    lon=[0.0, 8.0, 20.0, 30.0, 45.0, 70.0, 80.0],
    R=2.0,
)
GRIDS = {"plane": PLANE, "sphere": SPHERE}
COLUMN = np.linspace(0.5, 1.5, 6)[:, None]


def coast(grid: Any) -> tuple[np.ndarray, Any]:
    """A wet region with two dry cells, and its gridax mask."""
    wet = np.ones(grid.shape, dtype=bool)
    wet[2:4, 2] = False
    return wet, gridax.Mask2D.from_grid(grid, wet)


def balanced(grid: Any, eta: Any) -> tuple[np.ndarray, np.ndarray]:
    """The velocity in balance with a height, for f = `COLUMN` and g = 2."""
    slope_x, slope_y = (np.asarray(part) for part in gridax.gradient(eta, grid))
    return -2.0 * slope_y / COLUMN, 2.0 * slope_x / COLUMN


def of_imbalance(grid: Any, **options: Any) -> Callable[..., Any]:
    def of(u: Any, v: Any, eta: Any) -> Any:
        return geostrophic_imbalance(u, v, eta, grid, f=COLUMN, gravity=2.0, **options)

    return of


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS)
class TestGeostrophicImbalance:
    def test_a_geostrophic_flow(self, grid: Any) -> None:
        eta = noise(100, *grid.shape)
        u, v = balanced(grid, eta)
        assert float(of_imbalance(grid)(u, v, eta)) == pytest.approx(0.0, abs=1e-13)
        # a flat surface leaves all of Coriolis over
        assert float(of_imbalance(grid)(u, v, 0.0 * eta)) == pytest.approx(1.0)
        # and an opposite flow twice the pressure gradient
        assert float(of_imbalance(grid)(-u, -v, eta)) == pytest.approx(2.0)

    def test_the_scale_cancels(self, grid: Any) -> None:
        u, v, eta = noise(101, 3, *grid.shape)
        out = of_imbalance(grid)(u, v, eta)
        assert of_imbalance(grid)(1e-3 * u, 1e-3 * v, 1e-3 * eta) == pytest.approx(
            float(out), rel=1e-12
        )

    def test_the_formula(self, grid: Any) -> None:
        # with weights of 1, as xrops and somax have it
        u, v, eta = noise(102, 3, 2, *grid.shape)
        r_u, r_v = geostrophic_residual(u, v, eta, grid, f=COLUMN, gravity=2.0)
        area = np.broadcast_to(np.asarray(gridax.area_weights(grid)), grid.shape)
        for options, w in (({"weights": 1.0}, np.ones(grid.shape)), ({}, area)):
            out = of_imbalance(grid, **options)(u, v, eta)

            def total(x: Any, w: Any = w) -> Any:
                return np.sum(w * np.asarray(x) ** 2, axis=(-2, -1))

            left, coriolis = (
                total(r_u) + total(r_v),
                total(COLUMN * u) + total(COLUMN * v),
            )
            assert_matches_oracle(out, np.sqrt(left / coriolis), rtol=1e-12)

    def test_a_coast(self, grid: Any) -> None:
        wet, mask = coast(grid)
        u, v, eta = noise(103, 3, *grid.shape)
        out = of_imbalance(grid, mask=mask)(u, v, eta)
        for hole in (np.nan, 1e6):
            dry = [np.where(wet, a, hole) for a in (u, v, eta)]
            assert_matches_oracle(of_imbalance(grid, mask=mask)(*dry), out, rtol=1e-12)
        dry = [np.where(wet, a, np.nan) for a in (u, v, eta)]
        for which in range(3):

            def of(x: Any, which: int = which) -> Any:
                args = [x if i == which else a for i, a in enumerate(dry)]
                return of_imbalance(grid, mask=mask)(*args)

            assert_finite_gradient(of, dry[which])
        u[0, 0] = np.nan
        assert np.isnan(of_imbalance(grid, mask=mask)(u, v, eta))

    def test_rest(self, grid: Any) -> None:
        zero = np.zeros(grid.shape)
        assert np.isnan(of_imbalance(grid)(zero, zero, noise(104, *grid.shape)))

    @pytest.mark.slow
    def test_backends_and_precision(self, grid: Any) -> None:
        def of(u: Any, v: Any, eta: Any, f: Any = COLUMN) -> Any:
            return geostrophic_imbalance(u, v, eta, grid, f=f, gravity=2.0)

        u, v, eta = noise(105, 3, 2, *grid.shape)
        assert_backends_agree(of, u, v, eta, COLUMN, kinds=GRID_KINDS)
        assert_traces(of, u[None], v[None], eta[None], COLUMN[None])
        assert_refuses_dataarrays(of, u, v, eta, COLUMN)
        assert_grad_matches_fd(lambda eta: of(u[0], v[0], eta), eta[0])
        # at a geostrophic flow, whose residual of exactly 0 is that of no flow
        flat = np.ones(grid.shape)
        zero = np.zeros(grid.shape)
        assert_finite_gradient(lambda u: of(u, zero, flat), zero)


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS)
class TestRmsDivergence:
    def test_closed_forms(self, grid: Any) -> None:
        if grid is PLANE:
            x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
            out = rms_divergence(3.0 * x + y, x - 1.0 * y, grid)
            assert float(out) == pytest.approx(2.0, rel=1e-12)
            rotation = rms_divergence(-y, x, grid)
        else:
            latitude = np.deg2rad([-70.0, -35.0, -10.0, 15.0, 40.0, 65.0])[:, None]
            u = np.broadcast_to(np.cos(latitude), grid.shape)
            rotation = rms_divergence(u, 0.0 * u, grid)
        assert float(rotation) == pytest.approx(0.0, abs=1e-13)

    def test_gridax(self, grid: Any) -> None:
        u, v = noise(110, 2, 2, *grid.shape)
        divergence = np.asarray(gridax.divergence(u, v, grid))
        area = np.broadcast_to(np.asarray(gridax.area_weights(grid)), grid.shape)
        expected = np.sqrt(np.sum(area * divergence**2, axis=(-2, -1)) / np.sum(area))
        assert_matches_oracle(rms_divergence(u, v, grid), expected, rtol=1e-12)
        plain = rms_divergence(u, v, grid, weights=1.0)
        expected = np.sqrt(np.mean(divergence**2, axis=(-2, -1)))
        assert_matches_oracle(plain, expected, rtol=1e-12)

    def test_a_coast(self, grid: Any) -> None:
        wet, mask = coast(grid)
        u, v = noise(111, 2, *grid.shape)
        out = rms_divergence(u, v, grid, mask=mask)
        dry_u, dry_v = (np.where(wet, a, np.nan) for a in (u, v))
        again = rms_divergence(dry_u, dry_v, grid, mask=mask)
        assert_matches_oracle(again, out, rtol=1e-12)
        assert_finite_gradient(
            lambda u: rms_divergence(u, dry_v, grid, mask=mask), dry_u
        )

    @pytest.mark.slow
    def test_backends_and_precision(self, grid: Any) -> None:
        def of(u: Any, v: Any) -> Any:
            return rms_divergence(u, v, grid)

        u, v = noise(112, 2, 2, *grid.shape)
        assert_backends_agree(of, u, v, kinds=GRID_KINDS)
        assert_traces(of, u[None], v[None])
        assert_refuses_dataarrays(of, u, v)
        assert_grad_matches_fd(lambda u: of(u, v[0]), u[0])
        zero = np.zeros(grid.shape)
        assert_finite_gradient(lambda u: of(u, zero), zero)


class TestGridErrors:
    def test_grid_errors(self) -> None:
        wrong = np.ones((3, 3))
        with pytest.raises(ValidationError, match="ends in"):
            geostrophic_imbalance(wrong, wrong, wrong, PLANE, f=1.0)
        with pytest.raises(ValidationError, match="ends in"):
            rms_divergence(wrong, wrong, PLANE)
        with pytest.raises(ValidationError, match="'grid' must be"):
            rms_divergence(wrong, wrong, None)


class TestDensityInversionFraction:
    Z = -np.asarray([0.0, 5.0, 15.0, 30.0, 50.0])

    def test_known_columns(self) -> None:
        rho = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 1.5, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        expected = [0.0, 0.25, 1.0, 0.0]
        out = density_inversion_fraction(rho, self.Z)
        assert_matches_oracle(out, expected, rtol=0.0)
        # the other order, another axis, and one column of heights per point
        out = density_inversion_fraction(rho[:, ::-1], self.Z[::-1])
        assert_matches_oracle(out, expected, rtol=0.0)
        out = density_inversion_fraction(rho.T, self.Z, axis=0)
        assert_matches_oracle(out, expected, rtol=0.0)
        heights = np.broadcast_to(self.Z, rho.shape)
        out = density_inversion_fraction(rho.T, heights.T, axis=0)
        assert_matches_oracle(out, expected, rtol=0.0)
        # a depth as it is turns them round
        out = density_inversion_fraction(rho, -self.Z)
        assert_matches_oracle(out, [1.0, 0.75, 0.0, 0.0], rtol=0.0)

    def test_gaps(self) -> None:
        rho = np.asarray([[1.0, 2.0, 1.5, 4.0, 5.0], [3.0, 2.0, 9.0, 9.0, 9.0]])
        valid = np.asarray([[1, 1, 1, 0, 1], [1, 1, 0, 1, 0]], dtype=bool)
        for hole in (np.nan, 1e9):
            broken = np.where(valid, rho, hole)
            out = density_inversion_fraction(broken, self.Z, valid=valid)
            assert_matches_oracle(out, [0.5, 1.0], rtol=0.0)
            out = density_inversion_fraction(broken.T, self.Z, axis=0, valid=valid.T)
            assert_matches_oracle(out, [0.5, 1.0], rtol=0.0)
        none = density_inversion_fraction(rho, self.Z, valid=np.zeros(5, dtype=bool))
        assert np.isnan(none).all()
        rho[0, 1] = np.nan
        out = density_inversion_fraction(rho, self.Z, valid=valid)
        assert_matches_oracle(out, [np.nan, 1.0], rtol=0.0)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(rho: Any, z: Any, m: Any) -> Any:
            return density_inversion_fraction(rho, z, valid=m)

        rho = noise(120, 3, 5)
        m = np.random.default_rng(121).random((3, 5)) > 0.2
        assert_backends_agree(of, rho, self.Z, m)
        assert_traces(of, rho[None], self.Z[None], m[None])
        assert_refuses_dataarrays(lambda rho, z: of(rho, z, m), rho, self.Z)
        assert_finite_gradient(lambda rho: of(rho, self.Z, m), rho)

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="differ in levels"):
            density_inversion_fraction(np.ones(4), self.Z)
        with pytest.raises(ValueError, match="'valid' must be boolean"):
            density_inversion_fraction(np.ones(5), self.Z, valid=np.ones(5))


C_GRID = gridax.CartesianCGrid2D.from_interior(6, 5, 3.0, 5.0)


class TestQgBalanceResidual:
    def test_its_own_vorticity(self) -> None:
        psi = noise(130, *C_GRID.shape)
        options = {"f0": 0.5, "beta": 0.3, "y0": 1.0}
        q = np.asarray(gridax.qg_potential_vorticity(psi, C_GRID, **options)).copy()
        assert float(qg_balance_residual(q, psi, C_GRID, **options)) == 0.0
        # the ghost ring is the caller's, and does not count
        for edge in (0, -1):
            q[edge], q[:, edge] = np.nan, np.nan
        assert float(qg_balance_residual(q, psi, C_GRID, **options)) == 0.0
        # without β the rest is β (y − y₀) / f₀
        out = qg_balance_residual(q, psi, C_GRID, f0=0.5)
        y = np.asarray(C_GRID.coords("y"))[1:-1]
        rest = np.broadcast_to((0.3 * (y - 1.0) / 0.5)[:, None], (5, 6))
        expected = np.sqrt(np.mean(rest**2) / np.mean(q[1:-1, 1:-1] ** 2))
        assert float(out) == pytest.approx(expected, rel=1e-12)

    def test_a_known_error(self) -> None:
        psi = noise(131, 2, *C_GRID.shape)
        q = np.asarray(gridax.qg_potential_vorticity(psi, C_GRID, f0=0.5))
        error = noise(132, 2, *C_GRID.shape)
        out = qg_balance_residual(q + error, psi, C_GRID, f0=0.5)
        inner = (..., slice(1, -1), slice(1, -1))
        mean = [np.mean(a[inner] ** 2, axis=(-2, -1)) for a in (error, q + error)]
        assert_matches_oracle(out, np.sqrt(mean[0] / mean[1]), rtol=1e-12)

    def test_layers(self) -> None:
        psi = noise(133, 2, *C_GRID.shape)
        A = np.asarray([[2.0, -2.0], [-1.0, 1.0]])
        q = np.asarray(gridax.qg_potential_vorticity(psi, C_GRID, f0=0.5, A=A))
        out = qg_balance_residual(q, psi, C_GRID, f0=0.5, A=A)
        assert_matches_oracle(out, [0.0, 0.0], rtol=0.0, atol=1e-14)
        assert (qg_balance_residual(q, psi, C_GRID, f0=0.5) > 0.01).all()

    def test_a_coast(self) -> None:
        wet = np.ones(C_GRID.shape, dtype=bool)
        wet[2:4, 3] = False
        mask = gridax.Mask2D.from_grid(C_GRID, wet)
        psi = noise(134, *C_GRID.shape)
        q = np.asarray(gridax.qg_potential_vorticity(psi, C_GRID, f0=0.5, mask=mask))
        dry = np.where(wet, q, np.nan)
        assert float(qg_balance_residual(dry, psi, C_GRID, f0=0.5, mask=mask)) == 0.0

        def of(q: Any) -> Any:
            return qg_balance_residual(q, psi, C_GRID, f0=0.5, mask=mask)

        assert_finite_gradient(of, dry)

    def test_nothing(self) -> None:
        zero = np.zeros(C_GRID.shape)
        assert np.isnan(qg_balance_residual(zero, zero, C_GRID, f0=0.5))

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(q: Any, psi: Any) -> Any:
            return qg_balance_residual(q, psi, C_GRID, f0=0.5, beta=0.3)

        q, psi = noise(135, 2, 2, *C_GRID.shape)
        assert_backends_agree(of, q, psi, kinds=GRID_KINDS)
        assert_traces(of, q[None], psi[None])
        assert_refuses_dataarrays(of, q, psi)
        assert_grad_matches_fd(lambda psi: of(q[0], psi), psi[0])
        assert_finite_gradient(lambda psi: of(q[0], psi), psi[0])

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="must be a CartesianCGrid2D"):
            qg_balance_residual(np.ones((6, 7)), np.ones((6, 7)), PLANE, f0=1.0)
        with pytest.raises(ValidationError, match="ends in"):
            qg_balance_residual(np.ones((6, 7)), np.ones((6, 7)), C_GRID, f0=1.0)


class TestDieboldMariano:
    @pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
    def test_one_step_is_the_t_statistic(self, power: float) -> None:
        stats = pytest.importorskip("scipy.stats")
        a, b = noise(140, 2, 3, 40)
        d = np.abs(a) ** power - np.abs(b) ** power
        out = diebold_mariano(a, b, power=power)
        expected = stats.ttest_1samp(d, 0.0, axis=-1)
        assert_matches_oracle(out.statistic, expected.statistic, rtol=1e-12)
        assert out.degrees_of_freedom == 39
        p_value = 2.0 * stats.t.sf(np.abs(out.statistic), out.degrees_of_freedom)
        assert_matches_oracle(p_value, expected.pvalue, rtol=1e-11)

    @pytest.mark.parametrize("horizon", [1, 2, 4])
    @pytest.mark.parametrize("correction", [True, False])
    def test_the_formula(self, horizon: int, correction: bool) -> None:
        n = 50
        a = 1.5 * noise(141, n)
        b = noise(142, n)
        d = a**2 - b**2
        c = d - d.mean()
        variance = np.mean(c**2)
        for lag in range(1, horizon):
            variance += 2.0 * np.sum(c[lag:] * c[:-lag]) / n
        expected = d.mean() / np.sqrt(variance / n)
        if correction:
            expected *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        out = diebold_mariano(a, b, horizon=horizon, correction=correction)
        assert float(out.statistic) == pytest.approx(expected, rel=1e-12)
        swapped = diebold_mariano(b, a, horizon=horizon, correction=correction)
        assert float(swapped.statistic) == pytest.approx(-expected, rel=1e-12)
        assert out.statistic > 0.0

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_the_size_of_the_test(self, horizon: int) -> None:
        stats = pytest.importorskip("scipy.stats")
        # differences that are a moving average of the order of the horizon: the
        # loss is linear, so the errors are made of them
        steps = noise(143, 8000, 402)
        d = sum(steps[:, k : 400 + k] for k in range(horizon))
        out = diebold_mariano(100.0 + d, 100.0, power=1.0, horizon=horizon)
        critical = stats.t.ppf(0.975, out.degrees_of_freedom)
        rejected = np.abs(out.statistic) > critical
        # an estimated variance is not chi-square for h > 1: half a point of it
        share = np.mean(rejected)
        if horizon == 1:
            assert_mean(rejected, 0.05)
        else:
            assert share == pytest.approx(0.05, abs=0.015)

    def test_axis_and_broadcasting(self) -> None:
        a, b = noise(144, 30, 3), noise(145, 30, 1)
        out = diebold_mariano(a, b, axis=0, horizon=2)
        one = diebold_mariano(a[:, 1], b[:, 0], horizon=2)
        assert out.statistic.shape == (3,)
        assert float(out.statistic[1]) == pytest.approx(float(one.statistic), rel=1e-12)

    def test_undefined(self) -> None:
        a = noise(146, 12)
        assert np.isnan(diebold_mariano(a, a).statistic)
        # 0.1 is not a float, so the mean of twelve is not 0.1
        assert np.isnan(diebold_mariano(np.full(12, 0.1), 0.0, power=1.0).statistic)
        # differences that alternate: the variance of their mean is negative
        d = (-1.0) ** np.arange(12)
        out = diebold_mariano(10.0 + d, 10.0, power=1.0, horizon=2)
        assert np.isnan(out.statistic)
        a[3] = np.nan
        assert np.isnan(diebold_mariano(a, 0.0).statistic)

    @pytest.mark.slow
    @pytest.mark.parametrize("horizon", [1, 3])
    def test_backends_and_precision(self, horizon: int) -> None:
        def of(a: Any, b: Any) -> Any:
            return diebold_mariano(a, b, horizon=horizon).statistic

        a, b = noise(147, 2, 2, 30)
        assert_backends_agree(of, a, b, rtol32=1e-4)
        assert_traces(of, a[None], b[None])
        assert_refuses_dataarrays(of, a, b)
        assert_grad_matches_fd(lambda a: of(a, b), a)

    @needs_jax
    def test_gradient_beside_identical_errors(self) -> None:
        import jax

        a = noise(148, 2, 10)
        b = np.stack([a[0], a[1] + noise(149, 10)])
        assert_finite_gradient(lambda a: diebold_mariano(a, b).statistic, a)
        # the degrees of freedom are static
        out = diebold_mariano(a, b)
        assert len(jax.tree.leaves(out)) == 1

    def test_errors(self) -> None:
        a = noise(150, 8)
        for horizon in (0, 8):
            with pytest.raises(ValidationError, match="'horizon' must be"):
                diebold_mariano(a, a, horizon=horizon)
        for power in (0.0, -1.0, np.nan):
            with pytest.raises(ValidationError, match="'power' must be"):
                diebold_mariano(a, a, power=power)
        with pytest.raises(ValidationError, match="no axis 1"):
            diebold_mariano(a, a, axis=1)
