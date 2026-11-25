"""Scores of a prediction against its reference."""

from __future__ import annotations

import math
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
from _oracles import reliable_ensemble, xrops_values, xskillscore_crps
from _sampling import assert_mean
from hypothesis import given, strategies as st

from diagnostix import ValidationError
from diagnostix.skill import (
    bias,
    correlation,
    coverage,
    crps,
    energy_score,
    mae,
    nrmse_score,
    r2_score,
    rank_histogram,
    reliability,
    rmse,
    spread_skill,
)


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
