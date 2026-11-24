"""Sums, means and counts over static groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _backends import (
    KINDS,
    assert_backends_agree,
    assert_refuses_dataarrays,
)
from _harness import (
    assert_adjoint,
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import xrops_values
from hypothesis import given, strategies as st

from diagnostix import DiagnostixError
from diagnostix._src.groups import Groups


N, G = 200, 12
RNG = np.random.default_rng(0)
# labels 3 and 11 stay empty, and about a tenth of the samples are in no group
LABELS = RNG.choice([-1, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10], size=N)
GROUPS = Groups.from_labels(LABELS, G)
X = RNG.normal(size=(3, N))
W = RNG.uniform(0.1, 2.0, size=N)
M = RNG.random((3, N)) > 0.2
# no cancellation, so that float32 keeps its relative precision
POSITIVE = np.abs(X)

# the tolerances of #17, relative to the sum of magnitudes, because a sum of
# signed values can cancel to anything
PRECISIONS = [(np.float64, 1e-14), (np.float32, 1e-6)]


def bincount(values: np.ndarray, use: np.ndarray) -> np.ndarray:
    """The grouped sum of each row by ``np.bincount``, in float64."""
    use = np.broadcast_to(use, values.shape) & (LABELS >= 0)
    return np.stack(
        [
            np.bincount(LABELS[u], weights=row[u], minlength=G)
            for row, u in zip(np.asarray(values, np.float64), use, strict=True)
        ]
    )


def mean_of(x: Any, w: Any, m: Any) -> Any:
    return GROUPS.mean(x, weights=w, valid=m)


def sum_of(x: Any, m: Any) -> Any:
    return GROUPS.sum(x, valid=m)


class TestAgainstBincount:
    @pytest.mark.parametrize(("dtype", "atol"), PRECISIONS)
    def test_sum(self, dtype: Any, atol: float) -> None:
        x = X.astype(dtype)
        scale = np.maximum(bincount(np.abs(x), M), 1e-300)
        got = GROUPS.sum(x, valid=M)
        assert got.dtype == dtype
        assert_matches_oracle(got / scale, bincount(x, M) / scale, rtol=0.0, atol=atol)

    @pytest.mark.parametrize(("dtype", "atol"), PRECISIONS)
    def test_mean(self, dtype: Any, atol: float) -> None:
        x, w = X.astype(dtype), W.astype(dtype)
        weight = bincount(np.broadcast_to(w, X.shape), M)
        with np.errstate(invalid="ignore"):
            expected = bincount(np.float64(w) * x, M) / weight
            scale = bincount(np.float64(w) * np.abs(x), M) / weight
            got = GROUPS.mean(x, weights=w, valid=M) / scale
            assert_matches_oracle(got, expected / scale, rtol=0.0, atol=atol)
        assert GROUPS.mean(x, weights=w, valid=M).dtype == dtype

    def test_count(self) -> None:
        got = GROUPS.count(X, valid=M)
        assert_matches_oracle(got, bincount(np.ones_like(X), M).astype(int), rtol=0.0)
        plain = GROUPS.count(X)
        expected = np.bincount(LABELS[LABELS >= 0], minlength=G)
        assert_matches_oracle(plain[0], expected, rtol=0.0)

    @given(
        st.integers(1, 6).flatmap(
            lambda g: st.tuples(
                st.just(g), st.lists(st.integers(-1, g - 1), min_size=0, max_size=30)
            )
        )
    )
    def test_any_labels(self, case: tuple[int, list[int]]) -> None:
        n_groups, labels = case
        groups = Groups.from_labels(np.asarray(labels, dtype=int), n_groups)
        values = np.arange(1.0, len(labels) + 1.0)
        used = [label for label in labels if label >= 0]
        weights = [v for v, label in zip(values, labels, strict=True) if label >= 0]
        # whole numbers, so every sum is exact
        expected = np.bincount(
            np.asarray(used, dtype=int), weights=weights, minlength=n_groups
        )
        assert_matches_oracle(groups.sum(values), expected, rtol=0.0)

    def test_the_mean_without_weights_is_the_one_with_equal_weights(self) -> None:
        plain = GROUPS.mean(X, valid=M)
        assert_matches_oracle(
            GROUPS.mean(X, weights=2.5, valid=M), plain, rtol=1e-13, atol=1e-15
        )

    def test_an_average_over_rings_keeps_float32(self) -> None:
        # the case of #15: 512 × 512 power values into rings about a centre
        n = 512
        k = np.fft.fftfreq(n, 1.0 / n)
        ring = np.rint(np.hypot(k[:, None], k[None, :])).astype(int).ravel()
        groups = Groups.from_labels(ring, int(ring.max()) + 1)
        power = np.random.default_rng(1).exponential(size=n * n)
        expected = np.bincount(ring, weights=power.astype(np.float32).astype(float))
        got = groups.sum(power.astype(np.float32))
        assert_matches_oracle(got, expected, rtol=1e-6)

    def test_the_stored_monthly_climatology_and_anomaly(self) -> None:
        stored = xrops_values("climate")
        groups = Groups.from_labels(stored["month"] - 1, 12)
        monthly = groups.mean(stored["values"])
        assert_matches_oracle(monthly, stored["monthly_climatology"], rtol=1e-12)
        anomaly = stored["values"] - groups.spread(monthly)
        assert_matches_oracle(
            anomaly, stored["monthly_anomaly"], rtol=1e-12, atol=1e-14
        )


class TestAxes:
    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
    def test_the_groups_take_the_place_of_the_samples(self, axis: int) -> None:
        x = np.moveaxis(RNG.normal(size=(2, 3, N)), -1, axis)
        use = np.moveaxis(RNG.random((2, 3, N)) > 0.5, -1, axis)
        last = np.moveaxis(x, axis, -1)
        use_last = np.moveaxis(use, axis, -1)
        for method in (GROUPS.sum, GROUPS.mean, GROUPS.count):
            expected = np.moveaxis(method(last, valid=use_last), -1, axis)
            assert_matches_oracle(method(x, valid=use, axis=axis), expected, rtol=0.0)
        grouped = GROUPS.mean(x, axis=axis)
        spread = GROUPS.spread(grouped, axis=axis)
        assert spread.shape == x.shape
        expected = np.moveaxis(GROUPS.spread(np.moveaxis(grouped, axis, -1)), -1, axis)
        assert_matches_oracle(spread, expected, rtol=0.0)

    @pytest.mark.parametrize("axis", [0, 2, -3])
    def test_an_axis_of_another_size(self, axis: int) -> None:
        for method in (GROUPS.sum, GROUPS.mean, GROUPS.count):
            with pytest.raises(DiagnostixError, match=r"must hold the 200 samples"):
                method(X, axis=axis)
        with pytest.raises(DiagnostixError, match=r"must hold the 12 groups"):
            GROUPS.spread(X, axis=axis)


class TestSpread:
    def test_after_the_mean_of_one_member_groups_it_is_the_identity(self) -> None:
        order = RNG.permutation(N)
        groups = Groups.from_labels(order, N)
        assert_matches_oracle(groups.spread(groups.mean(X)), X, rtol=0.0)

    def test_a_sample_in_no_group_is_nan(self) -> None:
        spread = GROUPS.spread(np.arange(float(G)))
        assert_matches_oracle(np.isnan(spread), LABELS < 0, rtol=0.0)
        assert_matches_oracle(
            spread[LABELS >= 0], LABELS[LABELS >= 0].astype(float), rtol=0.0
        )

    def test_it_is_the_adjoint_of_the_sum(self) -> None:
        labels = np.where(LABELS < 0, 0, LABELS)
        groups = Groups.from_labels(labels, G)
        assert_adjoint(groups.sum, groups.spread, N, G)


class TestMissing:
    def test_a_nan_in_an_invalid_sample_changes_nothing(self) -> None:
        spoiled = np.where(M, X, np.nan)
        assert_matches_oracle(sum_of(spoiled, M), sum_of(X, M), rtol=0.0)
        assert_matches_oracle(mean_of(spoiled, W, M), mean_of(X, W, M), rtol=0.0)
        weights = np.where(M, W, np.nan)
        assert_matches_oracle(mean_of(X, weights, M), mean_of(X, W, M), rtol=0.0)

    def test_a_nan_in_a_valid_sample_spoils_its_group_only(self) -> None:
        spoiled = X.copy()
        spoiled[0, np.argmax(LABELS == 5)] = np.nan
        for out in (GROUPS.sum(spoiled), GROUPS.mean(spoiled)):
            assert np.isnan(out[0, 5])
        assert np.isnan(GROUPS.sum(spoiled)).sum() == 1

    def test_a_nan_at_the_position_the_padding_reads(self) -> None:
        groups = Groups.from_labels([0, 1, 1, 1], 3)
        values = np.asarray([np.nan, 1.0, 2.0, 3.0])
        assert_matches_oracle(groups.sum(values), [np.nan, 6.0, 0.0], rtol=0.0)
        assert_matches_oracle(groups.mean(values), [np.nan, 2.0, np.nan], rtol=0.0)

    def test_a_nan_in_no_group_changes_nothing(self) -> None:
        spoiled = np.where(LABELS < 0, np.nan, X)
        assert_matches_oracle(GROUPS.sum(spoiled), GROUPS.sum(X), rtol=0.0)
        assert_matches_oracle(GROUPS.mean(spoiled), GROUPS.mean(X), rtol=0.0)

    def test_an_empty_group(self) -> None:
        for empty in (3, 11):
            assert (GROUPS.sum(X)[:, empty] == 0.0).all()
            assert (GROUPS.count(X)[:, empty] == 0).all()
            assert np.isnan(GROUPS.mean(X)[:, empty]).all()

    def test_a_group_with_no_weight(self) -> None:
        weights = np.where(LABELS == 5, 0.0, W)
        assert np.isnan(GROUPS.mean(X, weights=weights)[:, 5]).all()

    def test_every_group_empty(self) -> None:
        for labels in (np.full(4, -1), np.zeros(0, dtype=int)):
            groups = Groups.from_labels(labels, 2)
            values = np.ones(labels.size)
            assert_matches_oracle(groups.sum(values), np.zeros(2), rtol=0.0)
            assert np.isnan(groups.mean(values)).all()
            assert np.isnan(groups.spread(np.ones(2))).all()

    def test_the_gradient_is_finite_next_to_a_nan_and_an_empty_group(self) -> None:
        few = slice(0, 40)
        groups = Groups.from_labels(LABELS[few], G)
        x, w, m = X[0, few], W[few], M[0, few]
        spoiled = np.where(m, x, np.nan)
        assert_finite_gradient(lambda x: groups.sum(x, valid=m), spoiled)
        assert_finite_gradient(lambda x: groups.mean(x, weights=w, valid=m), spoiled)
        weights = np.where(LABELS[few] == 5, 0.0, w)
        assert_finite_gradient(lambda w: groups.mean(x, weights=w), weights)
        assert_finite_gradient(groups.spread, np.ones(G))


class TestBackends:
    @pytest.mark.slow
    def test_agree(self) -> None:
        assert_backends_agree(sum_of, POSITIVE, M)
        assert_backends_agree(mean_of, POSITIVE, W, M)
        assert_backends_agree(lambda m: GROUPS.count(m, valid=m), M)
        assert_backends_agree(GROUPS.spread, X[:, :G])

    def test_an_axis_first(self) -> None:
        assert_backends_agree(
            lambda x, w: GROUPS.mean(x, weights=w, axis=0), POSITIVE.T, W[:, None]
        )

    @pytest.mark.slow
    def test_traces(self) -> None:
        assert_traces(sum_of, X, M)
        assert_traces(mean_of, X, np.broadcast_to(W, X.shape), M)
        assert_traces(GROUPS.spread, X[:, :G])

    @pytest.mark.slow
    def test_gradients(self) -> None:
        few = slice(0, 40)
        groups = Groups.from_labels(LABELS[few], G)
        x, w, m = X[0, few], W[few], M[0, few]
        assert_grad_matches_fd(lambda x: groups.sum(x, valid=m), x)
        assert_grad_matches_fd(lambda x: finite(groups.mean(x, weights=w)), x)
        assert_grad_matches_fd(lambda w: finite(groups.mean(x, weights=w)), w)
        assert_grad_matches_fd(lambda g: finite(groups.spread(g)), X[0, :G])

    def test_labels_of_every_library(self) -> None:
        for kind in KINDS:
            if kind.name in ("numpy", "jax"):
                groups = Groups.from_labels(kind.make(LABELS), G)
                assert_matches_oracle(groups.index, GROUPS.index, rtol=0.0)

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(sum_of, X, M)
        assert_refuses_dataarrays(mean_of, X, W, M)
        assert_refuses_dataarrays(GROUPS.count, X)
        assert_refuses_dataarrays(GROUPS.spread, X[:, :G])


def finite(values: Any) -> Any:
    """``values`` with 0 in place of the NaN of an empty group or a lone sample."""
    import jax.numpy as jnp

    return jnp.where(jnp.isnan(values), 0.0, values)


class TestJit:
    @pytest.mark.jax
    def test_a_closure_and_a_static_argument(self) -> None:
        import jax

        traces = []

        def grouped_mean(groups: Groups, x: Any) -> Any:
            traces.append(groups)
            return groups.mean(x)

        expected = GROUPS.mean(X)
        closure = jax.jit(lambda x: GROUPS.mean(x))
        assert_matches_oracle(closure(X), expected, rtol=1e-13)
        static = jax.jit(grouped_mean, static_argnums=0)
        for _ in range(2):
            assert_matches_oracle(static(GROUPS, X), expected, rtol=1e-13)
        # the second call reuses the first one's compilation
        assert len(traces) == 1
        static(Groups.from_labels(LABELS, G), X)
        assert len(traces) == 2


class TestFromLabels:
    def test_the_tables(self) -> None:
        groups = Groups.from_labels([2, 0, -1, 2, 0, 2], 4)
        assert groups.index.tolist() == [[1, 4, 0], [0, 0, 0], [0, 3, 5], [0, 0, 0]]
        assert groups.member.sum(axis=-1).tolist() == [2, 0, 3, 0]
        assert (groups.n_groups, groups.n_samples) == (4, 6)

    def test_the_tables_are_read_only_copies(self) -> None:
        labels = np.asarray([0, 1])
        groups = Groups.from_labels(labels, 2)
        labels[0] = 1
        assert groups.labels.tolist() == [0, 1]
        for table in (groups.index, groups.member, groups.labels):
            with pytest.raises(ValueError, match="read-only"):
                table[0] = 0

    @pytest.mark.parametrize("label", [3, 99, -2])
    def test_a_label_outside(self, label: int) -> None:
        with pytest.raises(DiagnostixError, match=rf"got {label} at position 1"):
            Groups.from_labels([0, label, 1], 3)

    @pytest.mark.parametrize("labels", [[0.0, 1.0], [True, False], [[0, 1]], 0, ["a"]])
    def test_labels_that_are_not_a_row_of_integers(self, labels: Any) -> None:
        with pytest.raises(DiagnostixError, match="1-D array of integers"):
            Groups.from_labels(labels, 2)

    def test_no_group(self) -> None:
        with pytest.raises(DiagnostixError, match="'n_groups' must be at least 1"):
            Groups.from_labels([], 0)
