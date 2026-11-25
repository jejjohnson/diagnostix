"""Area-weighted EOFs, and fields in terms of them."""

from __future__ import annotations

from typing import Any

import gridax
import numpy as np
import pytest
from _backends import assert_backends_agree, assert_refuses_dataarrays, needs_jax
from _harness import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import functax_eofs, sklearn_pca, two_patterns
from hypothesis import given, settings, strategies as st

from diagnostix import ValidationError
from diagnostix.eofs import EOFs, eofs, project, reconstruct


GRID = gridax.SphericalAGrid2D.from_coords(
    lat=np.linspace(-75.0, 75.0, 6), lon=np.linspace(0.0, 300.0, 8)
)
AREAS = np.broadcast_to(np.asarray(gridax.area_weights(GRID)), (6, 8))
#: Of mean 1, so that tolerances are relative to the fields.
AREAS = AREAS / np.mean(AREAS)


def noise(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def masked(shape: tuple[int, ...] = (6, 8)) -> np.ndarray:
    """A mask that leaves out a block and a few cells."""
    valid = np.ones(shape, dtype=bool)
    valid[1:3, 2:5] = False
    valid[-1, 0] = False
    return valid


def parts(out: EOFs) -> tuple[Any, ...]:
    return out.modes, out.principal_components, out.explained_variance


class TestEofs:
    def test_two_patterns(self) -> None:
        known = two_patterns()
        out = eofs(known.field, n_modes=2)
        for mode, amplitude in zip(range(2), known.amplitudes.T, strict=True):
            sign = np.sign(np.sum(out.modes[mode] * known.modes[mode]))
            assert_matches_oracle(
                out.modes[mode], sign * known.modes[mode], rtol=0.0, atol=1e-13
            )
            assert_matches_oracle(
                out.principal_components[:, mode], sign * amplitude, rtol=0, atol=1e-13
            )
        assert_matches_oracle(out.explained_variance, [0.9, 0.1], rtol=1e-13)

    @pytest.mark.parametrize("shape", [(30, 6, 8), (9, 6, 8)])
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_orthonormal_under_the_weights(self, shape: Any, with_valid: bool) -> None:
        valid = masked() if with_valid else None
        x = noise(0, *shape)
        n_modes = min(shape[0], 41 if with_valid else 48)
        out = eofs(x, n_modes=n_modes, weights=AREAS, valid=valid)
        e = np.nan_to_num(out.modes).reshape(n_modes, -1)
        gram = (e * AREAS.ravel()) @ e.T
        assert_matches_oracle(gram, np.eye(n_modes), rtol=0.0, atol=1e-12)
        # the principal components are uncorrelated, with the shares as variances
        a = out.principal_components
        covariance = a.T @ a
        inside = 1.0 if valid is None else valid
        total = np.sum(x**2 * AREAS * inside)
        expected = np.diag(out.explained_variance) * total
        assert_matches_oracle(covariance, expected, rtol=0.0, atol=1e-12 * total)
        assert (np.diff(out.explained_variance) <= 0.0).all()

    def test_the_modes_maximise_the_weighted_variance(self) -> None:
        x = noise(1, 40, 6, 8) * (1.0 + np.arange(6.0))[:, None]
        out = eofs(x, n_modes=1, weights=AREAS)
        best = float(out.explained_variance[0] * np.sum(x**2 * AREAS))
        for seed in range(20):
            other = noise(seed, 6, 8)
            other = other / np.sqrt(np.sum(AREAS * other**2))
            variance = np.sum(np.sum(AREAS * x * other, axis=(-2, -1)) ** 2)
            assert variance < best

    def test_a_cell_of_half_the_weight_twice(self) -> None:
        x = noise(2, 12, 5)
        out = eofs(x, n_modes=3)
        twice = np.concatenate([x, x[:, 2:3]], axis=-1)
        again = eofs(twice, n_modes=3, weights=[1.0, 1.0, 0.5, 1.0, 1.0, 0.5])
        assert_matches_oracle(again.modes[:, :5], out.modes, rtol=1e-10)
        assert_matches_oracle(
            again.principal_components, out.principal_components, rtol=1e-10
        )

    def test_scaled_weights(self) -> None:
        x = noise(3, 12, 5)
        out, more = (eofs(x, n_modes=3, weights=w) for w in (1.0, 16.0))
        assert_matches_oracle(more.modes, out.modes / 4.0, rtol=1e-12)
        assert_matches_oracle(
            more.principal_components, 4.0 * out.principal_components, rtol=1e-12
        )
        assert_matches_oracle(
            more.explained_variance, out.explained_variance, rtol=1e-12
        )

    @settings(max_examples=20)
    @given(st.integers(0, 10_000))
    def test_the_largest_loading_is_positive(self, seed: int) -> None:
        x = noise(seed, 8, 3, 4)
        for field in (x, -x):
            modes = eofs(field, n_modes=4).modes.reshape(4, -1)
            largest = np.argmax(np.abs(modes), axis=-1)
            assert (modes[np.arange(4), largest] > 0.0).all()

    def test_a_masked_cell_changes_nothing(self) -> None:
        valid = masked()
        x = noise(4, 10, 6, 8)
        out = eofs(x, n_modes=4, weights=AREAS, valid=valid)
        spoiled = np.where(valid, x, np.nan)
        again = eofs(spoiled, n_modes=4, weights=AREAS, valid=valid)
        for got, expected in zip(parts(again), parts(out), strict=True):
            assert_matches_oracle(got, expected, rtol=0.0)
        assert np.isnan(out.modes[:, ~valid]).all()
        assert np.isfinite(out.modes[:, valid]).all()
        # a mask of rows broadcasts, and the cells alone give the same modes
        rows = np.asarray([True, True, False, True, True, False])[:, None]
        by_rows = eofs(x, n_modes=4, valid=rows)
        alone = eofs(x[:, rows[:, 0]], n_modes=4)
        assert_matches_oracle(by_rows.modes[:, rows[:, 0]], alone.modes, rtol=1e-11)

    def test_a_cell_without_weight_is_like_a_masked_one(self) -> None:
        valid = masked()
        x = noise(5, 10, 6, 8)
        out = eofs(x, n_modes=4, valid=valid)
        x[:, ~valid] = np.nan
        again = eofs(x, n_modes=4, weights=valid.astype(float))
        assert_matches_oracle(again.modes, out.modes, rtol=0.0, atol=1e-12)
        assert_matches_oracle(
            again.explained_variance, out.explained_variance, rtol=1e-12
        )

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_nan_in_a_valid_cell_is_everywhere(self, bad: float) -> None:
        x = noise(6, 8, 3, 4)
        x[2, 1, 1] = bad
        for part in parts(eofs(x, n_modes=2)):
            assert np.isnan(part).all()

    def test_no_variance(self) -> None:
        out = eofs(np.zeros((4, 6)), n_modes=2)
        assert np.isnan(out.explained_variance).all()
        assert_matches_oracle(out.principal_components, np.zeros((4, 2)), rtol=0.0)

    @pytest.mark.oracle
    def test_scikit_learn(self) -> None:
        x = noise(7, 25, 12)
        x = x - x.mean(axis=0)
        modes, amplitudes, variance = sklearn_pca(x, 5)
        out = eofs(x, n_modes=5)
        sign = np.sign(np.sum(modes * out.modes, axis=-1))
        assert_matches_oracle(out.modes, sign[:, None] * modes, rtol=0, atol=1e-11)
        assert_matches_oracle(
            out.principal_components, sign * amplitudes, rtol=0, atol=1e-11
        )
        share = variance / np.sum(np.var(x, axis=0, ddof=1))
        assert_matches_oracle(out.explained_variance, share, rtol=1e-11)

    @pytest.mark.oracle
    def test_functax(self) -> None:
        x = noise(8, 25, 12)
        x = x - x.mean(axis=0)
        modes, variance = functax_eofs(x, 5)
        out = eofs(x, n_modes=5)
        sign = np.sign(np.sum(modes * out.modes, axis=-1))
        assert_matches_oracle(out.modes, sign[:, None] * modes, rtol=0, atol=1e-11)
        share = variance / np.sum(np.var(x, axis=0, ddof=1))
        assert_matches_oracle(out.explained_variance, share, rtol=1e-11)

    @pytest.mark.slow
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_backends_and_precision(self, with_valid: bool) -> None:
        valid = masked((4, 6)) if with_valid else None

        def of(x: Any, w: Any) -> Any:
            return parts(eofs(x, n_modes=3, weights=w, valid=valid))

        # well separated singular values, for float32
        x = noise(9, 8, 4, 6) * np.exp(-np.arange(8.0))[:, None, None]
        w = 0.5 + np.random.default_rng(10).random((4, 6))
        assert_backends_agree(of, x, w, rtol=1e-10, rtol32=5e-3)
        assert_traces(of, x[None], w[None], rtol=1e-9)
        assert_refuses_dataarrays(of, x, w)

    @pytest.mark.slow
    @needs_jax
    def test_gradient_of_the_explained_variance(self) -> None:
        x, w = noise(11, 5, 3, 4), 0.5 + noise(12, 3, 4) ** 2

        def share(x: Any, w: Any) -> Any:
            return eofs(x, n_modes=3, weights=w).explained_variance

        assert_grad_matches_fd(lambda x: share(x, w), x)
        assert_grad_matches_fd(lambda w: share(x, w), w)

    @needs_jax
    def test_gradient_beside_invalid_cells(self) -> None:
        valid = np.asarray([[True, False, True], [True, True, True]])
        x = np.where(valid, noise(13, 5, 2, 3), np.nan)

        def share(x: Any) -> Any:
            return eofs(x, n_modes=2, valid=valid).explained_variance

        assert_finite_gradient(share, x)
        # and beside cells that have no weight
        w = valid.astype(float)
        x = noise(14, 5, 2, 3)
        assert_finite_gradient(
            lambda w: eofs(x, n_modes=2, weights=w).explained_variance, w
        )

    @needs_jax
    def test_the_record_is_a_pytree(self) -> None:
        import jax

        out = eofs(noise(15, 5, 4), n_modes=2)
        leaves, tree = jax.tree.flatten(out)
        assert len(leaves) == 3
        assert isinstance(jax.tree.unflatten(tree, leaves), EOFs)

    def test_errors(self) -> None:
        x = noise(16, 5, 3, 4)
        for n_modes in (0, 6):
            with pytest.raises(ValidationError, match="'n_modes' must be"):
                eofs(x, n_modes=n_modes)
        few = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValidationError, match="leaves no cell"):
            eofs(x, n_modes=1, valid=few)
        few[0, :2] = True
        with pytest.raises(ValidationError, match="and 2 cells"):
            eofs(x, n_modes=3, valid=few)
        with pytest.raises(ValidationError, match="must be boolean"):
            eofs(x, n_modes=1, valid=np.ones((3, 4)))
        with pytest.raises(ValidationError, match="'valid' of shape"):
            eofs(x, n_modes=1, valid=np.ones((4, 3), dtype=bool))
        for shape in ((4, 3), (2, 3, 4)):
            with pytest.raises(ValidationError, match="'weights' of shape"):
                eofs(x, n_modes=1, weights=np.ones(shape))
        with pytest.raises(ValidationError, match="a time axis and a grid"):
            eofs(x[:, 0, 0], n_modes=1)


class TestProjectAndReconstruct:
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_round_trip(self, with_valid: bool) -> None:
        valid = masked() if with_valid else None
        x = noise(20, 10, 6, 8)
        out = eofs(x, n_modes=10, weights=AREAS, valid=valid)
        a = project(x, out, weights=AREAS, valid=valid)
        assert_matches_oracle(a, out.principal_components, rtol=0.0, atol=1e-11)
        expected = x if valid is None else np.where(valid, x, np.nan)
        for back in (reconstruct(out), reconstruct(out, a)):
            assert_matches_oracle(back, expected, rtol=0.0, atol=1e-12)

    def test_the_best_approximation(self) -> None:
        # of a field, the rest is orthogonal to the modes kept
        x = noise(21, 10, 6, 8)
        out = eofs(x, n_modes=3, weights=AREAS)
        rest = x - reconstruct(out)
        a = project(rest, out, weights=AREAS)
        assert_matches_oracle(a, np.zeros((10, 3)), rtol=0.0, atol=1e-12)
        lost = np.sum(AREAS * rest**2) / np.sum(AREAS * x**2)
        assert lost == pytest.approx(1.0 - np.sum(out.explained_variance), rel=1e-11)

    def test_leading_axes(self) -> None:
        out = eofs(noise(22, 10, 6, 8), n_modes=3, weights=AREAS)
        field = noise(23, 2, 4, 6, 8)
        a = project(field, out, weights=AREAS)
        assert a.shape == (2, 4, 3)
        one = project(field[1, 2], out, weights=AREAS)
        assert_matches_oracle(a[1, 2], one, rtol=1e-12)
        back = reconstruct(out, a)
        assert back.shape == (2, 4, 6, 8)
        assert_matches_oracle(back[1, 2], reconstruct(out, one), rtol=1e-12)

    def test_masks(self) -> None:
        valid = masked()
        x = noise(24, 10, 6, 8)
        out = eofs(x, n_modes=3, valid=valid)
        spoiled = np.where(valid, x, np.nan)
        a = project(spoiled, out, valid=valid)
        assert_matches_oracle(a, out.principal_components, rtol=0.0, atol=1e-12)
        # without the mask, the modes' own NaNs say so
        assert np.isnan(project(x, out)).all()
        # weights of 0 in place of a mask
        w = valid.astype(float)
        again = eofs(x, n_modes=3, weights=w)
        a = project(spoiled, again, weights=w)
        assert_matches_oracle(a, out.principal_components, rtol=0.0, atol=1e-12)
        # a NaN in a valid cell is in every amplitude of its field
        spoiled[4, 0, 0] = np.nan
        a = project(spoiled, out, valid=valid)
        assert np.isnan(a[4]).all()
        assert np.isfinite(np.delete(a, 4, axis=0)).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_backends_and_precision(self, with_valid: bool) -> None:
        valid = masked((4, 6)) if with_valid else None
        fitted = eofs(noise(25, 8, 4, 6), n_modes=3, valid=valid)

        def of(x: Any, modes: Any, w: Any) -> Any:
            record = EOFs(modes, x, x)
            a = project(x, record, weights=w, valid=valid)
            return 9.0 + a, 9.0 + reconstruct(record, a)

        x, w = noise(26, 2, 4, 6), 0.5 + np.random.default_rng(27).random((4, 6))
        modes = np.asarray(fitted.modes)
        assert_backends_agree(of, x, modes, w)
        assert_traces(of, x[None], modes[None], w[None])
        assert_refuses_dataarrays(of, x, modes, w)
        assert_grad_matches_fd(lambda x: of(x, modes, w)[0], x)

    def test_errors(self) -> None:
        out = eofs(noise(28, 5, 3, 4), n_modes=2)
        with pytest.raises(ValidationError, match="must end in the grid"):
            project(noise(29, 4, 3), out)
        with pytest.raises(ValidationError, match="must hold 2 modes"):
            reconstruct(out, np.ones(3))
        with pytest.raises(ValidationError, match="must hold 2 modes"):
            reconstruct(out, 1.0)
        with pytest.raises(ValidationError, match="'weights' of shape"):
            project(noise(30, 3, 4), out, weights=np.ones(3))
