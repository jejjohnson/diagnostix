"""Diagnostics of columns."""

from __future__ import annotations

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
from _oracles import linear_pycnocline, xrops_values

from diagnostix import ValidationError
from diagnostix.columns import first_crossing, mixed_layer_depth


NAN, INF = np.nan, np.inf


def loop_crossing(f: np.ndarray, z: np.ndarray, level: float, start: int = 0) -> float:
    """The first crossing of one column, by a loop."""
    if f[start] >= level:
        return float(z[start])
    for k in range(start + 1, f.size):
        if f[k] >= level:
            fraction = (level - f[k - 1]) / (f[k] - f[k - 1])
            return float(z[k - 1] + fraction * (z[k] - z[k - 1]))
    return NAN


class TestFirstCrossing:
    @pytest.mark.parametrize("start", [0, 3])
    def test_equals_a_loop(self, start: int) -> None:
        rng = np.random.default_rng(0)
        f = rng.normal(size=(1000, 12)).cumsum(axis=-1)
        z = np.sort(rng.uniform(0.0, 100.0, size=f.shape), axis=-1)
        level = rng.uniform(0.0, 3.0, size=1000)
        expected = [loop_crossing(*row, start) for row in zip(f, z, level, strict=True)]
        # some columns cross and some do not
        assert 100 < np.isnan(expected).sum() < 900
        out = first_crossing(f, z, level, start=start)
        assert_matches_oracle(out, np.asarray(expected), rtol=0.0)

    def test_levels_on_another_axis(self) -> None:
        rng = np.random.default_rng(1)
        f = rng.normal(size=(4, 3, 8)).cumsum(axis=-1)
        z = np.arange(8.0)
        expected = first_crossing(f, z, 0.7)
        moved = np.moveaxis(f, -1, 0)
        assert_matches_oracle(first_crossing(moved, z, 0.7, axis=0), expected, rtol=0.0)
        per_column = np.broadcast_to(z[:, None, None], moved.shape)
        out = first_crossing(moved, per_column, 0.7, axis=0)
        assert_matches_oracle(out, expected, rtol=0.0)

    def test_a_shared_profile_on_per_column_levels(self) -> None:
        z = np.asarray([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
        out = first_crossing([0.0, 1.0, 2.0], z, 1.5)
        assert_matches_oracle(out, np.asarray([1.5, 15.0]), rtol=0.0)

    @pytest.mark.parametrize(
        ("f", "expected"),
        [
            pytest.param([0.0, 1.0, 2.0, 3.0], 1.5, id="between"),
            pytest.param([0.0, 1.5, 2.0, 3.0], 1.0, id="on a level"),
            pytest.param([2.0, 1.0, 3.0, 3.0], 0.0, id="at the start already"),
            pytest.param([2.0, 2.0, 2.0, 2.0], 0.0, id="always above"),
            pytest.param([2.0, 1.0, 1.0, 1.0], 0.0, id="only at the start"),
            pytest.param([2.0, NAN, 1.0, 1.0], 0.0, id="NaN after the start"),
            pytest.param([0.0, 1.0, 1.0, 1.0], NAN, id="never"),
            pytest.param([0.0, 1.0, INF, 3.0], 1.0, id="infinite above"),
            pytest.param([0.0, -INF, 2.0, 3.0], 2.0, id="minus infinity below"),
            pytest.param([0.0, -INF, INF, 3.0], 1.0, id="both infinite"),
            pytest.param([0.0, -INF, -INF, -INF], NAN, id="minus infinity only"),
            pytest.param([0.0, NAN, 2.0, 3.0], NAN, id="NaN before the crossing"),
            pytest.param([NAN, 0.0, 2.0, 3.0], NAN, id="NaN at the start"),
            pytest.param([0.0, 2.0, NAN, NAN], 0.75, id="NaN after the crossing"),
        ],
    )
    def test_cases(self, f: list[float], expected: float) -> None:
        with np.errstate(all="raise"):
            out = first_crossing(np.asarray(f), np.arange(4.0), 1.5)
        assert_matches_oracle(out, np.asarray(expected), rtol=0.0)

    def test_a_nan_level_is_nan(self) -> None:
        out = first_crossing(np.zeros((2, 3)), np.arange(3.0), np.asarray([NAN, 0.0]))
        assert_matches_oracle(out, np.asarray([NAN, 0.0]), rtol=0.0)

    def test_backends_and_precision(self) -> None:
        rng = np.random.default_rng(2)
        f = np.abs(rng.normal(size=(6, 9))).cumsum(axis=-1)
        z = np.arange(9.0) ** 1.5
        # the last column never gets there
        f[-1] = 0.1
        assert_backends_agree(lambda f, z: first_crossing(f, z, 2.0), f, z)
        level = np.full(6, 2.0)
        assert_backends_agree(first_crossing, f, z, level)

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(
            first_crossing, np.zeros((2, 3)), np.arange(3.0), np.asarray([1.0, 2.0])
        )

    @needs_jax
    def test_transforms(self) -> None:
        rng = np.random.default_rng(3)
        f = rng.normal(size=(3, 5, 9)).cumsum(axis=-1)
        z = np.sort(rng.uniform(size=f.shape), axis=-1)
        assert_traces(first_crossing, f, z, rng.uniform(size=(3, 5)))

    @needs_jax
    def test_gradients(self) -> None:
        rng = np.random.default_rng(4)
        f = np.abs(rng.normal(size=(3, 6))).cumsum(axis=-1)
        z = np.arange(6.0) ** 1.2
        assert_grad_matches_fd(lambda f: first_crossing(f, z, 1.3), f)
        assert_grad_matches_fd(lambda z: first_crossing(f, z, 1.3), z)
        assert_grad_matches_fd(
            lambda level: first_crossing(f, z, level), np.full(3, 1.3)
        )

    @needs_jax
    def test_gradients_of_the_undefined_are_finite(self) -> None:
        f = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.0, NAN, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, INF, 0.0],
                [0.0, -INF, 2.0],
            ]
        )
        z = np.arange(3.0)
        assert_finite_gradient(lambda f: first_crossing(f, z, 1.5), f)
        # the same in the other cells, with a NaN in one column's levels
        levels = np.broadcast_to(z, f.shape).copy()
        levels[1, 1] = NAN
        assert_finite_gradient(
            lambda z: first_crossing(f, z, 1.5), levels, where=~np.isnan(levels)
        )

    @pytest.mark.parametrize("start", [-1, 2, 3])
    def test_start_out_of_range(self, start: int) -> None:
        with pytest.raises(ValidationError, match="'start' must be in"):
            first_crossing(np.zeros(3), np.arange(3.0), 1.0, start=start)

    def test_levels_that_differ(self) -> None:
        with pytest.raises(ValidationError, match=r"differ in levels: \[3, 4\]"):
            first_crossing(np.zeros((2, 3)), np.arange(4.0), 1.0)

    def test_a_single_level(self) -> None:
        with pytest.raises(ValidationError, match="at least two levels, got 1"):
            first_crossing(np.zeros((2, 1)), np.zeros(1), 1.0)

    def test_no_such_axis(self) -> None:
        with pytest.raises(ValidationError, match="no axis 2 for the levels"):
            first_crossing(np.zeros((2, 3)), np.arange(3.0), 1.0, axis=2)
        with pytest.raises(ValidationError, match="no axis -1 for the levels"):
            first_crossing(1.0, np.arange(3.0), 1.0)


#: Uneven depths, 2.4 to 10 m apart, with 10 m between two of them.
DEPTH = np.asarray([0.0, 2.4, 6.0, 9.0, 13.0, 20.0, 29.0, 39.0, 49.0, 59.0])


class TestMixedLayerDepth:
    @pytest.mark.parametrize("depth", [DEPTH, DEPTH + 2.0], ids=["zero", "deeper"])
    def test_linear_pycnocline(self, depth: np.ndarray) -> None:
        # linear everywhere, so that the interpolation is exact
        profile = linear_pycnocline(depth, top=0.0, slope=0.002, threshold=0.03)
        out = mixed_layer_depth(profile.fields["density"], depth)
        # 1025 kg m⁻³ in float64 is known to 2e-13, over a slope of 0.002
        assert_matches_oracle(out, np.asarray(25.0), rtol=0.0, atol=1e-9)

    def test_reference_level_below_the_reference_depth(self) -> None:
        # the first valid level, at 13 m, is the reference: 13 + 15 m
        density = 1025.0 + 0.002 * DEPTH
        out = mixed_layer_depth(density, DEPTH, valid=DEPTH > 10.0)
        assert_matches_oracle(out, np.asarray(28.0), rtol=0.0, atol=1e-9)

    def test_an_invalid_level_is_left_out(self) -> None:
        density = 1025.0 + 0.002 * DEPTH
        # 20 and 29 m bracket the answer, and 13 and 9 the reference depth
        density[[3, 6]] = NAN
        out = mixed_layer_depth(density, DEPTH, valid=~np.isnan(density))
        assert_matches_oracle(out, np.asarray(25.0), rtol=0.0, atol=1e-9)

    def test_land_is_nan_and_a_mixed_column_is_its_depth(self) -> None:
        # to 100 m on 0, 10, …, 200 m
        depth = 10.0 * np.arange(21)
        density = np.full((3, 21), 1025.0)
        valid = np.stack([np.zeros(21, bool), depth <= 100.0, np.ones(21, bool)])
        density[~valid] = NAN
        out = mixed_layer_depth(density, depth, valid=valid)
        assert_matches_oracle(out, np.asarray([NAN, 100.0, 200.0]), rtol=0.0)
        out = mixed_layer_depth(density, depth, valid=valid, unreached="nan")
        assert np.isnan(out).all()

    def test_a_column_shallower_than_the_reference_depth(self) -> None:
        density = np.asarray([0.0, 1.0, 2.0])
        depth = np.asarray([0.0, 3.0, 6.0])
        assert float(mixed_layer_depth(density, depth)) == 6.0

    def test_a_single_valid_level(self) -> None:
        valid = np.asarray([False, True, False])
        out = mixed_layer_depth(np.zeros(3), np.asarray([0.0, 3.0, 6.0]), valid=valid)
        assert float(out) == 3.0

    def test_a_level_above_the_reference_depth_never_counts(self) -> None:
        depth = np.asarray([0.0, 5.0, 10.0, 20.0, 30.0])
        # denser than the threshold at the surface, and unstable below it
        density = np.asarray([9.0, 9.0, 0.0, 0.02, 0.06])
        out = mixed_layer_depth(density, depth)
        assert_matches_oracle(out, np.asarray(22.5), rtol=1e-12)

    def test_a_nan_in_a_valid_level(self) -> None:
        density = np.broadcast_to(0.002 * DEPTH, (3, DEPTH.size)).copy()
        # above the answer, at the reference, and below the answer
        density[0, 5] = density[1, 3] = density[2, 8] = NAN
        out = mixed_layer_depth(density, DEPTH)
        assert_matches_oracle(out, np.asarray([NAN, NAN, 25.0]), rtol=1e-12)

    def test_within_one_level_of_xrops(self) -> None:
        stored = xrops_values("columns")
        depth, level = stored["ocean_depth"], stored["ocean_mixed_layer_level"]
        out = mixed_layer_depth(stored["ocean_density"], depth)
        spacing = depth[1] - depth[0]
        # xrops gives the first level past the crossing
        assert ((level - spacing <= out) & (out <= level)).all()

    def test_per_column_depths_on_another_axis(self) -> None:
        depth = np.stack([DEPTH, 1.5 * DEPTH], axis=-1)
        out = mixed_layer_depth(0.002 * depth, depth, axis=0, valid=depth < 70.0)
        assert_matches_oracle(out, np.full(2, 25.0), rtol=1e-12)
        shared = mixed_layer_depth(0.002 * depth, depth, axis=0, valid=DEPTH < 50.0)
        assert_matches_oracle(shared, np.full(2, 25.0), rtol=1e-12)

    def test_backends_and_precision(self) -> None:
        depth = np.stack([DEPTH, 1.5 * DEPTH, DEPTH])
        density = 0.002 * depth
        valid = np.stack([depth[0] >= 0.0, depth[1] < 60.0, depth[0] < 0.0])
        assert_backends_agree(
            lambda rho, z, m: mixed_layer_depth(rho, z, valid=m), density, depth, valid
        )
        # a table of the host, next to arrays of any library
        assert_backends_agree(
            lambda rho, z: mixed_layer_depth(rho, z, valid=valid), density, depth
        )
        assert_backends_agree(mixed_layer_depth, density, DEPTH)

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(mixed_layer_depth, 0.002 * DEPTH, DEPTH)

    @needs_jax
    def test_transforms(self) -> None:
        rng = np.random.default_rng(5)
        depth = np.sort(rng.uniform(0.0, 80.0, size=(3, 4, 12)), axis=-1)
        valid = rng.random(depth.shape) > 0.2
        assert_traces(
            lambda rho, z, m: mixed_layer_depth(rho, z, valid=m),
            0.002 * depth,
            depth,
            valid,
        )

    @needs_jax
    def test_gradients(self) -> None:
        depth = np.stack([DEPTH, 1.5 * DEPTH])
        density = 0.002 * depth + 1e-6 * depth**2
        assert_grad_matches_fd(
            lambda rho: mixed_layer_depth(rho, depth), density, eps=1e-7, rtol=1e-5
        )

    @needs_jax
    def test_gradients_on_the_valid_levels_are_finite(self) -> None:
        depth = np.broadcast_to(DEPTH, (4, DEPTH.size))
        density = 0.002 * depth
        valid = np.ones(depth.shape, dtype=bool)
        # land, an ocean floor, a hole, and a column mixed to its bottom
        valid[0] = False
        valid[1, 7:] = False
        valid[2, 5] = False
        density[3] = 0.0
        density[~valid] = NAN
        assert_finite_gradient(
            lambda rho: mixed_layer_depth(rho, depth, valid=valid), density
        )

    def test_unreached_is_one_of_two(self) -> None:
        with pytest.raises(ValidationError, match="'unreached' must be 'bottom' or"):
            mixed_layer_depth(DEPTH, DEPTH, unreached="top")  # type: ignore[arg-type]

    def test_valid_is_boolean(self) -> None:
        valid: Any = np.ones(DEPTH.size)
        with pytest.raises(ValidationError, match="'valid' must be boolean"):
            mixed_layer_depth(DEPTH, DEPTH, valid=valid)
