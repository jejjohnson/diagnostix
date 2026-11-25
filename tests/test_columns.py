"""Diagnostics of columns."""

from __future__ import annotations

from typing import Any

import gridax.analysis
import numpy as np
import pytest
from _backends import (
    GRID_KINDS,
    assert_backends_agree,
    assert_refuses_dataarrays,
    needs_jax,
)
from _harness import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import (
    linear_pycnocline,
    neutral_layer_under_inversion,
    xrops_values,
)

from diagnostix import ValidationError
from diagnostix.columns import (
    boundary_layer_height,
    bulk_richardson,
    buoyancy_frequency_squared,
    column_integral,
    first_crossing,
    lapse_rate,
    mixed_layer_depth,
    vertical_derivative,
)


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

    @pytest.mark.slow
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
    @pytest.mark.slow
    def test_transforms(self) -> None:
        rng = np.random.default_rng(3)
        f = rng.normal(size=(3, 5, 9)).cumsum(axis=-1)
        z = np.sort(rng.uniform(size=f.shape), axis=-1)
        assert_traces(first_crossing, f, z, rng.uniform(size=(3, 5)))

    @needs_jax
    @pytest.mark.slow
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

    @pytest.mark.slow
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
    @pytest.mark.slow
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
    @pytest.mark.slow
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
        with pytest.raises(ValueError, match="'valid' must be boolean"):
            mixed_layer_depth(DEPTH, DEPTH, valid=valid)


def sheared_columns(seed: int, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """θ_v, u, v and z of stable, sheared columns, the levels last."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(10.0, 2000.0, size=shape), axis=-1)
    theta = 295.0 + 0.004 * z + 0.2 * rng.random(shape)
    u = 5.0 + 0.004 * z + 0.5 * rng.random(shape)
    v = 0.001 * z + 0.5 * rng.random(shape)
    return theta, u, v, z


class TestBulkRichardson:
    def test_closed_form(self) -> None:
        theta, u, v, z = sheared_columns(0, (4, 10))
        shear = (u - u[:, :1]) ** 2 + (v - v[:, :1]) ** 2 + 80.0 * 0.4**2
        rise = 9.80665 * (theta - theta[:, :1]) * (z - z[:, :1])
        out = bulk_richardson(
            theta, u, v, z, friction_velocity=0.4, friction_coefficient=80.0
        )
        assert_matches_oracle(out, rise / theta[:, :1] / shear, rtol=1e-13)
        assert_matches_oracle(out[:, 0], np.zeros(4), rtol=0.0)

    def test_calm_levels(self) -> None:
        # neutral, stable, unstable, and a NaN, with no wind anywhere
        theta = np.asarray([300.0, 300.0, 301.0, 299.0, NAN])
        calm = np.full(5, 2.0)
        with np.errstate(all="raise"):
            out = bulk_richardson(theta, calm, calm, np.arange(5.0))
        assert_matches_oracle(out, np.asarray([0.0, 0.0, INF, -INF, NAN]), rtol=0.0)

    def test_levels_on_another_axis(self) -> None:
        theta, u, v, z = sheared_columns(1, (4, 3, 10))
        expected = bulk_richardson(theta, u, v, z)
        assert expected.shape == theta.shape
        moved = (np.moveaxis(field, -1, 1) for field in (theta, u, v, z))
        out = bulk_richardson(*moved, axis=1)
        assert_matches_oracle(np.moveaxis(out, 1, -1), expected, rtol=0.0)

    def test_backends_and_precision(self) -> None:
        # powers of two, so that float32 keeps the differences
        z = np.asarray([0.0, 64.0, 128.0, 256.0])
        theta = np.asarray([[256.0, 256.0, 257.0, 260.0], [256.0, 255.0, 258.0, 256.0]])
        u = np.asarray([2.0, 4.0, 2.0, 6.0])
        assert_backends_agree(
            lambda theta, u, z: bulk_richardson(theta, u, 0.5 * u, z), theta, u, z
        )
        friction = np.asarray([0.0, 0.5])
        assert_backends_agree(
            lambda theta, u, z, friction: bulk_richardson(
                theta, u, 0.5 * u, z, friction_velocity=friction
            ),
            theta,
            u,
            z,
            friction,
        )

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(bulk_richardson, *sheared_columns(2, (2, 4)))


class TestBoundaryLayerHeight:
    def test_neutral_layer_under_an_inversion(self) -> None:
        height = np.asarray([10.0, 150.0, 400.0, 700.0, 850.0, 1000.0, 1300.0])
        profile = neutral_layer_under_inversion(height)
        fields = profile.fields
        out = boundary_layer_height(fields["theta_v"], fields["u"], fields["v"], height)
        # 300 K in float64 is known to 6e-14, over a rise of 0.25 K
        assert_matches_oracle(out, np.asarray(profile.expected), rtol=1e-10)

    def test_stored_xrops_values(self) -> None:
        stored = xrops_values("columns")
        out = boundary_layer_height(
            stored["air_theta_v"],
            stored["air_u"],
            stored["air_v"],
            stored["air_height"],
        )
        # a calm level, and a neutral column that never gets there
        assert np.isnan(out[5]) and not np.isnan(out[:5]).any()
        assert_matches_oracle(out, stored["air_boundary_layer_height"], rtol=1e-12)

    def test_the_example_of_plumax(self) -> None:
        # plumax met/_src/pbl.py at 96449d2, whose levels are 25, 75, …, 975 m
        z = np.linspace(25.0, 975.0, 20)
        theta = 300.0 + 0.01 * np.maximum(z - 400.0, 0.0)
        out = boundary_layer_height(theta, 2.0 + 0.005 * z, np.zeros(20), z)
        # Ri_b(425 m) = g · 0.25 · 400 / (300 · 2²), from 0 at 375 m
        upper = 9.80665 * 0.25 * 400.0 / (300.0 * 4.0)
        assert_matches_oracle(out, np.asarray(375.0 + 50.0 * 0.25 / upper), rtol=1e-10)

    @pytest.mark.parametrize(
        ("aloft", "expected"),
        [
            pytest.param(301.0, 100.0, id="stable: the level below"),
            pytest.param(299.0, NAN, id="unstable"),
            pytest.param(300.0, NAN, id="neutral"),
        ],
    )
    def test_a_calm_level(self, aloft: float, expected: float) -> None:
        # the second level is calm against the surface, the first is sheared
        theta = np.asarray([300.0, 300.0, aloft])
        u = np.asarray([2.0, 4.0, 2.0])
        z = np.asarray([0.0, 100.0, 200.0])
        out = boundary_layer_height(theta, u, np.zeros(3), z)
        assert_matches_oracle(out, np.asarray(expected), rtol=0.0)

    def test_a_calm_column_with_a_friction_velocity(self) -> None:
        z = np.linspace(0.0, 1000.0, 11)
        theta = 300.0 + 0.005 * z
        out = boundary_layer_height(
            theta, np.zeros(11), np.zeros(11), z, friction_velocity=0.2
        )
        # Ri_b = g γ z² / (θ_s b u∗²) is 0.409 at 100 m, from 0 at the surface
        upper = 9.80665 * 0.005 * 100.0**2 / (300.0 * 100.0 * 0.2**2)
        assert_matches_oracle(out, np.asarray(100.0 * 0.25 / upper), rtol=1e-12)

    def test_a_friction_velocity_per_column(self) -> None:
        theta, u, v, z = sheared_columns(3, (5, 3, 12))
        friction = np.linspace(0.1, 1.0, 15).reshape(5, 3)
        out = boundary_layer_height(theta, u, v, z, friction_velocity=friction)
        one = boundary_layer_height(
            theta[2, 1], u[2, 1], v[2, 1], z[2, 1], friction_velocity=friction[2, 1]
        )
        assert out.shape == (5, 3)
        assert_matches_oracle(out[2, 1], one, rtol=0.0)

    def test_a_nan_below_the_height(self) -> None:
        theta, u, v, z = sheared_columns(4, (2, 12))
        theta[1, 1] = NAN
        out = boundary_layer_height(theta, u, v, z)
        assert np.isfinite(out[0]) and np.isnan(out[1])

    def test_levels_on_another_axis(self) -> None:
        theta, u, v, z = sheared_columns(5, (4, 3, 12))
        expected = boundary_layer_height(theta, u, v, z)
        moved = (np.moveaxis(field, -1, 0) for field in (theta, u, v, z))
        out = boundary_layer_height(*moved, axis=0)
        assert_matches_oracle(out, expected, rtol=0.0)

    def test_backends_and_precision(self) -> None:
        z = np.asarray([0.0, 64.0, 128.0, 256.0])
        theta = np.asarray([[256.0, 256.0, 257.0, 260.0], [256.0, 255.0, 256.0, 256.0]])
        u = np.asarray([2.0, 4.0, 6.0, 8.0])
        assert_backends_agree(boundary_layer_height, theta, u, 0.5 * u, z)

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(boundary_layer_height, *sheared_columns(6, (2, 4)))

    @needs_jax
    def test_transforms(self) -> None:
        assert_traces(boundary_layer_height, *sheared_columns(7, (3, 4, 12)))

    @needs_jax
    @pytest.mark.slow
    def test_gradients(self) -> None:
        theta, u, v, z = sheared_columns(8, (2, 10))
        assert_grad_matches_fd(
            lambda x: boundary_layer_height(x, u, v, z), theta, rtol=1e-5
        )
        assert_grad_matches_fd(
            lambda x: boundary_layer_height(theta, x, v, z), u, rtol=1e-5
        )
        assert_grad_matches_fd(
            lambda x: boundary_layer_height(
                theta, u, v, z, friction_velocity=x, critical=0.1
            ),
            np.full(2, 0.3),
            rtol=1e-5,
        )

    @needs_jax
    def test_gradients_of_the_undefined_are_finite(self) -> None:
        # calm and stable, calm and unstable, neutral, and all calm
        theta = np.asarray(
            [
                [300.0, 300.0, 301.0, 302.0],
                [300.0, 300.0, 299.0, 302.0],
                [300.0, 300.0, 300.0, 300.0],
                [300.0, 300.5, 301.0, 302.0],
            ]
        )
        u = np.asarray([[2.0, 4.0, 2.0, 5.0]] * 3 + [[2.0] * 4])
        v = np.zeros((4, 4))
        z = np.asarray([0.0, 100.0, 200.0, 300.0])
        assert_finite_gradient(lambda x: boundary_layer_height(x, u, v, z), theta)
        assert_finite_gradient(lambda x: boundary_layer_height(theta, x, v, z), u)


def uneven_levels(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Increasing levels (last axis) between half and one and a half apart."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.5, 1.5, size=shape), axis=-1)


class TestVerticalDerivative:
    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["up", "down"])
    @pytest.mark.parametrize("shape", [(9,), (4, 3, 9)], ids=["shared", "per column"])
    def test_a_quadratic_is_exact_inside(
        self, shape: tuple[int, ...], sign: float
    ) -> None:
        z = sign * uneven_levels(0, shape)
        values = np.broadcast_to(0.7 * z**2 - 2.0 * z + 1.0, (4, 3, 9))
        out = vertical_derivative(values, z)
        expected = np.broadcast_to(1.4 * z - 2.0, out.shape)
        assert_matches_oracle(out[..., 1:-1], expected[..., 1:-1], rtol=0.0, atol=1e-12)

    def test_shared_levels_are_gridax(self) -> None:
        z = uneven_levels(1, (9,))
        values = np.random.default_rng(1).normal(size=(4, 9))
        expected = gridax.analysis.difference(values, axis=-1, coords=z)
        assert_matches_oracle(vertical_derivative(values, z), expected, rtol=0.0)

    def test_the_two_paths_agree(self) -> None:
        z = uneven_levels(2, (9,))
        values = np.random.default_rng(2).normal(size=(4, 9))
        shared = vertical_derivative(values, z)
        columns = vertical_derivative(values, np.broadcast_to(z, values.shape))
        # a few units in the last place of the slopes, which are of order 1
        assert_matches_oracle(columns, shared, rtol=0.0, atol=2e-14)

    @pytest.mark.parametrize("shared", [True, False])
    def test_two_levels(self, shared: bool) -> None:
        values = np.asarray([[0.0, 3.0], [1.0, 0.0]])
        z = np.asarray([0.0, 2.0])
        z = z if shared else np.broadcast_to(z, values.shape)
        expected = np.asarray([[1.5, 1.5], [-0.5, -0.5]])
        assert_matches_oracle(vertical_derivative(values, z), expected, rtol=0.0)

    @pytest.mark.parametrize("axis", [0, 1, -2])
    def test_levels_on_another_axis(self, axis: int) -> None:
        z = uneven_levels(3, (4, 3, 9))
        values = np.sin(z)
        expected = vertical_derivative(values, z)
        out = vertical_derivative(
            np.moveaxis(values, -1, axis), np.moveaxis(z, -1, axis), axis=axis
        )
        assert_matches_oracle(np.moveaxis(out, axis, -1), expected, rtol=0.0)
        shared = vertical_derivative(np.moveaxis(values, -1, axis), z[0, 0], axis=axis)
        expected = vertical_derivative(values, z[0, 0])
        assert_matches_oracle(np.moveaxis(shared, axis, -1), expected, rtol=0.0)

    def test_a_shared_profile_on_per_column_levels(self) -> None:
        z = uneven_levels(4, (4, 9))
        profile = np.arange(9.0)
        out = vertical_derivative(profile, z)
        expected = vertical_derivative(np.broadcast_to(profile, z.shape), z)
        assert_matches_oracle(out, expected, rtol=0.0)

    def test_a_nan_reaches_the_levels_beside_it(self) -> None:
        z = uneven_levels(5, (2, 9))
        values = z.copy()
        values[0, 4] = NAN
        out = vertical_derivative(values, z)
        assert np.argwhere(np.isnan(out)).tolist() == [[0, 3], [0, 4], [0, 5]]

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        z = np.cumsum(np.asarray([0.5, 1.0, 1.5, 0.5, 2.0, 1.0]))
        values = np.stack([z**2, 2.0 * z])
        assert_backends_agree(
            vertical_derivative, values, np.stack([z, 2.0 * z]), rtol32=1e-5
        )
        # gridax, on a shared coordinate
        assert_backends_agree(vertical_derivative, values, z, kinds=GRID_KINDS)

    def test_refuses_dataarrays(self) -> None:
        z = uneven_levels(6, (2, 5))
        assert_refuses_dataarrays(vertical_derivative, z, z)
        assert_refuses_dataarrays(vertical_derivative, z, z[0])

    @needs_jax
    @pytest.mark.slow
    def test_transforms(self) -> None:
        z = uneven_levels(7, (3, 4, 9))
        values = np.sin(z)
        assert_traces(vertical_derivative, values, z, rtol=1e-11)
        assert_traces(vertical_derivative, values, z[:, 0], rtol=1e-11)

    @needs_jax
    @pytest.mark.slow
    def test_gradients(self) -> None:
        z = uneven_levels(8, (2, 5))
        values = np.sin(z)
        assert_grad_matches_fd(lambda f: vertical_derivative(f, z), values)
        assert_grad_matches_fd(lambda z: vertical_derivative(values, z), z)
        assert_grad_matches_fd(lambda f: vertical_derivative(f, z[0]), values)

    def test_levels_that_differ(self) -> None:
        with pytest.raises(ValidationError, match=r"differ in levels: \[3, 4\]"):
            vertical_derivative(np.zeros((2, 3)), np.arange(4.0))


class TestBuoyancyFrequencySquared:
    def test_a_linear_density_is_constant(self) -> None:
        z = uneven_levels(0, (3, 8))
        density = 1025.0 - 0.02 * z
        out = buoyancy_frequency_squared(
            density, z, reference_density=1000.0, gravity=10.0
        )
        # 1025 kg m⁻³ in float64 is known to 2e-13, over a step of 0.01
        assert_matches_oracle(out, np.full(z.shape, 2e-4), rtol=1e-10)

    def test_a_stable_column_is_positive_in_either_order(self) -> None:
        z = uneven_levels(1, (3, 8))
        density = 1025.0 - np.sqrt(z)
        out = buoyancy_frequency_squared(density, z)
        assert (out > 0).all()
        flipped = buoyancy_frequency_squared(density[:, ::-1], z[:, ::-1])
        assert_matches_oracle(flipped[:, ::-1], out, rtol=1e-12)

    def test_stored_xrops_values(self) -> None:
        stored = xrops_values("columns")
        out = buoyancy_frequency_squared(
            stored["ocean_density"],
            -stored["ocean_depth"],
            reference_density=1025.0,
            gravity=9.80665,
        )
        # 1025 kg m⁻³ is known to 2.3e-13, over 5 m, times g/ρ₀
        assert_matches_oracle(out, stored["ocean_n_squared"], rtol=0.0, atol=5e-16)

    def test_backends_and_precision(self) -> None:
        # an anomaly, so that float32 keeps it
        z = np.cumsum(np.asarray([[0.5, 1.0, 1.5, 0.5], [1.0, 0.5, 0.5, 2.0]]), -1)
        assert_backends_agree(buoyancy_frequency_squared, -0.02 * z, z)

    def test_refuses_dataarrays(self) -> None:
        z = uneven_levels(2, (2, 5))
        assert_refuses_dataarrays(buoyancy_frequency_squared, z, z)

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        z = uneven_levels(3, (3, 2, 6))
        density = -np.sqrt(z)
        assert_traces(buoyancy_frequency_squared, density, z, rtol=1e-11)
        assert_grad_matches_fd(
            lambda rho: buoyancy_frequency_squared(rho, z[0]), density[0]
        )


class TestLapseRate:
    def test_a_linear_temperature(self) -> None:
        z = 100.0 * uneven_levels(0, (3, 8))
        out = lapse_rate(288.15 - 0.0065 * z, z)
        # 288 K in float64 is known to 6e-14, over a step of 0.3 K
        assert_matches_oracle(out, np.full(z.shape, 0.0065), rtol=1e-11)

    def test_is_minus_the_derivative_on_every_axis(self) -> None:
        z = uneven_levels(1, (8, 3))
        values = np.sin(z)
        z = np.sort(z, axis=0)
        expected = -vertical_derivative(values, z, axis=0)
        assert_matches_oracle(lapse_rate(values, z, axis=0), expected, rtol=0.0)

    def test_backends_and_precision(self) -> None:
        z = np.cumsum(np.asarray([[0.5, 1.0, 1.5, 0.5], [1.0, 0.5, 0.5, 2.0]]), -1)
        assert_backends_agree(lapse_rate, -2.0 * z, z)

    def test_refuses_dataarrays(self) -> None:
        z = uneven_levels(2, (2, 5))
        assert_refuses_dataarrays(lapse_rate, z, z)

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        z = uneven_levels(3, (3, 2, 6))
        assert_traces(lapse_rate, np.sqrt(z), z, rtol=1e-11)
        assert_grad_matches_fd(lambda t: lapse_rate(t, z[0]), np.sqrt(z[0]))


class TestColumnIntegral:
    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["up", "down"])
    def test_a_constant_and_a_line_are_exact(self, sign: float) -> None:
        z = sign * uneven_levels(0, (3, 8))
        ends = z[:, -1], z[:, 0]
        constant = column_integral(np.full(z.shape, 2.5), z)
        assert_matches_oracle(constant, 2.5 * (ends[0] - ends[1]), rtol=1e-14)
        line = column_integral(3.0 * z - 1.0, z)
        expected = 1.5 * (ends[0] ** 2 - ends[1] ** 2) - (ends[0] - ends[1])
        assert_matches_oracle(line, expected, rtol=1e-13)

    @pytest.mark.parametrize("method", ["trapezoid", "sum"])
    @pytest.mark.parametrize("levels", ["per_column", "shared"])
    def test_stored_xrops_values(self, levels: str, method: str) -> None:
        # xrops' two rules are one sum, in two orders
        stored = xrops_values("columns")
        height = stored["air_height"]
        z = height if levels == "per_column" else height[0]
        out = column_integral(stored["air_theta_v"], z)
        expected = stored[f"air_integral_{levels}_{method}"]
        assert_matches_oracle(out, expected, rtol=1e-12)

    def test_levels_on_another_axis(self) -> None:
        z = uneven_levels(1, (4, 3, 8))
        expected = column_integral(np.sin(z), z)
        moved = np.moveaxis(z, -1, 0)
        out = column_integral(np.sin(moved), moved, axis=0)
        assert_matches_oracle(out, expected, rtol=1e-14)

    def test_a_nan_makes_its_column_nan(self) -> None:
        z = uneven_levels(2, (2, 8))
        values = np.ones_like(z)
        values[1, -1] = NAN
        assert np.isnan(column_integral(values, z)).tolist() == [False, True]

    def test_backends_and_precision(self) -> None:
        z = np.cumsum(np.asarray([[0.5, 1.0, 1.5, 0.5], [1.0, 0.5, 0.5, 2.0]]), -1)
        assert_backends_agree(column_integral, z**2, z)
        assert_backends_agree(column_integral, z**2, z[0])

    def test_refuses_dataarrays(self) -> None:
        z = uneven_levels(3, (2, 5))
        assert_refuses_dataarrays(column_integral, z, z)

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        z = uneven_levels(4, (3, 2, 6))
        values = np.cos(z)
        assert_traces(column_integral, values, z)
        assert_grad_matches_fd(lambda f: column_integral(f, z[0]), values[0])
        assert_grad_matches_fd(lambda z: column_integral(values[0], z), z[0])
