"""Budgets over a control volume."""

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

from diagnostix import ValidationError
from diagnostix.budgets import (
    ControlVolume,
    boundary_flux,
    face_fluxes,
    flux_divergence,
)


GRIDS = {
    "plane": gridax.CartesianAGrid2D.from_interior(9, 8, 4.5, 6.0),
    "channel": gridax.CartesianAGrid2D.from_interior(9, 8, 4.5, 6.0, periodic=("x",)),
    "torus": gridax.CartesianAGrid2D.from_interior(9, 8, 4.5, 6.0, periodic=("x", "y")),
    "uneven plane": gridax.CartesianAGrid2D.from_coords(
        y=[7.0, 5.0, 4.5, 3.0, 1.0, 0.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
    ),
    "sphere": gridax.SphericalAGrid2D.from_interior(
        12, 10, (0.0, 360.0), (-90.0, 90.0), R=2.0, periodic=("x",)
    ),
    "uneven sphere": gridax.SphericalAGrid2D.from_coords(
        lat=[65.0, 40.0, 15.0, -10.0, -35.0, -70.0],
        lon=[0.0, 8.0, 20.0, 30.0, 45.0, 70.0, 80.0],
        R=2.0,
    ),
}
NAMES = sorted(GRIDS)


def fields(seed: int, grid: Any, *batch: int) -> list[np.ndarray]:
    """q, u and v of order 1, positive, so that float32 keeps their sums."""
    rng = np.random.default_rng(seed)
    return list(1.0 + rng.random((3, *batch, *grid.shape)))


def region(seed: int, grid: Any) -> np.ndarray:
    """A random region, with holes and parts, off the edges that do not wrap."""
    inside = np.random.default_rng(seed).random(grid.shape) > 0.4
    if "x" not in grid.periodic:
        inside[:, [0, -1]] = False
    if "y" not in grid.periodic:
        inside[[0, -1], :] = False
    return inside


def box(grid: Any, rows: slice, columns: slice) -> ControlVolume:
    inside = np.zeros(grid.shape, dtype=bool)
    inside[rows, columns] = True
    return ControlVolume.from_region(inside, grid)


class TestControlVolume:
    def test_a_box_on_a_plane(self) -> None:
        volume = box(GRIDS["plane"], slice(2, 5), slice(3, 7))
        assert volume.east_sign.shape == (8, 8)
        assert volume.north_sign.shape == (7, 9)
        expected = np.zeros((8, 8))
        expected[2:5, 2], expected[2:5, 6] = -1, 1
        assert_matches_oracle(volume.east_sign, expected.astype(np.int8), rtol=0.0)
        expected = np.zeros((7, 9))
        expected[1, 3:7], expected[4, 3:7] = -1, 1
        assert_matches_oracle(volume.north_sign, expected.astype(np.int8), rtol=0.0)

    def test_a_box_across_the_seam(self) -> None:
        grid = GRIDS["channel"]
        inside = np.zeros(grid.shape, dtype=bool)
        inside[2:5, [0, 1, -1]] = True
        volume = ControlVolume.from_region(inside, grid)
        assert volume.east_sign.shape == grid.shape
        # entered between cells 7 and 8, left between 1 and 2, whole at the seam
        assert volume.east_sign[2].tolist() == [0, 1, 0, 0, 0, 0, 0, -1, 0]

    def test_a_hole_has_faces_of_its_own(self) -> None:
        grid = GRIDS["plane"]
        inside = np.zeros(grid.shape, dtype=bool)
        inside[1:6, 1:6] = True
        inside[3, 3] = False
        volume = ControlVolume.from_region(inside, grid)
        assert volume.east_sign[3].tolist() == [-1, 0, 1, -1, 0, 1, 0, 0]

    def test_north_is_towards_larger_y(self) -> None:
        # y falls with the index, so the region is left northwards at its first row
        volume = box(GRIDS["uneven plane"], slice(2, 4), slice(2, 4))
        assert volume.north_sign[:, 2].tolist() == [0, 1, 0, -1, 0]

    def test_the_tables_are_read_only_copies(self) -> None:
        grid = GRIDS["plane"]
        inside = region(0, grid)
        volume = ControlVolume.from_region(inside, grid)
        before = volume.inside.copy()
        inside[:] = False
        assert_matches_oracle(volume.inside, before, rtol=0.0)
        for table in (volume.inside, volume.east_sign, volume.north_sign):
            with pytest.raises(ValueError, match="read-only"):
                table[0, 0] = 1

    @pytest.mark.parametrize(
        ("rows", "columns", "axis"),
        [
            (slice(0, 3), slice(2, 4), "y"),
            (slice(3, 8), slice(2, 4), "y"),
            (slice(2, 4), slice(0, 3), "x"),
            (slice(2, 4), slice(3, 9), "x"),
        ],
    )
    def test_a_region_at_an_edge(self, rows: slice, columns: slice, axis: str) -> None:
        with pytest.raises(ValidationError, match=f"edge of the grid along {axis}"):
            box(GRIDS["plane"], rows, columns)

    @pytest.mark.parametrize("value", [False, True])
    def test_an_empty_or_a_full_region(self, value: bool) -> None:
        with pytest.raises(ValidationError, match="and not all of them"):
            ControlVolume.from_region(
                np.full(GRIDS["torus"].shape, value), GRIDS["torus"]
            )

    @pytest.mark.parametrize(
        "inside", [np.ones((8, 9)), np.ones((9, 8), dtype=bool), [True, False]]
    )
    def test_not_a_boolean_field(self, inside: Any) -> None:
        with pytest.raises(ValidationError, match=r"boolean of shape \(8, 9\)"):
            ControlVolume.from_region(inside, GRIDS["plane"])

    def test_a_grid_that_is_not_collocated(self) -> None:
        grid: Any = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
        with pytest.raises(ValidationError, match="must be a CartesianAGrid2D or"):
            ControlVolume.from_region(np.ones(grid.shape, dtype=bool), grid)


class TestFaceFluxes:
    def test_on_a_uniform_plane(self) -> None:
        # cells of 0.5 m by 0.75 m
        grid = GRIDS["plane"]
        q, u, v = fields(0, grid)
        east, north = face_fluxes(q, u, v, grid)
        expected = 0.5 * ((q * u)[:, :-1] + (q * u)[:, 1:]) * 0.75
        assert_matches_oracle(east, expected, rtol=1e-14)
        expected = 0.5 * ((q * v)[:-1] + (q * v)[1:]) * 0.5
        assert_matches_oracle(north, expected, rtol=1e-14)

    def test_on_a_sphere(self) -> None:
        # cells of 30° by 18°, on a radius of 2 m
        grid = GRIDS["sphere"]
        q, u, v = fields(1, grid)
        east, north = face_fluxes(q, u, v, grid)
        assert east.shape == (10, 12)
        wrapped = 0.5 * (q * u + np.roll(q * u, -1, axis=-1))
        assert_matches_oracle(east, wrapped * 2.0 * np.deg2rad(18.0), rtol=1e-13)
        between = np.deg2rad(np.arange(-72.0, 73.0, 18.0))[:, None]
        length = 2.0 * np.cos(between) * np.deg2rad(30.0)
        expected = 0.5 * ((q * v)[:-1] + (q * v)[1:]) * length
        assert_matches_oracle(north, expected, rtol=1e-13)

    def test_the_thickness_multiplies(self) -> None:
        grid = GRIDS["uneven sphere"]
        q, u, v = fields(2, grid)
        height = fields(3, grid)[0]
        out = face_fluxes(q, u, v, grid, thickness=height)
        expected = face_fluxes(q * height, u, v, grid)
        for value, want in zip(out, expected, strict=True):
            assert_matches_oracle(value, want, rtol=1e-14)

    @pytest.mark.parametrize("name", NAMES)
    def test_leading_axes(self, name: str) -> None:
        grid = GRIDS[name]
        q, u, v = fields(4, grid, 2)
        out = face_fluxes(q[0], u, v, grid)
        for index in range(2):
            expected = face_fluxes(q[0], u[index], v[index], grid)
            for value, want in zip(out, expected, strict=True):
                assert_matches_oracle(value[index], want, rtol=0.0)

    @pytest.mark.parametrize("name", NAMES)
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        assert_backends_agree(
            lambda q, u, v: face_fluxes(q, u, v, grid), *fields(5, grid)
        )

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        q, u, v = fields(6, grid)
        assert_refuses_dataarrays(
            lambda q, u, v, h: face_fluxes(q, u, v, grid, thickness=h), q, u, v, q
        )

    def test_a_velocity_of_another_shape(self) -> None:
        with pytest.raises(ValidationError, match=r"ends in \(8, 9\) \(y, x\)"):
            face_fluxes(1.0, np.zeros((9, 8)), np.zeros((8, 9)), GRIDS["plane"])


class TestBoundaryFlux:
    @pytest.mark.parametrize("name", ["plane", "channel", "torus", "uneven plane"])
    def test_a_uniform_flow_passes_through(self, name: str) -> None:
        grid = GRIDS[name]
        volume = ControlVolume.from_region(region(0, grid), grid)
        u, v = np.full(grid.shape, 3.0), np.full(grid.shape, -2.0)
        out = boundary_flux(1.5, u, v, grid, volume)
        assert_matches_oracle(out, np.asarray(0.0), rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("name", ["plane", "uneven plane"])
    def test_a_radial_flow_is_twice_the_area(self, name: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        volume = ControlVolume.from_region(region(1, grid), grid)
        # (x_{i+½} − x_{i−½}) Δy + … is the cell, whose faces are half-way
        area = gridax.area_weights(grid)[volume.inside].sum()
        out = boundary_flux(1.0, x, y, grid, volume)
        assert_matches_oracle(out, np.asarray(2.0 * area), rtol=1e-13)

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("name", NAMES)
    def test_it_closes_with_the_divergence(self, name: str, seed: int) -> None:
        grid = GRIDS[name]
        rng = np.random.default_rng(seed)
        q, u, v, height = rng.normal(size=(4, *grid.shape))
        volume = ControlVolume.from_region(region(seed, grid), grid)
        out = boundary_flux(q, u, v, grid, volume, thickness=height)
        inside = flux_divergence(q, u, v, grid, thickness=height)[volume.inside]
        inside = inside * gridax.area_weights(grid)[volume.inside]
        assert abs(out - inside.sum()) < 1e-12 * np.abs(inside).sum()

    @pytest.mark.parametrize("name", ["plane", "channel", "torus"])
    def test_the_divergence_of_gridax_on_a_plane(self, name: str) -> None:
        grid = GRIDS[name]
        q, u, v = np.random.default_rng(2).normal(size=(3, *grid.shape))
        volume = ControlVolume.from_region(region(2, grid), grid)
        inside = gridax.divergence(q * u, q * v, grid) * gridax.area_weights(grid)
        inside = inside[volume.inside]
        out = boundary_flux(q, u, v, grid, volume)
        assert abs(out - inside.sum()) < 1e-13 * np.abs(inside).sum()

    def test_a_plume_of_known_rate(self) -> None:
        # c = Q/(u √(2π) σ(x)) exp(−y²/2σ²) downwind of the source, 0 upwind
        grid = gridax.CartesianAGrid2D.from_interior(80, 81, 8.0e3, 8.1e3, y0=-4.05e3)
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        wind, rate, source = 4.0, 2.5, 2.0e3
        sigma = 200.0 + 0.1 * np.maximum(x - source, 0.0)
        shape = np.exp(-0.5 * (y / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
        column = np.where(x > source, rate / wind * shape, 0.0)
        u, v = np.full(grid.shape, wind), np.zeros(grid.shape)
        around = box(grid, slice(10, 71), slice(10, 50))
        downwind = box(grid, slice(10, 71), slice(35, 60))
        # the Gaussian is summed over 3 km either side, at least 5 σ
        out = boundary_flux(column, u, v, grid, around)
        assert_matches_oracle(out, np.asarray(rate), rtol=1e-6)
        out = boundary_flux(column, u, v, grid, downwind)
        assert_matches_oracle(out, np.asarray(0.0), rtol=0.0, atol=1e-6 * rate)

    def test_only_the_cells_beside_the_boundary_are_read(self) -> None:
        grid = GRIDS["plane"]
        volume = box(grid, slice(2, 6), slice(2, 7))
        q, u, v = fields(3, grid)
        expected = boundary_flux(q, u, v, grid, volume)
        # the middle of the box, and a corner outside it
        q[3:5, 3:6] = np.nan
        q[0, 0] = np.nan
        out = boundary_flux(q, u, v, grid, volume)
        assert_matches_oracle(out, expected, rtol=0.0)
        q[4, 2] = np.nan
        assert np.isnan(boundary_flux(q, u, v, grid, volume))

    @pytest.mark.parametrize("name", NAMES)
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        q, u, v = fields(4, grid, 2)
        volume = ControlVolume.from_region(region(4, grid), grid)
        # signed sums of terms of order 1, relative to their number
        faces = np.abs(volume.east_sign).sum() + np.abs(volume.north_sign).sum()

        def shifted(q: Any, u: Any, v: Any) -> Any:
            return boundary_flux(q, u, v, grid, volume) + 10.0 * float(faces)

        assert_backends_agree(shifted, q, u, v)

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        import jax

        grid = GRIDS["sphere"]
        q, u, v = fields(5, grid, 3)
        volume = ControlVolume.from_region(region(5, grid), grid)
        assert_traces(
            lambda q, u, v: boundary_flux(q, u, v, grid, volume),
            q,
            u,
            v,
            rtol=1e-11,
        )
        assert_grad_matches_fd(
            lambda q: boundary_flux(q, u[0], v[0], grid, volume), q[0]
        )
        # the volume is a static argument, and the grid a traced one
        static = jax.jit(boundary_flux, static_argnums=4)
        out = static(q, u, v, grid, volume)
        assert_matches_oracle(out, boundary_flux(q, u, v, grid, volume), rtol=1e-11)

    @needs_jax
    def test_gradients_are_finite_beside_a_nan_off_the_boundary(self) -> None:
        grid = GRIDS["plane"]
        volume = box(grid, slice(2, 6), slice(2, 7))
        q, u, v = fields(6, grid)
        q[3:5, 3:6] = np.nan
        assert_finite_gradient(
            lambda q: boundary_flux(q, u, v, grid, volume), q, where=~np.isnan(q)
        )

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        volume = box(grid, slice(2, 6), slice(2, 7))
        assert_refuses_dataarrays(
            lambda q, u, v: boundary_flux(q, u, v, grid, volume), *fields(7, grid)
        )

    def test_a_volume_of_another_grid(self) -> None:
        volume = box(GRIDS["plane"], slice(2, 6), slice(2, 7))
        grid = GRIDS["sphere"]
        with pytest.raises(ValidationError, match=r"of a grid of shape \(8, 9\)"):
            boundary_flux(*fields(8, grid), grid, volume)


class TestFluxDivergence:
    def test_a_linear_flow_is_exact(self) -> None:
        grid = GRIDS["uneven plane"]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        out = flux_divergence(2.0, 3.0 * x, -0.5 * y, grid)
        assert_matches_oracle(out[1:-1, 1:-1], np.full((4, 5), 5.0), rtol=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_edge_cells_that_do_not_wrap_are_nan(self, name: str) -> None:
        grid = GRIDS[name]
        out = flux_divergence(*fields(0, grid), grid)
        edge = np.zeros(grid.shape, dtype=bool)
        if "x" not in grid.periodic:
            edge[:, [0, -1]] = True
        if "y" not in grid.periodic:
            edge[[0, -1], :] = True
        assert_matches_oracle(np.isnan(out), edge, rtol=0.0)

    def test_rotation_on_the_sphere_converges(self) -> None:
        worst = []
        for step in (3.0, 1.5, 0.75):
            grid = gridax.SphericalAGrid2D.from_interior(
                round(360 / step),
                round(120 / step),
                (0.0, 360.0),
                (-60.0, 60.0),
                R=2.0,
                periodic=("x",),
            )
            lam, phi = np.meshgrid(grid.coords("x"), grid.coords("y"))
            # a rotation about (1, 0, 1): no divergence
            u = np.cos(phi) - np.sin(phi) * np.cos(lam)
            out = flux_divergence(1.0, u, np.sin(lam), grid)
            worst.append(np.abs(out[1:-1]).max())
        assert_matches_oracle(
            np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.4
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        # it grows with both coordinates, so that nothing cancels in float32
        u, v = 1.0 + 9.0 * np.sin(x / 2.0) ** 2, 10.0 * y
        sure = np.full(grid.shape, 1.0e3)
        assert_backends_agree(
            lambda q, u, v, c: flux_divergence(q, u, v, grid) + c,
            np.ones(grid.shape),
            u,
            v,
            sure,
            rtol=1e-11,
        )

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        import jax
        import jax.numpy as jnp

        grid = GRIDS["uneven sphere"]
        q, u, v = fields(1, grid, 3)
        assert_traces(
            lambda q, u, v: flux_divergence(q, u, v, grid), q, u, v, rtol=1e-10
        )

        def inner(u: Any) -> Any:
            return flux_divergence(q[0], u, v[0], grid)[1:-1, 1:-1]

        assert_grad_matches_fd(inner, u[0])
        # finite on the edges too, whose divergence is NaN
        assert_finite_gradient(
            lambda u: jnp.nan_to_num(flux_divergence(q[0], u, v[0], grid)), u[0]
        )
        # a traced grid
        out = jax.jit(flux_divergence)(q, u, v, grid)
        assert_matches_oracle(out, flux_divergence(q, u, v, grid), rtol=1e-10)

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        assert_refuses_dataarrays(
            lambda q, u, v: flux_divergence(q, u, v, grid), *fields(2, grid)
        )
