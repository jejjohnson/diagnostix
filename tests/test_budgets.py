"""Budgets over a control volume."""

from __future__ import annotations

from typing import Any

import gridax
import numpy as np
import pytest
from _backends import assert_backends_agree, assert_refuses_dataarrays, needs_jax
from pipekit_array.testing import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from diagnostix import ValidationError
from diagnostix.budgets import (
    Budget,
    ControlVolume,
    boundary_flux,
    budget,
    cell_residual,
    face_fluxes,
    flux_divergence,
    region_integral,
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


TERMS = ("tendency", "boundary_flux", "source", "residual")
#: Uneven times, in seconds since 1970, whose steps float32 cannot hold.
TIMES = 1.7e9 + np.cumsum([0.0, 60.0, 30.0, 90.0, 60.0])


def terms(out: Budget) -> tuple[Any, ...]:
    return tuple(getattr(out, name) for name in TERMS)


def run(seed: int, grid: Any, *batch: int) -> list[np.ndarray]:
    """q, u, v, S and H over `TIMES`, positive and of order 1."""
    rng = np.random.default_rng(seed)
    return list(1.0 + rng.random((5, TIMES.size, *batch, *grid.shape)))


class TestRegionIntegral:
    @pytest.mark.parametrize("name", NAMES)
    def test_the_area_of_the_region(self, name: str) -> None:
        grid = GRIDS[name]
        volume = ControlVolume.from_region(region(0, grid), grid)
        area = gridax.area_weights(grid)[volume.inside].sum()
        out = region_integral(1.0, grid, volume)
        assert_matches_oracle(out, np.asarray(area), rtol=1e-14)

    def test_a_field_with_a_thickness(self) -> None:
        grid = GRIDS["uneven sphere"]
        q, _, _, _, height = run(0, grid)
        volume = ControlVolume.from_region(region(1, grid), grid)
        expected = (q * height * gridax.area_weights(grid))[:, volume.inside]
        out = region_integral(q, grid, volume, thickness=height)
        assert_matches_oracle(out, expected.sum(axis=-1), rtol=1e-13)

    def test_only_the_cells_inside_are_read(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(2, grid), grid)
        q = fields(0, grid)[0]
        expected = region_integral(q, grid, volume)
        spoiled = np.where(volume.inside, q, np.nan)
        assert_matches_oracle(
            region_integral(spoiled, grid, volume), expected, rtol=0.0
        )
        assert np.isnan(
            region_integral(np.where(volume.inside, np.nan, q), grid, volume)
        )

    @pytest.mark.parametrize("name", NAMES)
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        volume = ControlVolume.from_region(region(3, grid), grid)
        assert_backends_agree(
            lambda q, h: region_integral(q, grid, volume, thickness=h),
            *fields(1, grid, 2)[:2],
        )

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(4, grid), grid)
        q, height, _ = fields(2, grid)
        assert_refuses_dataarrays(
            lambda q, h: region_integral(q, grid, volume, thickness=h), q, height
        )

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(5, grid), grid)
        q = fields(3, grid, 3)[0]
        assert_traces(lambda q: region_integral(q, grid, volume), q)
        assert_grad_matches_fd(lambda q: region_integral(q, grid, volume), q[0])
        spoiled = np.where(volume.inside, q[0], np.nan)
        assert_finite_gradient(lambda q: region_integral(q, grid, volume), spoiled)


class TestBudget:
    @pytest.mark.parametrize("name", ["torus", "channel"])
    def test_a_forward_euler_run_closes(self, name: str) -> None:
        # q^{n+1} = q^n + Δt (S − ∇·(q u)), off the edges
        grid = GRIDS[name]
        u, v = np.random.default_rng(0).normal(size=(2, *grid.shape))
        added = np.random.default_rng(1).random(grid.shape)
        time = np.asarray([0.0, 0.1, 0.15, 0.3])
        steps = [np.ones(grid.shape)]
        for step in np.diff(time):
            out = np.nan_to_num(flux_divergence(steps[-1], u, v, grid))
            steps.append(steps[-1] + step * (added - out))
        q = np.stack(steps)
        flow = [np.broadcast_to(field, q.shape) for field in (u, v)]
        volume = ControlVolume.from_region(region(0, grid), grid)
        options = {"time": time, "source": added, "tendency": "forward"}
        out = budget(q, *flow, grid, volume, **options)
        scale = np.abs(out.tendency).max() + np.abs(out.boundary_flux).max()
        assert out.residual.shape == (3,)
        assert_matches_oracle(out.residual / scale, np.zeros(3), rtol=0.0, atol=1e-13)
        cells = cell_residual(q, *flow, grid, **options)[:, volume.inside]
        assert_matches_oracle(cells, np.zeros_like(cells), rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("tendency", ["centred", "forward"])
    @pytest.mark.parametrize("name", NAMES)
    def test_the_cells_add_up(self, name: str, tendency: Any) -> None:
        grid = GRIDS[name]
        q, u, v, added, height = run(1, grid)
        volume = ControlVolume.from_region(region(1, grid), grid)
        options = {
            "time": TIMES,
            "source": added,
            "thickness": height,
            "tendency": tendency,
        }
        out = budget(q, u, v, grid, volume, **options)
        area = gridax.area_weights(grid)[volume.inside]
        cells = cell_residual(q, u, v, grid, **options)[:, volume.inside] * area
        error = np.abs(cells.sum(axis=-1) - out.residual)
        assert (error < 1e-12 * np.abs(cells).sum(axis=-1)).all()
        assert_matches_oracle(
            out.residual,
            out.tendency + out.boundary_flux - out.source,
            rtol=0.0,
        )

    def test_a_closed_box_in_a_flow_without_divergence(self) -> None:
        # a vortex of the periodic plane, whose streamlines close in the box
        grid = gridax.CartesianAGrid2D.from_interior(
            32, 32, 1.0, 1.0, periodic=("x", "y")
        )
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        wave = 2.0 * np.pi
        u = -np.sin(wave * x) * np.cos(wave * y)
        v = np.cos(wave * x) * np.sin(wave * y)
        u, v = (np.broadcast_to(field, (3, 32, 32)) for field in (u, v))
        volume = box(grid, slice(8, 24), slice(8, 24))
        out = budget(1.0, u, v, grid, volume, time=[0.0, 1.0, 2.0])
        assert_matches_oracle(out.tendency, np.zeros(3), rtol=0.0)
        # every signed face has its mirror image
        assert_matches_oracle(out.boundary_flux, np.zeros(3), rtol=0.0, atol=1e-13)
        assert_matches_oracle(out.source, np.zeros(3), rtol=0.0)

    def test_a_box_cut_through_an_advected_blob(self) -> None:
        # the box on its flank: what flows in is what it gains, at second order
        # in Δt, and the space is resolved far better than that
        grid = gridax.CartesianAGrid2D.from_interior(
            512, 16, 2.0, 1.0, periodic=("x", "y")
        )
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        volume = box(grid, slice(4, 12), slice(160, 360))
        wind = 0.5

        def blob(t: float) -> np.ndarray:
            return np.exp(
                -(((x - 0.6 - wind * t) / 0.15) ** 2) - ((y - 0.5) / 0.1) ** 2
            )

        worst = []
        for step in (0.1, 0.05, 0.025):
            time = 0.2 + step * np.arange(-1, 2)
            q = np.stack([blob(t) for t in time])
            u, v = np.full(q.shape, wind), np.zeros(q.shape)
            out = budget(q, u, v, grid, volume, time=time)
            worst.append(abs(float(out.residual[1] / out.boundary_flux[1])))
        assert_matches_oracle(
            np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.6
        )

    def test_a_steady_source(self) -> None:
        # u c = Q(x): what the source adds flows out
        grid = gridax.CartesianAGrid2D.from_interior(40, 10, 40.0, 10.0)
        x, _ = np.meshgrid(grid.coords("x"), grid.coords("y"))
        added = np.where((x > 10.0) & (x < 20.0), 0.3, 0.0)
        wind = 2.0
        # ∂ₓ(u c) = S with the flux of a face the mean of its two cells
        faces = np.cumsum(added, axis=-1)
        column = np.zeros_like(faces)
        for i in range(1, 40):
            column[:, i] = 2.0 * faces[:, i - 1] / wind - column[:, i - 1]
        flow = np.full((3, *grid.shape), wind)
        volume = box(grid, slice(2, 8), slice(5, 30))
        out = budget(
            column, flow, 0.0 * flow, grid, volume, time=[0.0, 1.0, 2.0], source=added
        )
        assert_matches_oracle(
            out.source, np.full(3, added[volume.inside].sum()), rtol=1e-12
        )
        assert_matches_oracle(out.boundary_flux, out.source, rtol=1e-12)
        assert_matches_oracle(out.residual, np.zeros(3), rtol=0.0, atol=1e-11)

    def test_a_volume_budget_with_a_thickness(self) -> None:
        grid = GRIDS["sphere"]
        _, u, v, added, height = run(2, grid)
        volume = ControlVolume.from_region(region(2, grid), grid)
        out = budget(
            1.0, u, v, grid, volume, time=TIMES, source=added, thickness=height
        )
        held = region_integral(height, grid, volume)
        # numpy's second-order difference on uneven times, one-sided at the ends
        expected = np.gradient(held, TIMES - TIMES[0], edge_order=1)
        assert_matches_oracle(out.tendency, expected, rtol=1e-10)

    def test_a_centred_tendency_is_exact_for_a_parabola(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(3, grid), grid)
        t = TIMES - TIMES[0]
        rest = np.zeros((t.size, *grid.shape))
        q = (0.5 * t**2 - 3.0 * t)[:, None, None] + rest
        out = budget(q, rest, rest, grid, volume, time=TIMES)
        area = gridax.area_weights(grid)[volume.inside].sum()
        assert_matches_oracle(out.tendency[1:-1], area * (t - 3.0)[1:-1], rtol=1e-12)

    def test_batches_between_time_and_space(self) -> None:
        grid = GRIDS["uneven plane"]
        q, u, v, added, _ = run(4, grid, 2)
        volume = ControlVolume.from_region(region(4, grid), grid)
        for tendency in ("centred", "forward"):
            options: dict[str, Any] = {"time": TIMES, "tendency": tendency}
            out = budget(q, u, v, grid, volume, source=added, **options)
            for index in range(2):
                one = [field[:, index] for field in (q, u, v)]
                expected = budget(*one, grid, volume, source=added[:, index], **options)
                for value, want in zip(terms(out), terms(expected), strict=True):
                    assert_matches_oracle(value[:, index], want, rtol=1e-12)

    @pytest.mark.slow
    @pytest.mark.parametrize("tendency", ["centred", "forward"])
    def test_backends_and_precision(self, tendency: Any) -> None:
        grid = GRIDS["sphere"]
        q, u, v, added, height = run(5, grid)
        # it grows, so that its tendency does not cancel
        q = q + 50.0 * np.arange(TIMES.size)[:, None, None]
        volume = box(grid, slice(2, 7), slice(3, 8))

        def of(q: Any, u: Any, v: Any, added: Any, height: Any) -> Any:
            out = budget(
                q,
                u,
                v,
                grid,
                volume,
                time=TIMES,
                source=added,
                thickness=1.0 + 0.0 * height,
                tendency=tendency,
            )
            return out.tendency, out.source

        # relative to a tendency that is a difference of contents
        assert_backends_agree(of, q, u, v, added, height, rtol=1e-11, rtol32=1e-4)

    @needs_jax
    @pytest.mark.slow
    def test_jit_and_gradients(self) -> None:
        import jax
        import jax.numpy as jnp

        grid = GRIDS["uneven plane"]
        q, u, v, added, height = run(6, grid)
        volume = ControlVolume.from_region(region(6, grid), grid)

        def of(q: Any, time: Any) -> Budget:
            return budget(
                q, u, v, grid, volume, time=time, source=added, thickness=height
            )

        out = jax.jit(of)(q, jnp.asarray(TIMES))
        assert isinstance(out, Budget)
        for value, want in zip(terms(out), terms(of(q, TIMES)), strict=True):
            assert_matches_oracle(value, want, rtol=1e-9, atol=1e-9)
        assert_grad_matches_fd(
            lambda q: jnp.stack(terms(of(q, TIMES))), q[:2].repeat(3, axis=0)[:5]
        )
        assert_traces(lambda q: terms(of(q, TIMES)), np.stack([q, 2.0 * q]), rtol=1e-9)

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(7, grid), grid)
        q, u, v, added, height = run(7, grid)
        assert_refuses_dataarrays(
            lambda q, u, v, s, h, t: budget(
                q, u, v, grid, volume, time=t, source=s, thickness=h
            ),
            q,
            u,
            v,
            added,
            height,
            TIMES,
        )

    def test_another_tendency(self) -> None:
        grid = GRIDS["plane"]
        volume = ControlVolume.from_region(region(8, grid), grid)
        tendency: Any = "backward"
        with pytest.raises(ValidationError, match="must be 'centred' or 'forward'"):
            budget(*run(8, grid)[:3], grid, volume, time=TIMES, tendency=tendency)

    @pytest.mark.parametrize("time", [TIMES[:-1], TIMES[:, None], 1.0])
    def test_times_that_are_not_the_first_axis(self, time: Any) -> None:
        with pytest.raises(ValidationError, match="'time' must be 1-D, the first"):
            cell_residual(*run(9, GRIDS["plane"])[:3], GRIDS["plane"], time=time)

    def test_one_time_or_none(self) -> None:
        grid = GRIDS["plane"]
        q, u, v = fields(9, grid, 1)
        with pytest.raises(ValidationError, match="at least two"):
            cell_residual(q, u, v, grid, time=[0.0])
        with pytest.raises(ValidationError, match="'time' must be 1-D"):
            cell_residual(q[0], u[0], v[0], grid, time=[0.0] * 8)


class TestCellResidual:
    @pytest.mark.slow
    @pytest.mark.parametrize("tendency", ["centred", "forward"])
    def test_backends_and_precision(self, tendency: Any) -> None:
        grid = GRIDS["torus"]
        q, u, v, added, _ = run(0, grid)
        q = q + 1e3 * np.arange(TIMES.size)[:, None, None]
        assert_backends_agree(
            lambda q, u, v, s: cell_residual(
                q, 1e-3 * u, 1e-3 * v, grid, time=TIMES, source=s, tendency=tendency
            ),
            q,
            u,
            v,
            added,
            rtol=1e-10,
            rtol32=1e-3,
        )

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        grid = GRIDS["torus"]
        q, u, v, added, _ = run(1, grid)

        def of(q: Any, s: Any) -> Any:
            return cell_residual(q, u, v, grid, time=TIMES, source=s)

        assert_traces(of, np.stack([q, 2.0 * q]), np.stack([added] * 2), rtol=1e-9)
        assert_grad_matches_fd(lambda s: of(q, s)[2], added)

    def test_refuses_dataarrays(self) -> None:
        grid = GRIDS["plane"]
        q, u, v, added, _ = run(2, grid)
        assert_refuses_dataarrays(
            lambda q, u, v, s: cell_residual(q, u, v, grid, time=TIMES, source=s),
            q,
            u,
            v,
            added,
        )
