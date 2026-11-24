"""Diagnostics of a horizontal velocity on collocated grids."""

from __future__ import annotations

import dataclasses
from typing import Any

import gridax
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
from _oracles import Motion, pure_strain, solid_body_sphere, xrops_values

from diagnostix import ValidationError
from diagnostix.kinematics import (
    VelocityGradient,
    curvature_vorticity,
    shear_vorticity,
    velocity_gradient,
)


RADIUS = 6.0e6
OMEGA = 1.0e-5
PLANE = gridax.CartesianAGrid2D.from_coords(
    y=[0.0, 1.0, 3.0, 4.5, 5.0, 7.0], x=[0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 6.0]
)
SPHERE = gridax.SphericalAGrid2D.from_coords(
    lat=[-70.0, -35.0, -10.0, 15.0, 40.0, 65.0],
    lon=[0.0, 8.0, 20.0, 30.0, 45.0, 70.0, 80.0],
    R=2.0,
)
GRIDS = {"plane": PLANE, "sphere": SPHERE}
#: Off the edges, where the differences are centred.
INNER = (..., slice(1, -1), slice(1, -1))
#: Rotation vectors: about the poles, and tilted by 0.65 rad.
AXES = {
    "polar": np.asarray([0.0, 0.0, OMEGA]),
    "tilted": OMEGA
    * np.asarray([np.sin(0.65) * 0.8, np.sin(0.65) * 0.6, np.cos(0.65)]),
}
FOUR = ("dudx", "dudy", "dvdx", "dvdy")


def flow(seed: int, *shape: int) -> tuple[np.ndarray, np.ndarray]:
    """Two positive fields of order 1, so that nothing is calm by accident."""
    rng = np.random.default_rng(seed)
    u, v = 1.0 + rng.random((2, *shape))
    return u, v


def four(gradient: VelocityGradient) -> tuple[Any, ...]:
    return tuple(getattr(gradient, name) for name in FOUR)


def coast(grid: Any) -> tuple[np.ndarray, Any]:
    """A wet region with two dry cells, and its gridax mask."""
    wet = np.ones(grid.shape, dtype=bool)
    wet[2:4, 2] = False
    return wet, gridax.Mask2D.from_grid(grid, wet)


def rotation(step: float, axis: str) -> tuple[Any, Motion]:
    """Solid-body rotation between 70° S and 70° N on a ``step``-degree grid."""
    lat = np.arange(-70.0, 70.0 + step / 2, step)
    lon = np.arange(0.0, 90.0, step)
    grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon, R=RADIUS)
    motion = solid_body_sphere(lon, lat[:, None], AXES[axis], RADIUS)
    return grid, motion


class TestVelocityGradient:
    def test_a_linear_flow_is_exact(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        out = velocity_gradient(0.3 * x + 0.7 * y, 0.2 * x - 0.5 * y, PLANE)
        for value, expected in zip(four(out), (0.3, 0.7, 0.2, -0.5), strict=True):
            assert_matches_oracle(value, np.full(PLANE.shape, expected), rtol=1e-12)

    def test_pure_strain(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        motion = pure_strain(x, y, 0.4)
        out = velocity_gradient(motion.u, motion.v, PLANE)
        assert_matches_oracle(
            out.dudx - out.dvdy, np.full(PLANE.shape, 0.8), rtol=1e-12
        )
        for zero in (out.dvdx + out.dudy, out.dudx + out.dvdy):
            assert_matches_oracle(zero, np.zeros(PLANE.shape), rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("axis", sorted(AXES))
    def test_rotation_on_the_sphere(self, axis: str) -> None:
        strain, spin = [], []
        for step in (4.0, 2.0, 1.0):
            grid, motion = rotation(step, axis)
            out = velocity_gradient(motion.u, motion.v, grid)
            strain.append(
                max(
                    np.abs((out.dvdx + out.dudy)[INNER]).max(),
                    np.abs((out.dudx - out.dvdy)[INNER]).max(),
                )
                / OMEGA
            )
            error = out.dvdx - out.dudy - motion.vorticity
            spin.append(np.abs(error[INNER]).max() / OMEGA)
        # second order, and 1.1e-4 ω at 1° in the worse case
        assert strain[-1] < 2e-4
        assert spin[-1] < 2e-4
        for worst in (strain, spin):
            assert_matches_oracle(
                np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.4
            )

    @pytest.mark.parametrize("axis", sorted(AXES))
    def test_vorticity_against_gridax(self, axis: str) -> None:
        worst = []
        for step in (4.0, 2.0, 1.0):
            grid, motion = rotation(step, axis)
            out = velocity_gradient(motion.u, motion.v, grid)
            # gridax differentiates u cos φ: the same at truncation order
            error = out.dvdx - out.dudy - gridax.curl(motion.u, motion.v, grid)
            worst.append(np.abs(error[INNER]).max())
        assert_matches_oracle(
            np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.4
        )

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_the_derivatives_are_gridax(self, name: str) -> None:
        grid = GRIDS[name]
        u, v = flow(0, *grid.shape)
        out = velocity_gradient(u, v, grid)
        for value, field in ((out.dudy, u), (out.dvdy, v)):
            expected = gridax.difference(field, grid, axis="y")
            assert_matches_oracle(value, expected, rtol=0.0)
        turn = np.tan(grid.coords("y"))[:, None] / 2.0 if name == "sphere" else 0.0
        for value, field, metric in ((out.dudx, u, -v), (out.dvdx, v, u)):
            expected = gridax.difference(field, grid, axis="x") + metric * turn
            assert_matches_oracle(value, expected, rtol=1e-12)

    def test_a_row_at_a_pole(self) -> None:
        grid = gridax.SphericalAGrid2D.from_coords(
            lat=[-90.0, -45.0, 0.0, 90.0], lon=[0.0, 10.0, 30.0]
        )
        out = velocity_gradient(*flow(1, *grid.shape), grid)
        for value, at_pole in zip(four(out), (True, False, True, False), strict=True):
            expected = [at_pole, False, False, at_pole]
            assert np.isnan(value).any(axis=-1).tolist() == expected

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_dry_cells_are_nan_and_never_read(self, name: str) -> None:
        grid = GRIDS[name]
        wet, mask = coast(grid)
        u, v = flow(2, *grid.shape)
        outs = [
            velocity_gradient(
                np.where(wet, u, dry), np.where(wet, v, dry), grid, mask=mask
            )
            for dry in (0.0, 1e3, np.nan)
        ]
        for out in outs:
            for value, first in zip(four(out), four(outs[0]), strict=True):
                assert_matches_oracle(value, first, rtol=0.0)
                assert (np.isnan(value) == ~wet).all()

    def test_leading_axes_broadcast(self) -> None:
        u, v = flow(3, 2, *SPHERE.shape)
        out = velocity_gradient(u, v[0], SPHERE)
        for index in range(2):
            one = velocity_gradient(u[index], v[0], SPHERE)
            for value, expected in zip(four(out), four(one), strict=True):
                assert_matches_oracle(value[index], expected, rtol=1e-14)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        # no cancellation, so that float32 keeps its relative precision
        u, v = 3.0 * x + 4.0 * y, 5.0 * x + 2.0 * y
        _, mask = coast(grid)
        for options in ({}, {"mask": mask}):
            assert_backends_agree(
                lambda u, v, options=options: four(
                    velocity_gradient(u, v, grid, **options)
                ),
                u,
                v,
                kinds=GRID_KINDS,
                rtol=1e-11,
                rtol32=2e-5,
            )

    def test_refuses_dataarrays(self) -> None:
        u, v = flow(4, *PLANE.shape)
        assert_refuses_dataarrays(lambda u, v: velocity_gradient(u, v, PLANE), u, v)

    @needs_jax
    def test_transforms(self) -> None:
        u, v = flow(5, 3, *SPHERE.shape)
        assert_traces(
            lambda u, v: four(velocity_gradient(u, v, SPHERE)), u, v, rtol=1e-11
        )

    @needs_jax
    def test_the_record_and_the_grid_are_pytrees(self) -> None:
        import jax

        u, v = flow(6, *SPHERE.shape)
        out = jax.jit(velocity_gradient)(u, v, SPHERE)
        assert isinstance(out, VelocityGradient)
        expected = velocity_gradient(u, v, SPHERE)
        for value, want in zip(four(out), four(expected), strict=True):
            assert_matches_oracle(value, want, rtol=1e-11)

    @needs_jax
    def test_gradients(self) -> None:
        import jax.numpy as jnp

        u, v = flow(7, *SPHERE.shape)
        assert_grad_matches_fd(
            lambda x: jnp.stack(four(velocity_gradient(x, v, SPHERE))), u
        )

    def test_a_grid_that_is_not_collocated(self) -> None:
        grid: Any = gridax.CartesianCGrid2D.from_interior(4, 4, 1.0, 1.0)
        with pytest.raises(ValidationError, match="must be a CartesianAGrid2D or"):
            velocity_gradient(np.zeros(grid.shape), np.zeros(grid.shape), grid)

    def test_a_field_of_another_shape(self) -> None:
        with pytest.raises(ValidationError, match=r"ends in \(6, 7\) \(y, x\)"):
            velocity_gradient(np.zeros((6, 7)), np.zeros((7, 6)), PLANE)
        with pytest.raises(ValidationError, match=r"got shape \(\)"):
            velocity_gradient(0.0, np.zeros((6, 7)), PLANE)

    def test_the_record_is_frozen(self) -> None:
        out = velocity_gradient(*flow(8, *PLANE.shape), PLANE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            out.dudx = 0.0  # type: ignore[misc]


PARTS = {"shear": shear_vorticity, "curvature": curvature_vorticity}


class TestShearAndCurvatureVorticity:
    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_the_parts_add_up(self, name: str) -> None:
        grid = GRIDS[name]
        u, v = flow(0, 4, *grid.shape)
        out = velocity_gradient(u, v, grid)
        zeta = out.dvdx - out.dudy
        total = shear_vorticity(u, v, grid) + curvature_vorticity(u, v, grid)
        # relative to the terms that are added
        scale = np.abs(out.dudx) + np.abs(out.dvdx) + np.abs(out.dudy)
        scale = scale + np.abs(out.dvdy)
        assert_matches_oracle(total / scale, zeta / scale, rtol=0.0, atol=1e-14)

    def test_rotation_on_a_plane_is_half_and_half(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x") + 1.0, PLANE.coords("y") + 1.0)
        for part in PARTS.values():
            out = part(-0.3 * y, 0.3 * x, PLANE)
            assert_matches_oracle(out, np.full(PLANE.shape, 0.3), rtol=1e-12)

    @pytest.mark.parametrize("part", sorted(PARTS))
    def test_stored_xrops_values(self, part: str) -> None:
        stored = xrops_values("kinematics")
        grid = gridax.CartesianAGrid2D.from_coords(
            y=stored["plane_y"], x=stored["plane_x"]
        )
        u, v = stored["plane_u"], stored["plane_v"]
        calm = calm_at(u, 5, 7), calm_at(v, 5, 7)
        # relative to the size of the field: some points are near zero
        for name, velocity in (("plane", (u, v)), ("plane_calm", calm)):
            out = PARTS[part](*velocity, grid)
            expected = stored[f"{name}_{part}_vorticity"]
            scale = np.abs(expected).max()
            assert_matches_oracle(out / scale, expected / scale, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("part", sorted(PARTS))
    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_a_calm_point_is_zero(self, name: str, part: str) -> None:
        grid = GRIDS[name]
        u, v = (calm_at(field, 2, 3) for field in flow(1, *grid.shape))
        out = PARTS[part](u, v, grid)
        assert out[2, 3] == 0.0
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("part", sorted(PARTS))
    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_a_masked_cell(self, name: str, part: str) -> None:
        grid = GRIDS[name]
        wet, mask = coast(grid)
        u, v = flow(2, *grid.shape)
        # at rest and NaN on land
        for dry in (0.0, np.nan):
            out = PARTS[part](
                np.where(wet, u, dry), np.where(wet, v, dry), grid, mask=mask
            )
            assert (np.isnan(out) == ~wet).all()

    @pytest.mark.parametrize("part", sorted(PARTS))
    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_backends_and_precision(self, name: str, part: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        # ζ_s and ζ_c of one sign in every term, for float32
        u, v = -(1.0 + y), 1.0 + x
        assert_backends_agree(
            lambda u, v: PARTS[part](u, v, grid),
            u,
            v,
            kinds=GRID_KINDS,
            rtol=1e-10,
            rtol32=1e-4,
        )

    @pytest.mark.parametrize("part", sorted(PARTS))
    def test_refuses_dataarrays(self, part: str) -> None:
        u, v = flow(3, *PLANE.shape)
        assert_refuses_dataarrays(lambda u, v: PARTS[part](u, v, PLANE), u, v)

    @needs_jax
    @pytest.mark.parametrize("part", sorted(PARTS))
    def test_transforms(self, part: str) -> None:
        u, v = flow(4, 3, *SPHERE.shape)
        assert_traces(lambda u, v: PARTS[part](u, v, SPHERE), u, v, rtol=1e-10)

    @needs_jax
    @pytest.mark.parametrize("part", sorted(PARTS))
    def test_gradients(self, part: str) -> None:
        u, v = flow(5, *SPHERE.shape)
        assert_grad_matches_fd(lambda u: PARTS[part](u, v, SPHERE), u, rtol=1e-5)
        assert_grad_matches_fd(lambda v: PARTS[part](u, v, SPHERE), v, rtol=1e-5)

    @needs_jax
    @pytest.mark.parametrize("part", sorted(PARTS))
    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_gradients_are_finite(self, name: str, part: str) -> None:
        import jax.numpy as jnp

        grid = GRIDS[name]
        wet, mask = coast(grid)
        u, v = (calm_at(field, 1, 5) for field in flow(6, *grid.shape))
        u, v = np.where(wet, u, np.nan), np.where(wet, v, np.nan)

        def wet_only(u: Any) -> Any:
            return jnp.where(wet, PARTS[part](u, v, grid, mask=mask), 0.0)

        assert_finite_gradient(wet_only, u, where=wet)


def calm_at(field: np.ndarray, j: int, i: int) -> np.ndarray:
    """``field`` with one point at rest."""
    out = field.copy()
    out[j, i] = 0.0
    return out
