"""Diagnostics of a horizontal velocity on collocated grids."""

from __future__ import annotations

import dataclasses
from typing import Any

import gridax
import numpy as np
import pytest
from _backends import (
    GRID_KINDS,
    HAS_JAX,
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
    Motion,
    metpy_advection,
    metpy_frontogenesis,
    pure_strain,
    solid_body_plane,
    solid_body_sphere,
    xrops_values,
)
from geophysix.earth import SPHERE as EARTH

from diagnostix import ValidationError
from diagnostix.kinematics import (
    VelocityGradient,
    advection,
    ageostrophic_velocity,
    curvature_vorticity,
    frontogenesis,
    geostrophic_residual,
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
    @pytest.mark.slow
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
    @pytest.mark.slow
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


def xrops_grid(stored: dict[str, np.ndarray], name: str) -> Any:
    """The grid of the stored xrops values: ``plane`` or ``sphere``."""
    if name == "plane":
        return gridax.CartesianAGrid2D.from_coords(
            y=stored["plane_y"], x=stored["plane_x"]
        )
    return gridax.SphericalAGrid2D.from_coords(
        lat=stored["sphere_lat"], lon=stored["sphere_lon"], R=EARTH.semi_major
    )


def assert_close_to_scale(out: Any, expected: np.ndarray, atol: float) -> None:
    """Agreement relative to the largest value: some points are near zero."""
    scale = np.abs(expected).max()
    assert_matches_oracle(out / scale, expected / scale, rtol=0.0, atol=atol)


class TestAdvection:
    def test_a_linear_scalar_is_exact(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        u, v = flow(0, *PLANE.shape)
        out = advection(2.0 * x - 3.0 * y, u, v, PLANE)
        assert_matches_oracle(out, -(2.0 * u - 3.0 * v), rtol=0.0, atol=1e-11)

    @pytest.mark.parametrize("name", ["plane", "sphere"])
    def test_stored_xrops_values(self, name: str) -> None:
        stored = xrops_values("kinematics")
        out = advection(
            stored["plane_scalar"],
            stored["plane_u"],
            stored["plane_v"],
            xrops_grid(stored, name),
        )
        assert_close_to_scale(out, stored[f"{name}_advection"], 1e-12)

    @pytest.mark.oracle
    def test_metpy(self) -> None:
        stored = xrops_values("kinematics")
        fields = [stored[f"plane_{name}"] for name in ("scalar", "u", "v")]
        dx, dy = (np.diff(stored[f"plane_{name}"])[0] for name in "xy")
        expected = metpy_advection(*fields, dx, dy)[INNER]
        out = advection(*fields, xrops_grid(stored, "plane"))[INNER]
        assert_close_to_scale(out, expected, 1e-12)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_dry_cells_are_nan_and_never_read(self, name: str) -> None:
        check_dry_cells(GRIDS[name], lambda c, u, v, **kw: (advection(c, u, v, **kw),))

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        assert_backends_agree(
            lambda c, u, v: advection(c, u, v, grid),
            3.0 * x + 4.0 * y,
            *flow(1, *grid.shape),
            kinds=GRID_KINDS,
            rtol=1e-11,
            rtol32=2e-5,
        )

    def test_refuses_dataarrays(self) -> None:
        u, v = flow(2, *PLANE.shape)
        assert_refuses_dataarrays(lambda c, u, v: advection(c, u, v, PLANE), u, u, v)

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        u, v = flow(3, 3, *SPHERE.shape)
        assert_traces(lambda c, u, v: advection(c, u, v, SPHERE), u, u, v, rtol=1e-10)
        for position in range(3):
            check_gradient(advection, (u[1], u[0], v[0]), position, SPHERE)

    def test_a_scalar_of_another_shape(self) -> None:
        with pytest.raises(ValidationError, match=r"got shape \(7, 6\)"):
            advection(np.zeros((7, 6)), *flow(4, *PLANE.shape), PLANE)


class TestFrontogenesis:
    def test_pure_strain(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        motion = pure_strain(x, y, 0.4)
        # F = α |∂ᵧc| and −α |∂ₓc|
        for scalar, expected in ((-3.0 * y, 0.4 * 3.0), (2.0 * x, -0.4 * 2.0)):
            out = frontogenesis(scalar, motion.u, motion.v, PLANE)
            assert_matches_oracle(out, np.full(PLANE.shape, expected), rtol=1e-11)

    def test_rotation_and_uniform_flow_on_a_plane(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        scalar = np.sin(x) + np.cos(0.5 * y)
        motion = solid_body_plane(x, y, 0.3)
        for u, v in ((motion.u, motion.v), (0.0 * x + 2.0, 0.0 * x - 1.0)):
            out = frontogenesis(scalar, u, v, PLANE)
            assert_matches_oracle(out, np.zeros(PLANE.shape), rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("axis", sorted(AXES))
    def test_rotation_on_the_sphere(self, axis: str) -> None:
        worst = []
        for step in (4.0, 2.0, 1.0):
            grid, motion = rotation(step, axis)
            lam, phi = np.meshgrid(grid.coords("x"), grid.coords("y"))
            scalar = np.cos(phi) * np.sin(2.0 * lam) + np.sin(phi)
            out = frontogenesis(scalar, motion.u, motion.v, grid)
            norm = np.hypot(*gridax.gradient(scalar, grid))
            worst.append(np.abs(out[INNER]).max() / norm.max() / OMEGA)
        # second order, and at most the strain of the test above
        assert worst[-1] < 2e-4
        assert_matches_oracle(
            np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.4
        )

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_petterssens_form(self, name: str) -> None:
        grid = GRIDS[name]
        u, v = flow(0, *grid.shape)
        scalar = np.random.default_rng(1).normal(size=grid.shape)
        c_x, c_y = gridax.gradient(scalar, grid)
        a = velocity_gradient(u, v, grid)
        stretching, shearing = a.dudx - a.dvdy, a.dvdx + a.dudy
        deformation = np.hypot(stretching, shearing)
        # ψ, the angle of the axis of dilatation, and β, from the isolines to it
        psi = 0.5 * np.arctan2(shearing, stretching)
        beta = psi - np.arctan2(c_x, -c_y)
        expected = np.hypot(c_x, c_y) / 2.0
        expected = expected * (deformation * np.cos(2.0 * beta) - a.dudx - a.dvdy)
        out = frontogenesis(scalar, u, v, grid)
        assert_close_to_scale(out, expected, 1e-13)

    def test_stored_xrops_values(self) -> None:
        stored = xrops_values("kinematics")
        out = frontogenesis(
            stored["plane_scalar"],
            stored["plane_u"],
            stored["plane_v"],
            xrops_grid(stored, "plane"),
        )
        assert_close_to_scale(out, stored["plane_frontogenesis"], 1e-12)

    @pytest.mark.oracle
    def test_metpy(self) -> None:
        stored = xrops_values("kinematics")
        fields = [stored[f"plane_{name}"] for name in ("scalar", "u", "v")]
        dx, dy = (np.diff(stored[f"plane_{name}"])[0] for name in "xy")
        expected = metpy_frontogenesis(*fields, dx, dy)[INNER]
        out = frontogenesis(*fields, xrops_grid(stored, "plane"))[INNER]
        assert_close_to_scale(out, expected, 1e-12)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_a_uniform_scalar_is_zero(self, name: str) -> None:
        grid = GRIDS[name]
        u, v = flow(2, *grid.shape)
        out = frontogenesis(np.zeros(grid.shape), u, v, grid)
        assert_matches_oracle(out, np.zeros(grid.shape), rtol=0.0)
        # the uneven stencil of a constant is its rounding
        out = frontogenesis(np.full(grid.shape, 2.0), u, v, grid)
        assert_matches_oracle(out, np.zeros(grid.shape), rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    @pytest.mark.slow
    def test_dry_cells_are_nan_and_never_read(self, name: str) -> None:
        check_dry_cells(
            GRIDS[name], lambda c, u, v, **kw: (frontogenesis(c, u, v, **kw),)
        )

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_backends_and_precision(self, name: str) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        # every term of one sign, for float32
        assert_backends_agree(
            lambda c, u, v: frontogenesis(c, u, v, grid),
            3.0 * x + 4.0 * y,
            2.0 * x + y - 1e3,
            x + 3.0 * y + 1e3,
            kinds=GRID_KINDS,
            rtol=1e-10,
            rtol32=5e-3,
        )

    def test_refuses_dataarrays(self) -> None:
        u, v = flow(3, *PLANE.shape)
        assert_refuses_dataarrays(
            lambda c, u, v: frontogenesis(c, u, v, PLANE), u, u, v
        )

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        u, v = flow(4, 3, *SPHERE.shape)
        assert_traces(
            lambda c, u, v: frontogenesis(c, u, v, SPHERE), u, u, v, rtol=1e-10
        )
        for position in range(3):
            check_gradient(frontogenesis, (u[1], u[0], v[0]), position, SPHERE)

    @needs_jax
    def test_gradients_where_the_scalar_is_uniform(self) -> None:
        u, v = flow(5, *PLANE.shape)
        assert_finite_gradient(
            lambda c: frontogenesis(c, u, v, PLANE), np.ones(PLANE.shape)
        )


class TestGeostrophy:
    def test_a_linear_height_on_a_plane(self) -> None:
        x, y = np.meshgrid(PLANE.coords("x"), PLANE.coords("y"))
        height = 0.3 * x - 0.5 * y
        u, v = flow(0, *PLANE.shape)
        # u_g = g/f · 0.5, v_g = g/f · 0.3
        u_g, v_g = 9.0 / 2.0 * 0.5, 9.0 / 2.0 * 0.3
        out = ageostrophic_velocity(u, v, height, PLANE, f=2.0, gravity=9.0)
        for value, expected in zip(out, (u - u_g, v - v_g), strict=True):
            assert_matches_oracle(value, expected, rtol=0.0, atol=1e-11)
        out = geostrophic_residual(u, v, height, PLANE, f=2.0, gravity=9.0)
        for value, expected in zip(out, (u - u_g, v - v_g), strict=True):
            assert_matches_oracle(value, 2.0 * expected, rtol=0.0, atol=1e-11)

    def test_a_geostrophic_flow_on_the_sphere(self) -> None:
        residual, rest = [], []
        for step in (4.0, 2.0, 1.0):
            lat = np.arange(-60.0, 60.0 + step / 2, step)
            grid = gridax.SphericalAGrid2D.from_coords(
                lat=lat, lon=np.arange(0.0, 90.0, step), R=2.0
            )
            lam, phi = np.meshgrid(grid.coords("x"), grid.coords("y"))
            f = 3.0 * np.sin(phi)
            # η = sin² φ (1 + cos φ sin 2λ), whose flow is finite at the equator
            wave = np.cos(phi) * np.sin(2.0 * lam)
            height = np.sin(phi) ** 2 * (1.0 + wave)
            slope_x = 2.0 * np.sin(phi) ** 2 * np.cos(2.0 * lam) / 2.0
            slope_y = np.sin(2.0 * phi) * (1.0 + wave) / 2.0
            slope_y = slope_y - np.sin(phi) ** 3 * np.sin(2.0 * lam) / 2.0
            # u = −(g/f) ∂ᵧη and v = (g/f) ∂ₓη, with sin φ divided out
            sine = np.where(phi == 0.0, 1.0, np.sin(phi))
            by_x = np.where(phi == 0.0, 0.0, slope_x / sine)
            by_y = np.where(phi == 0.0, 1.0 + wave, slope_y / sine)
            u, v = -9.0 / 3.0 * by_y, 9.0 / 3.0 * by_x
            out = geostrophic_residual(u, v, height, grid, f=f, gravity=9.0)
            residual.append(max(np.abs(value[INNER]).max() for value in out))
            out = ageostrophic_velocity(u, v, height, grid, f=f, gravity=9.0)
            off = np.abs(lat[1:-1]) > 10.0
            rest.append(max(np.abs(value[INNER][off]).max() for value in out))
        # the flow is of order g/f = 3 m s⁻¹, on a sphere of 2 m
        for worst in (residual, rest):
            assert worst[-1] < 3e-3
            assert_matches_oracle(
                np.divide(worst[:-1], worst[1:]), np.full(2, 4.0), rtol=0.0, atol=0.4
            )

    @pytest.mark.parametrize("name", sorted(GRIDS))
    def test_the_two_are_one_balance(self, name: str) -> None:
        grid = GRIDS[name]
        u, v = flow(1, *grid.shape)
        height, f = flow(2, *grid.shape)
        residual = geostrophic_residual(u, v, height, grid, f=f)
        rest = ageostrophic_velocity(u, v, height, grid, f=f)
        for value, expected in zip(residual, rest, strict=True):
            assert_matches_oracle(value / f, expected, rtol=1e-11)

    def test_no_coriolis(self) -> None:
        u, v = flow(3, *SPHERE.shape)
        f = np.sin(SPHERE.coords("y"))[:, None]
        f[2] = 0.0
        for value in ageostrophic_velocity(u, v, u, SPHERE, f=f):
            assert (
                np.isnan(value).any(axis=-1).tolist()
                == [False] * 2 + [True] + [False] * 3
            )
        for value in geostrophic_residual(u, v, u, SPHERE, f=f):
            assert np.isfinite(value).all()

    @pytest.mark.parametrize("name", ["plane", "sphere"])
    def test_stored_xrops_values(self, name: str) -> None:
        stored = xrops_values("kinematics")
        f = stored[f"{name}_coriolis"]
        out = geostrophic_residual(
            stored["plane_u"],
            stored["plane_v"],
            stored["plane_height"],
            xrops_grid(stored, name),
            f=f[:, None] if f.ndim else f,
            gravity=float(stored["gravity"]),
        )
        for value, part in zip(out, "uv", strict=True):
            assert_close_to_scale(value, stored[f"{name}_residual_{part}"], 1e-12)

    @pytest.mark.parametrize("name", sorted(GRIDS))
    @pytest.mark.parametrize("fn", [ageostrophic_velocity, geostrophic_residual])
    def test_dry_cells_are_nan_and_never_read(self, name: str, fn: Any) -> None:
        check_dry_cells(GRIDS[name], lambda eta, u, v, **kw: fn(u, v, eta, f=2.0, **kw))

    @pytest.mark.parametrize("name", sorted(GRIDS))
    @pytest.mark.parametrize("fn", [ageostrophic_velocity, geostrophic_residual])
    def test_backends_and_precision(self, name: str, fn: Any) -> None:
        grid = GRIDS[name]
        x, y = np.meshgrid(grid.coords("x"), grid.coords("y"))
        u, v = flow(4, *grid.shape)
        f = 2.0 + np.sin(y)
        # η falls with y and rises with x, so nothing cancels in float32
        height = 10.0 * x - 10.0 * y if fn is geostrophic_residual else 10.0 * (y - x)
        sign = -1.0 if fn is geostrophic_residual else 1.0
        assert_backends_agree(
            lambda u, v, eta, f: fn(u, v, eta, grid, f=f),
            sign * u,
            sign * -v,
            height,
            f,
            kinds=GRID_KINDS,
            rtol=1e-11,
            rtol32=1e-4,
        )

    @pytest.mark.parametrize("fn", [ageostrophic_velocity, geostrophic_residual])
    def test_the_coriolis_parameter_broadcasts(self, fn: Any) -> None:
        u, v = flow(5, 2, *SPHERE.shape)
        row = np.sin(SPHERE.coords("y"))[:, None]
        wide = np.broadcast_to(row, u.shape)
        for out, expected in zip(
            fn(u, v, u, SPHERE, f=row), fn(u, v, u, SPHERE, f=wide), strict=True
        ):
            assert_matches_oracle(out, expected, rtol=0.0)

    @pytest.mark.parametrize("fn", [ageostrophic_velocity, geostrophic_residual])
    def test_refuses_dataarrays(self, fn: Any) -> None:
        u, v = flow(6, *PLANE.shape)
        assert_refuses_dataarrays(
            lambda u, v, eta, f: fn(u, v, eta, PLANE, f=f), u, v, u, v
        )

    @needs_jax
    @pytest.mark.parametrize("fn", [ageostrophic_velocity, geostrophic_residual])
    @pytest.mark.slow
    def test_transforms_and_gradients(self, fn: Any) -> None:
        import jax.numpy as jnp

        u, v = flow(7, 3, *SPHERE.shape)
        height, f = flow(8, 3, *SPHERE.shape)
        assert_traces(
            lambda u, v, eta, f: fn(u, v, eta, SPHERE, f=f), u, v, height, f, rtol=1e-10
        )

        def stacked(u: Any, v: Any, eta: Any, f: Any, grid: Any) -> Any:
            return jnp.stack(fn(u, v, eta, grid, f=f))

        for position in range(4):
            args = (u[0], v[0], height[0], f[0])
            check_gradient(stacked, args, position, SPHERE)

    @needs_jax
    def test_gradients_where_there_is_no_coriolis(self) -> None:
        import jax.numpy as jnp

        u, v = flow(9, *SPHERE.shape)
        f = np.sin(SPHERE.coords("y"))[:, None] * np.ones(SPHERE.shape)
        f[2] = 0.0

        def off_the_equator(eta: Any) -> Any:
            out = ageostrophic_velocity(u, v, eta, SPHERE, f=f)
            return jnp.where(f == 0.0, 0.0, jnp.stack(out))

        assert_finite_gradient(off_the_equator, v)


def check_dry_cells(grid: Any, fn: Any) -> None:
    """Dry cells are NaN, never read, and leave the other gradients finite."""
    wet, mask = coast(grid)
    fields = [*flow(20, *grid.shape), flow(21, *grid.shape)[0]]
    outs = [
        fn(*(np.where(wet, field, dry) for field in fields), grid=grid, mask=mask)
        for dry in (0.0, 1e3, np.nan)
    ]
    for out in outs:
        for value, first in zip(out, outs[0], strict=True):
            assert_matches_oracle(value, first, rtol=0.0)
            assert (np.isnan(value) == ~wet).all()
    if HAS_JAX:
        import jax.numpy as jnp

        dry = [np.where(wet, field, np.nan) for field in fields]
        for position in range(3):

            def wet_only(x: Any, position: int = position) -> Any:
                args = [*dry[:position], x, *dry[position + 1 :]]
                out = fn(*args, grid=grid, mask=mask)
                return jnp.where(wet, jnp.stack(out), 0.0)

            assert_finite_gradient(wet_only, dry[position], where=wet)


def check_gradient(fn: Any, args: tuple[Any, ...], position: int, grid: Any) -> None:
    """The derivative of ``fn(*args, grid)`` by one argument, against differences."""

    def of_one(x: Any) -> Any:
        return fn(*args[:position], x, *args[position + 1 :], grid)

    assert_grad_matches_fd(of_one, args[position], rtol=1e-5)


def calm_at(field: np.ndarray, j: int, i: int) -> np.ndarray:
    """``field`` with one point at rest."""
    out = field.copy()
    out[j, i] = 0.0
    return out
