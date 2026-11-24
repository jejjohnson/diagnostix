"""Each check passes on a right function and fails on a wrong one, and each
reference is what its name says."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import gridax
import numpy as np
import pytest
from _backends import (
    GRID_KINDS,
    HAS_STRICT,
    KINDS,
    assert_backends_agree,
    assert_refuses_dataarrays,
    needs_jax,
    needs_xarray,
)
from _harness import (
    assert_adjoint,
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import (
    XROPS_COMMIT,
    XROPS_FAMILIES,
    functax_eofs,
    linear_pycnocline,
    metpy_advection,
    metpy_frontogenesis,
    neutral_layer_under_inversion,
    pure_strain,
    reliable_ensemble,
    sinusoid,
    sklearn_pca,
    solid_body_plane,
    solid_body_sphere,
    two_patterns,
    xrft_power_spectrum,
    xrops_values,
    xskillscore_crps,
)
from _sampling import assert_mean
from geophysix.earth import SPHERE
from hypothesis import given, strategies as st
from pipekit_array.functions import elementwise, floating_inputs, like


SLOPE = 0.3


@elementwise(n_outputs=2)
def toy(values, offset=0.0) -> Any:
    """sin(0.3 x) + o, and x²: what an elementwise diagnostix function looks like."""
    xp, (values, offset) = floating_inputs(values, offset)
    slope = like(SLOPE, values, xp)
    return xp.sin(slope * values) + offset, values**2


def toy_mean(values, valid) -> Any:
    """The mean of the valid samples: what a function along an axis looks like."""
    xp, (values,) = floating_inputs(values, params=(valid,))
    valid = valid if hasattr(valid, "dtype") else xp.asarray(valid)
    kept = xp.where(valid, values, xp.zeros_like(values))
    count = xp.sum(xp.astype(valid, values.dtype), axis=-1)
    return xp.sum(kept, axis=-1) / count


X = np.linspace(0.1, 2.0, 6).reshape(2, 3)
VALID = np.asarray([[True, False, True], [True, True, False]])


class TestTheChecksOnAToyFunction:
    def test_oracle(self) -> None:
        assert_matches_oracle(toy(X)[0], np.sin(SLOPE * X), rtol=1e-15)

    @given(st.floats(-5.0, 5.0))
    def test_property(self, value: float) -> None:
        assert toy(value)[1] == toy(-value)[1]

    def test_gradient(self) -> None:
        assert_grad_matches_fd(lambda x: toy(x)[0], X)
        assert_finite_gradient(lambda x: toy(x)[0], X)

    def test_backends(self) -> None:
        assert_backends_agree(toy, X, np.asarray([1.0, 2.0, 3.0]), labelled=True)
        assert_traces(toy, X, np.asarray([1.0, 2.0]))

    def test_a_function_along_an_axis(self) -> None:
        assert_backends_agree(toy_mean, X, VALID)
        assert_traces(toy_mean, X, VALID)
        assert_refuses_dataarrays(toy_mean, X, VALID)
        assert_finite_gradient(lambda x: toy_mean(x, VALID), X)


class TestChecksFail:
    def test_oracle_on_a_wrong_value_names_the_worst_index(self) -> None:
        with pytest.raises(AssertionError, match=r"worst at index \(1,\)"):
            assert_matches_oracle([1.0, 2.5], [1.0, 2.0], rtol=1e-12)

    def test_oracle_matches_nan_only_with_nan(self) -> None:
        assert_matches_oracle([np.nan], [np.nan], rtol=0.0)
        for pair in (([np.nan], [1.0]), ([1.0], [np.nan])):
            with pytest.raises(AssertionError):
                assert_matches_oracle(*pair, rtol=1.0)

    def test_oracle_masks_and_compares_integers_exactly(self) -> None:
        assert_matches_oracle([1.0, 9.0], [1.0, 2.0], rtol=0.0, where=[True, False])
        with pytest.raises(AssertionError):
            assert_matches_oracle([1], [2], rtol=10.0)
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_matches_oracle([1.0], [[1.0]], rtol=0.0)

    def test_adjoint_of_the_wrong_matrix(self) -> None:
        h = np.random.default_rng(0).normal(size=(3, 4))
        assert_adjoint(lambda x: h @ x, lambda r: h.T @ r, 4, 3)
        with pytest.raises(AssertionError, match="dot-product"):
            assert_adjoint(lambda x: h @ x, lambda r: 1.1 * h.T @ r, 4, 3)
        with pytest.raises(AssertionError, match="shape"):
            assert_adjoint(lambda x: h @ x, lambda r: r, 4, 3)
        with pytest.raises(AssertionError, match="H x has shape"):
            assert_adjoint(lambda x: x, lambda r: h.T @ r, 4, 3)

    @needs_jax
    def test_gradient_of_a_function_that_cuts_it(self) -> None:
        import jax

        with pytest.raises(AssertionError):
            assert_grad_matches_fd(lambda x: jax.lax.stop_gradient(x) * x, X)

    @needs_jax
    def test_gradient_that_is_nan(self) -> None:
        import jax.numpy as jnp

        with pytest.raises(AssertionError):
            assert_grad_matches_fd(lambda x: jnp.sqrt(jnp.abs(x)), np.zeros(2))

    @needs_jax
    def test_finite_gradient_of_a_single_where(self) -> None:
        import jax.numpy as jnp

        values = np.asarray([0.0, 2.0])

        def single(x: Any) -> Any:
            return jnp.sum(jnp.where(x > 0.0, 1.0 / x, 0.0))

        def double(x: Any) -> Any:
            safe = jnp.where(x > 0.0, x, 1.0)
            return jnp.sum(jnp.where(x > 0.0, 1.0 / safe, 0.0))

        assert_finite_gradient(double, values)
        assert_finite_gradient(single, values, where=values > 0.0)
        with pytest.raises(AssertionError, match="of output 0 by input 0"):
            assert_finite_gradient(single, values)

    @needs_jax
    def test_transforms_of_an_impure_function(self) -> None:
        calls = itertools.count()
        with pytest.raises(AssertionError, match=r"jit\(fn\), output 0"):
            assert_traces(lambda x: x + next(calls), X)

    @needs_jax
    def test_transforms_need_one_batch_axis(self) -> None:
        with pytest.raises(AssertionError, match="same batch axis"):
            assert_traces(toy, X, np.ones(5))

    def test_backends_of_a_function_that_drops_the_kind(self) -> None:
        if len(KINDS) == 1:
            pytest.skip("only NumPy is installed")
        with pytest.raises(AssertionError, match="in must give"):
            assert_backends_agree(lambda x: np.sin(np.asarray(x)), X, labelled=True)

    def test_backends_of_a_function_that_promotes_float32(self) -> None:
        def promotes(x: Any) -> Any:
            xp, (x,) = floating_inputs(x)
            return x * xp.asarray(SLOPE, dtype=xp.float64)

        with pytest.raises(AssertionError, match="float32: float64 came back"):
            assert_backends_agree(promotes, X)

    @pytest.mark.skipif(not HAS_STRICT, reason="array_api_strict is not installed")
    def test_backends_of_a_function_that_is_wrong_on_one_library(self) -> None:
        def wrong(x: Any) -> Any:
            xp, (x,) = floating_inputs(x)
            return x + (0.0 if xp.__name__.endswith("numpy") else 1.0)

        with pytest.raises(AssertionError, match="strict float64"):
            assert_backends_agree(wrong, X)
        # the kinds of a grid function leave strict out
        assert_backends_agree(wrong, X, kinds=GRID_KINDS)

    def test_backends_compare_counts_exactly(self) -> None:
        def count(valid: Any) -> Any:
            xp, _ = floating_inputs(0.0, params=(valid,))
            return xp.sum(xp.astype(valid, xp.int32), axis=-1)

        assert_backends_agree(count, VALID)

    @needs_xarray
    def test_refusing_dataarrays_of_a_function_that_takes_them(self) -> None:
        with pytest.raises(AssertionError, match="argument 0 accepted"):
            assert_refuses_dataarrays(lambda x: x, X)

    @needs_xarray
    def test_the_labelled_kind_broadcasts_as_numpy_does(self) -> None:
        kind = next(kind for kind in KINDS if kind.name == "xarray")
        out = kind.make(np.ones((2, 3))) * kind.make(np.arange(3.0))
        assert out.shape == (2, 3)

    def test_a_mean_is_bounded_by_its_standard_error(self) -> None:
        draws = np.random.default_rng(0).normal(2.0, 1.0, size=1000)
        assert_mean(draws, 2.0)
        with pytest.raises(AssertionError, match="standard errors"):
            assert_mean(draws, 2.2)

    @pytest.mark.parametrize(
        ("sample", "message"),
        [
            (np.zeros(5), "too few"),
            (np.zeros(50), "no spread"),
            (np.append(np.arange(50.0), np.nan), "not finite"),
        ],
    )
    def test_a_mean_that_cannot_be_bounded(
        self, sample: np.ndarray, message: str
    ) -> None:
        with pytest.raises(AssertionError, match=message):
            assert_mean(sample, 0.0)


class TestClosedForms:
    """Each closed form against an independent evaluation of it."""

    @pytest.mark.parametrize("motion", [solid_body_plane, pure_strain])
    def test_motion_on_a_plane(self, motion: Any) -> None:
        x, y = 10.0 * np.arange(9), 20.0 * np.arange(7)
        field = motion(x[None, :], y[:, None], 1e-4)
        # a linear field, so centred differences are exact inside
        dudy, dudx = np.gradient(field.u, y, x)
        dvdy, dvdx = np.gradient(field.v, y, x)
        assert_matches_oracle(dvdx - dudy, field.vorticity, rtol=1e-12, atol=1e-18)
        assert_matches_oracle(dudx + dvdy, field.divergence, rtol=1e-12, atol=1e-18)
        # W = (∂u/∂x − ∂v/∂y)² + (∂v/∂x + ∂u/∂y)² − ζ²
        strain = (dudx - dvdy) ** 2 + (dvdx + dudy) ** 2
        assert_matches_oracle(
            strain - field.vorticity**2, field.okubo_weiss, rtol=1e-12
        )

    @pytest.mark.parametrize(
        "axis", [(0.0, 0.0, 1e-5), (0.0, 6e-6, 8e-6), (1e-5, 0.0, 0.0)]
    )
    def test_motion_on_a_sphere(self, axis: tuple[float, ...]) -> None:
        errors = []
        for step in (2.0, 1.0):
            lon = np.arange(-30.0, 30.0 + step, step)
            lat = np.arange(-60.0, 60.0 + step, step)
            radius = SPHERE.semi_major
            field = solid_body_sphere(lon, lat[:, None], axis, radius)
            grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon, R=radius)
            inside = (slice(1, -1), slice(1, -1))
            vorticity = np.asarray(gridax.curl(field.u, field.v, grid))[inside]
            divergence = np.asarray(gridax.divergence(field.u, field.v, grid))[inside]
            errors.append(
                (
                    np.max(np.abs(vorticity - field.vorticity[inside])),
                    np.max(np.abs(divergence)),
                )
            )
        # gridax's centred differences converge to the closed form at order two
        for coarse, fine in zip(*errors, strict=True):
            assert fine < 1e-3 * 1e-5
            assert fine < coarse / 3.5 or fine < 1e-18

    def test_polar_rotation_is_zonal_and_fastest_at_the_equator(self) -> None:
        field = solid_body_sphere(30.0, np.asarray([0.0, 60.0]), (0, 0, 1e-5), 6.0e6)
        assert_matches_oracle(field.u, [60.0, 30.0], rtol=1e-14)
        assert_matches_oracle(field.v, [0.0, 0.0], rtol=0.0, atol=1e-14)

    def test_the_pycnocline_reaches_the_threshold_at_25_m(self) -> None:
        profile = linear_pycnocline(np.arange(0.0, 61.0, 5.0))
        assert profile.expected == pytest.approx(25.0, rel=1e-14)
        density = profile.fields["density"]
        at = np.interp(profile.expected, profile.level, density)
        assert at - density[2] == pytest.approx(0.03, rel=1e-9)

    def test_the_inversion_is_crossed_between_its_bracketing_levels(self) -> None:
        profile = neutral_layer_under_inversion(np.arange(10.0, 3000.0, 100.0))
        theta, u = profile.fields["theta_v"], profile.fields["u"]
        z = profile.level
        richardson = (
            9.80665
            * (theta - theta[0])
            * (z - z[0])
            / (theta[0] * np.where(u == u[0], 1.0, (u - u[0]) ** 2))
        )
        assert np.interp(0.25, richardson[8:], z[8:]) == pytest.approx(
            profile.expected, rel=1e-12
        )
        assert profile.expected > 800.0

    def test_an_inversion_that_is_never_crossed(self) -> None:
        with pytest.raises(ValueError, match="critical"):
            neutral_layer_under_inversion([10.0, 50.0, 100.0])

    def test_the_sinusoid_holds_its_variance_at_one_wavenumber(self) -> None:
        values, wavenumber, variance = sinusoid(64, 0.5, 5, 3.0)
        assert np.mean(values**2) == pytest.approx(variance, rel=1e-13)
        power = np.abs(np.fft.rfft(values)) ** 2
        peak = int(np.argmax(power))
        assert np.fft.rfftfreq(64, 0.5)[peak] == pytest.approx(wavenumber, rel=1e-14)
        assert power[peak] / np.sum(power) == pytest.approx(1.0, rel=1e-13)

    def test_the_patterns_are_orthonormal_and_uncorrelated(self) -> None:
        patterns = two_patterns()
        gram = patterns.modes @ patterns.modes.T
        assert_matches_oracle(gram, np.eye(2), rtol=0.0, atol=1e-14)
        covariance = np.cov(patterns.amplitudes, rowvar=False)
        assert_matches_oracle(
            covariance, np.diag(patterns.variances), rtol=1e-13, atol=1e-13
        )

    def test_the_ensemble_is_reliable(self) -> None:
        ensemble, truth, crps = reliable_ensemble(5, 4000)
        members = ensemble.shape[-1]
        # fair CRPS: E|X − y| − ½ E|X − X'|, over distinct members
        error = np.mean(np.abs(ensemble - truth[:, None]), axis=-1)
        pairs = np.abs(ensemble[:, :, None] - ensemble[:, None, :])
        spread = np.sum(pairs, axis=(1, 2)) / (members * (members - 1))
        assert_mean(error - 0.5 * spread, crps)
        # (E + 1)/E · s² and the squared error of the mean have the same mean
        variance = np.var(ensemble, axis=-1, ddof=1) * (members + 1) / members
        squared_error = (np.mean(ensemble, axis=-1) - truth) ** 2
        assert_mean(variance - squared_error, 0.0)


class TestStoredXropsValues:
    def test_the_files_are_small_and_float64(self) -> None:
        folder = Path(__file__).parent / "data"
        files = sorted(folder.glob("*.npz"))
        names = [f"xrops_{family}_{XROPS_COMMIT}.npz" for family in XROPS_FAMILIES]
        assert [file.name for file in files] == names
        assert sum(file.stat().st_size for file in files) < 500_000
        for family in XROPS_FAMILIES:
            for array in xrops_values(family).values():
                assert array.dtype in (np.float64, np.int64)

    def test_the_plane_vorticities_add_up_inside(self) -> None:
        stored = xrops_values("kinematics")
        x, y = stored["plane_x"], stored["plane_y"]
        dudy = np.gradient(stored["plane_u"], y, axis=0)
        dvdx = np.gradient(stored["plane_v"], x, axis=1)
        total = stored["plane_shear_vorticity"] + stored["plane_curvature_vorticity"]
        inside = (slice(1, -1), slice(1, -1))
        scale = np.max(np.abs(total))
        assert_matches_oracle(
            total[inside], (dvdx - dudy)[inside], rtol=0.0, atol=1e-12 * scale
        )
        for name in ("shear", "curvature"):
            assert stored[f"plane_calm_{name}_vorticity"][5, 7] == 0.0

    def test_the_plane_fields_are_their_formulas_inside(self) -> None:
        stored = xrops_values("kinematics")
        x, y = stored["plane_x"], stored["plane_y"]
        u, v = stored["plane_u"], stored["plane_v"]
        gradient = {
            name: np.gradient(stored[f"plane_{name}"], y, x)
            for name in ("u", "v", "scalar", "height")
        }
        (dudy, dudx), (dvdy, dvdx) = gradient["u"], gradient["v"]
        (dcdy, dcdx), (dhdy, dhdx) = gradient["scalar"], gradient["height"]
        f, g = stored["plane_coriolis"], stored["gravity"]
        # F = −∇c · E ∇c / |∇c|, with E the symmetric part of the gradient of u
        cross = 0.5 * (dudy + dvdx)
        stretching = dcdx**2 * dudx + 2.0 * dcdx * dcdy * cross + dcdy**2 * dvdy
        for name, expected in (
            ("advection", -(u * dcdx + v * dcdy)),
            ("frontogenesis", -stretching / np.hypot(dcdx, dcdy)),
            ("residual_u", f * u + g * dhdy),
            ("residual_v", f * v - g * dhdx),
        ):
            actual = stored[f"plane_{name}"][1:-1, 1:-1]
            scale = np.max(np.abs(actual))
            assert_matches_oracle(
                actual, expected[1:-1, 1:-1], rtol=0.0, atol=1e-11 * scale
            )

    def test_the_sphere_fields_are_their_formulas_inside(self) -> None:
        stored = xrops_values("kinematics")
        phi = np.deg2rad(stored["sphere_lat"])[:, None]
        radius, g = SPHERE.semi_major, stored["gravity"]

        def gradient(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dy, dx = np.gradient(values, phi[:, 0], np.deg2rad(stored["sphere_lon"]))
            return dx / (radius * np.cos(phi)), dy / radius

        u, v = stored["plane_u"], stored["plane_v"]
        dcdx, dcdy = gradient(stored["plane_scalar"])
        dhdx, dhdy = gradient(stored["plane_height"])
        f = stored["sphere_coriolis"][:, None]
        for name, expected in (
            ("advection", -(u * dcdx + v * dcdy)),
            ("residual_u", f * u + g * dhdy),
            ("residual_v", f * v - g * dhdx),
        ):
            actual = stored[f"sphere_{name}"][1:-1, 1:-1]
            scale = np.max(np.abs(actual))
            assert_matches_oracle(
                actual, expected[1:-1, 1:-1], rtol=0.0, atol=1e-11 * scale
            )

    def test_the_stratification_is_its_formula(self) -> None:
        stored = xrops_values("columns")
        slope = np.gradient(stored["ocean_density"], stored["ocean_depth"], axis=-1)
        inside = (slice(None), slice(1, -1))
        assert_matches_oracle(
            stored["ocean_n_squared"][inside],
            (9.80665 / 1025.0 * slope)[inside],
            rtol=1e-11,
        )
        # the first level below 10 m that is heavier than the one at 10 m by Δρ
        below = stored["ocean_density"] - stored["ocean_density"][:, 2:3] > 0.03
        below[:, :2] = False
        expected = stored["ocean_depth"][np.argmax(below, axis=-1)]
        assert_matches_oracle(stored["ocean_mixed_layer_level"], expected, rtol=0.0)

    def test_the_boundary_layer_ends_at_the_critical_number(self) -> None:
        stored = xrops_values("columns")
        heights = stored["air_boundary_layer_height"]
        for column in range(5):
            z, theta, u, v = (
                stored[f"air_{name}"][column]
                for name in ("height", "theta_v", "u", "v")
            )
            shear = (u - u[0]) ** 2 + (v - v[0]) ** 2
            buoyancy = 9.80665 * (theta - theta[0]) * (z - z[0])
            richardson = np.zeros(z.size)
            moving = shear > 0.0
            richardson[moving] = buoyancy[moving] / (theta[0] * shear[moving])
            calm = ~moving & (buoyancy != 0.0)
            richardson[calm] = np.copysign(np.inf, buoyancy[calm])
            upper = int(np.argmax(richardson >= 0.25))
            lower = upper - 1
            # an infinite number is crossed at the level below it
            fraction = (0.25 - richardson[lower]) / (
                richardson[upper] - richardson[lower]
            )
            expected = z[lower] + fraction * (z[upper] - z[lower])
            assert heights[column] == pytest.approx(expected, rel=1e-12)

    def test_the_column_integrals_are_trapezoids(self) -> None:
        stored = xrops_values("columns")
        z, theta = stored["air_height"], stored["air_theta_v"]
        assert_matches_oracle(
            stored["air_integral_per_column_trapezoid"],
            np.trapezoid(theta, z, axis=-1),
            rtol=1e-13,
        )
        assert_matches_oracle(
            stored["air_integral_shared_trapezoid"],
            np.trapezoid(theta, z[0], axis=-1),
            rtol=1e-13,
        )
        # only the neutral column never reaches the critical number
        height = stored["air_boundary_layer_height"]
        assert list(np.isnan(height)) == [False] * 5 + [True]

    def test_the_monthly_climatology_is_a_grouped_mean(self) -> None:
        stored = xrops_values("climate")
        values, month = stored["values"], stored["month"]
        means = np.stack(
            [np.mean(values[:, month == m], axis=-1) for m in range(1, 13)], axis=-1
        )
        assert_matches_oracle(stored["monthly_climatology"], means, rtol=1e-13)
        assert_matches_oracle(
            stored["monthly_anomaly"],
            values - means[:, month - 1],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_the_spectra_satisfy_parseval(self) -> None:
        stored = xrops_values("spectra")
        u = stored["series_u"]
        step = stored["series_frequency"][1] - stored["series_frequency"][0]
        assert_matches_oracle(
            np.sum(stored["series_psd_boxcar"], axis=-1) * step,
            np.var(u, axis=-1),
            rtol=1e-12,
        )
        rotary = stored["rotary_psd_ccw"] + stored["rotary_psd_cw"]
        assert_matches_oracle(
            np.sum(rotary, axis=-1) * stored["rotary_wavenumber"][0],
            np.var(u, axis=-1) + np.var(stored["series_v"], axis=-1),
            rtol=1e-12,
        )

    def test_the_plane_spectrum_is_the_transform_of_the_field(self) -> None:
        stored = xrops_values("spectra")
        u, spacing = stored["plane_u"], float(stored["plane_spacing"])
        # |û|² Δx Δy / (Nx Ny), zero wavenumber in the middle
        power = np.abs(np.fft.fft2(u - np.mean(u))) ** 2 * spacing**2 / u.size
        scale = np.max(power)
        assert_matches_oracle(
            stored["plane_psd"], np.fft.fftshift(power), rtol=0.0, atol=1e-12 * scale
        )

    def test_the_resolved_scale_is_where_the_score_crosses_a_half(self) -> None:
        stored = xrops_values("skill")
        truth = stored["spectral_truth"]
        error = stored["spectral_prediction"] - truth
        n = truth.size
        frequency = np.fft.rfftfreq(n, float(stored["spectral_spacing"]))[1:]
        power = [np.abs(np.fft.rfft(x - np.mean(x)))[1:] ** 2 for x in (error, truth)]
        score = 1.0 - power[0] / power[1]
        middle = np.argsort(stored["spectral_frequency"])[n // 2 + 1 :]
        # the Nyquist mode is left out: it is the first of the two-sided axis
        assert_matches_oracle(
            stored["spectral_psd_score"][middle], score[:-1], rtol=1e-9, atol=1e-9
        )
        wavelength = float(stored["spectral_resolved_scale"])
        at = np.interp(1.0 / wavelength, frequency, score)
        assert at == pytest.approx(0.5, abs=0.01)

    def test_the_scores_are_their_formulas(self) -> None:
        stored = xrops_values("skill")
        error = stored["prediction"] - stored["reference"]
        rmse = np.sqrt(np.mean(error**2, axis=-1))
        rms = np.sqrt(np.mean(stored["reference"] ** 2, axis=-1))
        deviation = np.std(stored["reference"], axis=-1)
        for name, expected in (
            ("rmse", rmse),
            ("mae", np.mean(np.abs(error), axis=-1)),
            ("bias", np.mean(error, axis=-1)),
            ("nrmse_rms", 1.0 - rmse / rms),
            ("nrmse_std", 1.0 - rmse / deviation),
        ):
            assert_matches_oracle(stored[name], expected, rtol=1e-12)


@pytest.mark.oracle
class TestLibraries:
    """Each library against a closed form, so that a wrong call shows."""

    def test_metpy_advection_of_a_linear_scalar(self) -> None:
        x, y = np.meshgrid(1e3 * np.arange(8), 2e3 * np.arange(6))
        u, v = 3.0 + 0.0 * x, -2.0 + 0.0 * x
        out = metpy_advection(290.0 + 1e-4 * x + 2e-4 * y, u, v, 1e3, 2e3)
        assert_matches_oracle(out, np.full(x.shape, -3e-4 + 4e-4), rtol=1e-10)

    def test_metpy_frontogenesis_of_pure_strain(self) -> None:
        # isentropes along the axis of dilatation: F = α |∇θ|
        x, y = np.meshgrid(1e3 * np.arange(8), 2e3 * np.arange(6))
        field = pure_strain(x, y, 1e-5)
        out = metpy_frontogenesis(290.0 + 1e-4 * y, field.u, field.v, 1e3, 2e3)
        assert_matches_oracle(out, np.full(x.shape, 1e-9), rtol=1e-9)

    def test_xrft_puts_a_sinusoid_in_one_bin(self) -> None:
        values, wavenumber, variance = sinusoid(64, 0.5, 5, 3.0)
        frequency, density = xrft_power_spectrum(values, 0.5)
        step = frequency[1] - frequency[0]
        assert np.sum(density) * step == pytest.approx(variance, rel=1e-12)
        peak = np.abs(np.abs(frequency) - wavenumber) < 1e-9
        assert np.sum(density[peak]) * step == pytest.approx(variance, rel=1e-12)

    def test_xrft_reproduces_the_stored_spectra(self) -> None:
        stored = xrops_values("spectra")
        for window, name in ((None, "boxcar"), ("hann", "hann")):
            frequency, density = xrft_power_spectrum(
                stored["series_u"], float(stored["series_spacing"]), window=window
            )
            assert_matches_oracle(frequency, stored["series_frequency"], rtol=1e-12)
            assert_matches_oracle(density, stored[f"series_psd_{name}"], rtol=1e-12)

    def test_xskillscore_crps_is_the_biased_one(self) -> None:
        ensemble, truth, _ = reliable_ensemble(5, 50)
        error = np.mean(np.abs(ensemble - truth[:, None]), axis=-1)
        pairs = np.abs(ensemble[:, :, None] - ensemble[:, None, :])
        biased = error - 0.5 * np.mean(pairs, axis=(1, 2))
        assert_matches_oracle(xskillscore_crps(truth, ensemble), biased, rtol=1e-12)

    @pytest.mark.slow
    def test_the_eofs_are_the_two_patterns_up_to_sign(self) -> None:
        patterns = two_patterns()
        sklearn_modes, _, sklearn_variances = sklearn_pca(patterns.field, 2)
        functax_modes, functax_variances = functax_eofs(patterns.field, 2)
        for modes, variances in (
            (sklearn_modes, sklearn_variances),
            (functax_modes, functax_variances),
        ):
            assert_matches_oracle(variances, patterns.variances, rtol=1e-12)
            overlap = np.abs(np.sum(modes * patterns.modes, axis=-1))
            assert_matches_oracle(overlap, np.ones(2), rtol=1e-12)
