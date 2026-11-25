"""Spectra: how the variance of a field is spread over scales."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _backends import assert_backends_agree, assert_refuses_dataarrays, needs_jax
from _harness import (
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import sinusoid, xrft_power_spectrum, xrops_values
from _sampling import assert_mean
from hypothesis import given, settings, strategies as st
from scipy import signal

from diagnostix import ValidationError
from diagnostix.spectra import (
    Spectrum,
    Spectrum2D,
    isotropic_spectrum,
    power_spectrum,
    power_spectrum_2d,
)


WINDOWS = (None, "hann")
#: Odd and even sizes, oblong, with unequal spacings.
PLANES = {
    "square": ((16, 16), 1.5),
    "oblong": ((15, 18), (2.0, 0.5)),
    "long": ((12, 31), (0.7, 1.3)),
    "tall": ((32, 9), (1.0, 1.0)),
}


def noise(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def hann(*shape: int) -> np.ndarray:
    """SciPy's periodic Hann taper over one or two axes."""
    tapers = [signal.get_window("hann", n) for n in shape]
    return tapers[0] if len(shape) == 1 else np.outer(*tapers)


def step_of(wavenumber: Any) -> float:
    return float(wavenumber[1] - wavenumber[0])


def parts(spectrum: Any) -> tuple[Any, ...]:
    if isinstance(spectrum, Spectrum2D):
        return spectrum.wavenumber_y, spectrum.wavenumber_x, spectrum.power
    return spectrum.wavenumber, spectrum.power


class TestPowerSpectrum:
    @pytest.mark.parametrize("n", [2, 3, 16, 17])
    @pytest.mark.parametrize("window", WINDOWS)
    def test_parseval(self, n: int, window: str | None) -> None:
        x = noise(0, 3, n)
        out = power_spectrum(x, 0.25, window=window, detrend="constant")
        w = np.ones(n) if window is None else hann(n)
        tapered = w * (x - x.mean(-1, keepdims=True))
        expected = np.mean(tapered**2, axis=-1) / np.mean(w**2)
        total = np.sum(out.power, axis=-1) * step_of(out.wavenumber)
        assert out.power.shape == (3, n // 2 + 1)
        assert_matches_oracle(total, expected, rtol=1e-13)

    @given(st.integers(1, 14))
    def test_a_sinusoid_is_in_one_bin(self, cycles: int) -> None:
        x, wavenumber, variance = sinusoid(30, 0.4, cycles, 3.0)
        out = power_spectrum(x, 0.4, window=None, detrend=None)
        expected = np.zeros(16)
        expected[cycles] = variance
        assert out.wavenumber[cycles] == pytest.approx(wavenumber, rel=1e-14)
        step = step_of(out.wavenumber)
        assert_matches_oracle(out.power * step, expected, rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("n", [30, 31])
    def test_scipy_periodogram(self, window: str | None, n: int) -> None:
        x = noise(1, 2, n)
        for detrend in ("constant", "linear", None):
            wavenumber, power = signal.periodogram(
                x, 1.0 / 0.4, window=window or "boxcar", detrend=detrend or False
            )
            out = power_spectrum(x, 0.4, window=window, detrend=detrend)
            assert_matches_oracle(out.wavenumber, wavenumber, rtol=1e-13)
            # the bins that a detrended, untapered series has nothing in
            assert_matches_oracle(out.power, power, rtol=1e-11, atol=1e-13)

    def test_a_taper_of_the_callers(self) -> None:
        x = noise(2, 40)
        w = signal.windows.tukey(40, 0.5)
        _, power = signal.periodogram(x, 2.0, window=w, detrend="linear")
        out = power_spectrum(x, 0.5, window=w)
        assert_matches_oracle(out.power, power, rtol=1e-11)
        same = power_spectrum(x, 0.5, window=hann(40))
        assert_matches_oracle(same.power, power_spectrum(x, 0.5).power, rtol=1e-12)

    @pytest.mark.parametrize("window", WINDOWS)
    def test_stored_xrops(self, window: str | None) -> None:
        stored = xrops_values("spectra")
        spacing = float(stored["series_spacing"])
        out = power_spectrum(
            stored["series_u"], spacing, window=window, detrend="constant"
        )
        # xrops is two-sided, from −Nyquist: half of ours between the ends
        two_sided = stored[f"series_psd_{window or 'boxcar'}"]
        assert_matches_oracle(
            out.power[:, 1:-1], 2.0 * two_sided[:, 31:], rtol=1e-11, atol=1e-13
        )
        assert_matches_oracle(out.power[:, -1], two_sided[:, 0], rtol=1e-11, atol=1e-13)

    @pytest.mark.oracle
    def test_xrft(self) -> None:
        x = noise(3, 2, 24)
        for window in WINDOWS:
            frequency, density = xrft_power_spectrum(x, 0.5, window=window)
            out = power_spectrum(x, 0.5, window=window, detrend="constant")
            assert_matches_oracle(out.wavenumber[1:-1], frequency[13:], rtol=1e-12)
            assert_matches_oracle(out.power[:, 1:-1], 2 * density[:, 13:], rtol=1e-11)

    def test_white_noise_is_flat(self) -> None:
        # each bin between the ends is exponential about 2 σ² Δx
        out = power_spectrum(noise(4, 200, 32), 0.5, window=None, detrend=None)
        assert_mean(out.power[:, 1:-1], 2.0 * 0.5)
        assert_mean(out.power[:, 0], 0.5)
        assert_mean(out.power[:, -1], 0.5)

    def test_the_line_is_removed(self) -> None:
        position = np.arange(25.0)
        x = noise(5, 25)
        line = 4.0 - 3.0 * position
        for window in WINDOWS:
            bare = power_spectrum(x, 1.0, window=window)
            assert_matches_oracle(
                power_spectrum(x + line, 1.0, window=window).power,
                bare.power,
                rtol=0.0,
                atol=1e-12,
            )

    def test_axis(self) -> None:
        x = noise(6, 4, 9, 3)
        out = power_spectrum(x, 2.0, axis=1)
        moved = power_spectrum(np.moveaxis(x, 1, -1), 2.0)
        assert out.power.shape == (4, 3, 5)
        assert_matches_oracle(out.power, moved.power, rtol=0.0)

    def test_a_nan_is_in_every_bin_of_its_series(self) -> None:
        x = noise(7, 2, 12)
        x[0, 3] = np.nan
        out = power_spectrum(x, 1.0)
        assert np.isnan(out.power[0]).all()
        assert np.isfinite(out.power[1]).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("n", [12, 13])
    def test_backends_and_precision(self, window: str | None, n: int) -> None:
        def of(x: Any) -> Any:
            wavenumber, power = parts(power_spectrum(x, 0.3, axis=0, window=window))
            # a detrended series has no power in bin 0, but for its rounding
            return wavenumber, 1.0 + power

        x = 5.0 + noise(8, n, 2)
        assert_backends_agree(of, x)
        assert_traces(of, noise(9, 1, n, 2), rtol=1e-11)
        assert_refuses_dataarrays(of, x)

    @pytest.mark.slow
    @needs_jax
    def test_gradient_of_a_spectral_loss(self) -> None:
        import jax.numpy as jnp

        target = power_spectrum(noise(10, 14), 1.0).power

        def loss(x: Any) -> Any:
            return jnp.sum((power_spectrum(x, 1.0).power - target) ** 2)

        assert_grad_matches_fd(loss, noise(11, 14))
        # a bin without power still has a finite gradient
        assert_grad_matches_fd(
            lambda x: jnp.sum(power_spectrum(x, 1.0).power), np.zeros(8), rtol=1e-5
        )

    def test_errors(self) -> None:
        x = noise(12, 8)
        for spacing in (0.0, -1.0, np.nan, np.inf, (1.0, 2.0)):
            with pytest.raises(ValidationError, match="spacing"):
                power_spectrum(x, spacing)
        with pytest.raises(ValidationError, match="no axis 1"):
            power_spectrum(x, 1.0, axis=1)
        with pytest.raises(ValidationError, match="at least 2 samples"):
            power_spectrum(x[:1], 1.0)
        with pytest.raises(ValidationError, match="'hann'"):
            power_spectrum(x, 1.0, window="hanning")
        for window in (np.ones(7), np.zeros(8), np.full(8, np.nan)):
            with pytest.raises(ValidationError, match="a window must be"):
                power_spectrum(x, 1.0, window=window)
        with pytest.raises(ValidationError, match="'detrend'"):
            power_spectrum(x, 1.0, detrend="quadratic")


@pytest.mark.parametrize(("shape", "spacing"), PLANES.values(), ids=PLANES)
class TestPlanes:
    @pytest.mark.parametrize("window", WINDOWS)
    def test_parseval(self, shape: Any, spacing: Any, window: str | None) -> None:
        x = noise(13, 2, *shape)
        w = np.ones(shape) if window is None else hann(*shape)
        tapered = w * (x - x.mean((-2, -1), keepdims=True))
        expected = np.mean(tapered**2, axis=(-2, -1)) / np.mean(w**2)
        options: Any = {"window": window, "detrend": "constant"}
        plane = power_spectrum_2d(x, spacing, **options)
        cell = step_of(plane.wavenumber_y) * step_of(plane.wavenumber_x)
        total = np.sum(plane.power, axis=(-2, -1)) * cell
        assert_matches_oracle(total, expected, rtol=1e-12)
        rings = isotropic_spectrum(x, spacing, **options)
        total = np.sum(rings.power, axis=-1) * step_of(rings.wavenumber)
        assert_matches_oracle(total, expected, rtol=1e-12)

    def test_the_rings_are_those_of_the_plane(self, shape: Any, spacing: Any) -> None:
        x = noise(14, *shape)
        plane = power_spectrum_2d(x, spacing)
        rings = isotropic_spectrum(x, spacing)
        ky, kx = plane.wavenumber_y, plane.wavenumber_x
        step = min(step_of(ky), step_of(kx))
        assert step_of(rings.wavenumber) == pytest.approx(step, rel=1e-12)
        size = np.hypot(ky[:, None], kx)
        cell = step_of(ky) * step_of(kx)
        for n, power in enumerate(rings.power):
            # 1e-9 of a ring: no mode of these grids is so near an edge
            inside = np.abs(size / step - n) < 0.5 + 1e-9
            expected = np.sum(plane.power[inside]) * cell / step
            assert power == pytest.approx(expected, rel=1e-11)

    def test_a_plane_is_removed(self, shape: Any, spacing: Any) -> None:
        x = noise(15, *shape)
        rows, columns = np.mgrid[: shape[0], : shape[1]]
        plane = 7.0 - 2.0 * rows + 3.0 * columns
        for spectrum in (power_spectrum_2d, isotropic_spectrum):
            assert_matches_oracle(
                spectrum(x + plane, spacing).power,
                spectrum(x, spacing).power,
                rtol=0.0,
                atol=1e-9,
            )


class TestPowerSpectrum2D:
    def test_stored_xrops(self) -> None:
        stored = xrops_values("spectra")
        out = power_spectrum_2d(
            stored["plane_u"],
            float(stored["plane_spacing"]),
            window=None,
            detrend="constant",
        )
        scale = np.max(stored["plane_psd"])
        assert_matches_oracle(
            out.wavenumber_x, stored["plane_frequency"], rtol=1e-12, atol=1e-15
        )
        assert_matches_oracle(
            out.power, stored["plane_psd"], rtol=0.0, atol=1e-12 * scale
        )

    def test_a_wave_is_at_plus_and_minus_its_wavenumber(self) -> None:
        y, x = np.mgrid[:8, :12]
        wave = 2.0 * np.cos(2.0 * np.pi * (2 * y / 8 + 3 * x / 12))
        out = power_spectrum_2d(wave, (1.0, 0.5), window=None, detrend=None)
        expected = np.zeros((8, 12))
        for sign in (1, -1):
            expected[4 + 2 * sign, 6 + 3 * sign] = 1.0
        cell = step_of(out.wavenumber_y) * step_of(out.wavenumber_x)
        assert_matches_oracle(out.power * cell, expected, rtol=0.0, atol=1e-13)

    def test_white_noise_is_flat(self) -> None:
        out = power_spectrum_2d(noise(16, 60, 8, 8), 0.5, window=None, detrend=None)
        # a mode and its conjugate share a draw, so average over each field
        assert_mean(np.mean(out.power, axis=(-2, -1)), 0.25)

    @pytest.mark.slow
    @pytest.mark.parametrize("window", WINDOWS)
    def test_backends_and_precision(self, window: str | None) -> None:
        def of(x: Any) -> Any:
            spectrum = power_spectrum_2d(x, (0.5, 2.0), window=window)
            ky, kx, power = parts(spectrum)
            return ky, kx, 1.0 + power

        x = noise(17, 2, 6, 7)
        assert_backends_agree(of, x)
        assert_traces(of, noise(18, 1, 6, 7), rtol=1e-11)
        assert_refuses_dataarrays(of, x)
        assert_grad_matches_fd(lambda x: of(x)[2], x[0])

    def test_errors(self) -> None:
        with pytest.raises(ValidationError, match="at least 2 samples"):
            power_spectrum_2d(noise(0, 8), 1.0)
        with pytest.raises(ValidationError, match="spacing"):
            power_spectrum_2d(noise(0, 8, 8), (1.0, 1.0, 1.0))
        with pytest.raises(ValidationError, match="a window must be"):
            power_spectrum_2d(noise(0, 8, 6), 1.0, window=np.ones((6, 8)))


class TestIsotropicSpectrum:
    def test_every_mode_of_integer_radius_is_in_its_ring(self) -> None:
        # floor(|k| / Δk) misfiles some: jejjohnson/diagnostix#15
        for n in (32, 60, 96):
            y, x = np.mgrid[:n, :n]
            for j, i in ((3, 4), (4, 3), (0, 7), (5, 12), (8, 6)):
                if max(i, j) >= n // 2:
                    continue
                wave = np.cos(2.0 * np.pi * (j * y - i * x) / n)
                out = isotropic_spectrum(wave, 0.1, window=None, detrend=None)
                radius = round(np.hypot(i, j))
                variance = out.power * step_of(out.wavenumber)
                expected = np.zeros(variance.shape)
                expected[radius] = 0.5
                assert_matches_oracle(variance, expected, rtol=0.0, atol=1e-12)

    @settings(max_examples=20)
    @given(st.integers(2, 12), st.integers(2, 12), st.floats(0.2, 5.0))
    def test_parseval_on_any_grid(self, ny: int, nx: int, dy: float) -> None:
        x = noise(19, ny, nx)
        out = isotropic_spectrum(x, (dy, 1.0), window=None, detrend=None)
        # a grid of few modes may have one ring only
        step = min(1.0 / (ny * dy), 1.0 / nx)
        assert np.sum(out.power) * step == pytest.approx(np.mean(x**2), rel=1e-12)

    def test_it_does_not_depend_on_direction(self) -> None:
        x = noise(20, 18, 11)
        # a periodic taper is not symmetric, and its detrended field is not either
        flipped = (x[::-1, :], (0.5, 2.0), None)
        for same, spacing, window in ((x.T, (2.0, 0.5), "hann"), flipped):
            out = isotropic_spectrum(x, (0.5, 2.0), window=window)
            again = isotropic_spectrum(same, spacing, window=window)
            assert_matches_oracle(again.wavenumber, out.wavenumber, rtol=1e-13)
            assert_matches_oracle(again.power, out.power, rtol=0.0, atol=1e-13)

    @pytest.mark.slow
    @pytest.mark.parametrize("window", WINDOWS)
    def test_backends_and_precision(self, window: str | None) -> None:
        def of(x: Any) -> Any:
            spectrum = isotropic_spectrum(x, (0.5, 0.75), window=window)
            wavenumber, power = parts(spectrum)
            return wavenumber, 1.0 + power

        x = noise(21, 2, 9, 8)
        assert_backends_agree(of, x)
        assert_traces(of, noise(22, 1, 9, 8), rtol=1e-11)
        assert_refuses_dataarrays(of, x)
        assert_grad_matches_fd(lambda x: of(x)[1], x[0])

    @needs_jax
    def test_the_records_are_pytrees(self) -> None:
        import jax

        spectrum = power_spectrum(noise(23, 8), 1.0)
        leaves, tree = jax.tree.flatten(spectrum)
        assert len(leaves) == 2
        assert isinstance(jax.tree.unflatten(tree, leaves), Spectrum)
        assert len(jax.tree.leaves(power_spectrum_2d(noise(24, 4, 4), 1.0))) == 3

    def test_a_nan_is_in_every_ring(self) -> None:
        x = noise(25, 6, 6)
        x[2, 2] = np.nan
        assert np.isnan(isotropic_spectrum(x, 1.0).power).all()
