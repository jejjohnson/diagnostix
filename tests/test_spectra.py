"""Spectra: how the variance of a field is spread over scales."""

from __future__ import annotations

import math
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
from _oracles import sinusoid, xrft_power_spectrum, xrops_values
from _sampling import assert_mean
from hypothesis import given, settings, strategies as st
from pipekit_array.functions import namespace_of
from scipy import signal

from diagnostix import ValidationError
from diagnostix.spectra import (
    Spectrum,
    Spectrum2D,
    coherence,
    cross_spectrum,
    enstrophy_flux,
    integral_scale,
    inverse_wavelet_transform,
    isotropic_spectrum,
    kinetic_energy_flux,
    power_spectrum,
    power_spectrum_2d,
    rotary_spectrum,
    spectral_slope,
    wavelet_power_2d,
    wavelet_significance,
    wavelet_transform,
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


def two_parts(z: Any) -> tuple[Any, Any]:
    """9 + the real and the imaginary part, which may be 0, of any library."""
    xp = namespace_of(z)
    return 9.0 + xp.real(z), 9.0 + xp.imag(z)


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


SEGMENTS = {
    "whole": (60, None, 0.5),
    "half overlap": (100, 16, 0.5),
    "odd segments": (97, 15, 0.5),
    "no overlap, a tail": (70, 16, 0.0),
    "three quarters": (64, 16, 0.75),
}


def scipy_options(segment: int | None, overlap: float, detrend: Any) -> dict[str, Any]:
    if segment is None:
        return {"nperseg": 60, "detrend": detrend or False}
    noverlap = int(overlap * segment)
    return {"nperseg": segment, "noverlap": noverlap, "detrend": detrend or False}


@pytest.mark.parametrize(("n", "segment", "overlap"), SEGMENTS.values(), ids=SEGMENTS)
@pytest.mark.parametrize("detrend", ["constant", "linear", None])
class TestSegments:
    def test_scipy_csd(
        self, n: int, segment: int | None, overlap: float, detrend: Any
    ) -> None:
        x, y = noise(30, 2, 3, n)
        options = scipy_options(segment, overlap, detrend)
        wavenumber, expected = signal.csd(x, y, 1.0 / 0.4, **options)
        out = cross_spectrum(
            x, y, 0.4, segment=segment, overlap=overlap, detrend=detrend
        )
        assert_matches_oracle(out.wavenumber, wavenumber, rtol=1e-13)
        # SciPy averages x̂* ŷ
        for part, sign in ((np.real, 1.0), (np.imag, -1.0)):
            assert_matches_oracle(
                part(out.cross), sign * part(expected), rtol=0.0, atol=1e-13
            )

    def test_scipy_welch(
        self, n: int, segment: int | None, overlap: float, detrend: Any
    ) -> None:
        x = noise(31, n)
        options = scipy_options(segment, overlap, detrend)
        _, expected = signal.welch(x, 1.0 / 0.4, **options)
        out = cross_spectrum(
            x, x, 0.4, segment=segment, overlap=overlap, detrend=detrend
        )
        assert_matches_oracle(np.real(out.cross), expected, rtol=1e-11)
        assert_matches_oracle(np.imag(out.cross), 0.0 * expected, rtol=0.0, atol=1e-16)

    def test_scipy_coherence(
        self, n: int, segment: int | None, overlap: float, detrend: Any
    ) -> None:
        if segment is None:
            return
        shared, x, y = noise(32, 3, 2, n)
        x, y = x + shared, y - shared
        options = scipy_options(segment, overlap, detrend)
        _, expected = signal.coherence(x, y, 1.0 / 0.4, **options)
        out = coherence(x, y, 0.4, segment=segment, overlap=overlap, detrend=detrend)
        assert_matches_oracle(out.coherence, expected, rtol=1e-10)


class TestCrossSpectrum:
    @pytest.mark.parametrize("n", [20, 21])
    def test_parseval(self, n: int) -> None:
        x, y = noise(33, 2, n)
        out = cross_spectrum(x, y, 0.5, window=None)
        covariance = np.mean((x - x.mean()) * (y - y.mean()))
        total = np.sum(np.real(out.cross)) * step_of(out.wavenumber)
        assert total == pytest.approx(covariance, rel=1e-12)

    def test_the_power_spectrum_is_that_of_one_series(self) -> None:
        x = noise(34, 2, 18)
        for window in WINDOWS:
            out = cross_spectrum(x, x, 0.5, window=window, detrend="linear")
            power = power_spectrum(x, 0.5, window=window).power
            assert_matches_oracle(np.real(out.cross), power, rtol=1e-12)

    def test_the_phase_is_positive_where_x_leads(self) -> None:
        x = noise(35, 16)
        out = cross_spectrum(x, np.roll(x, 3), 0.5, window=None)
        # y(t) = x(t − τ): the phase is 2π k τ
        phase = 2.0 * np.pi * out.wavenumber * 1.5
        expected = np.abs(out.cross) * np.exp(1j * phase)
        for part in (np.real, np.imag):
            assert_matches_oracle(part(out.cross), part(expected), rtol=0, atol=1e-13)

    def test_axis_and_broadcasting(self) -> None:
        x, y = noise(36, 3, 20, 2), noise(37, 20, 1)
        out = cross_spectrum(x, y, 1.0, axis=1, segment=8)
        assert out.cross.shape == (3, 2, 5)
        one = cross_spectrum(x[2, :, 1], y[:, 0], 1.0, segment=8)
        assert_matches_oracle(np.abs(out.cross[2, 1]), np.abs(one.cross), rtol=1e-12)

    @pytest.mark.slow
    @pytest.mark.parametrize("segment", [None, 7])
    def test_backends_and_precision(self, segment: int | None) -> None:
        def of(x: Any, y: Any) -> Any:
            out = cross_spectrum(x, y, 0.3, axis=0, segment=segment)
            return out.wavenumber, *two_parts(out.cross)

        x, y = noise(38, 2, 20, 2)
        assert_backends_agree(of, x, y)
        assert_traces(of, noise(39, 1, 20, 2), noise(40, 1, 20, 2))
        assert_refuses_dataarrays(of, x, y)
        assert_grad_matches_fd(lambda x: of(x, y[:, :1])[1], x[:, :1])

    def test_errors(self) -> None:
        x = noise(41, 12)
        for segment in (1, 0, 13):
            with pytest.raises(ValidationError, match="a segment must have"):
                cross_spectrum(x, x, 1.0, segment=segment)
        for overlap in (1.0, -0.1, np.nan):
            with pytest.raises(ValidationError, match="overlap"):
                cross_spectrum(x, x, 1.0, segment=4, overlap=overlap)
        with pytest.raises(ValidationError, match="a window must be"):
            cross_spectrum(x, x, 1.0, segment=4, window=np.ones(12))
        with pytest.raises(ValidationError, match="spacing"):
            cross_spectrum(x, x, 0.0)
        with pytest.raises(ValidationError, match="no axis"):
            cross_spectrum(x, x, 1.0, axis=2)


class TestCoherence:
    @pytest.mark.parametrize("n_averages", [4, 16, 64])
    def test_independent_noise_falls_as_one_over_the_averages(
        self, n_averages: int
    ) -> None:
        # M independent transforms: each bin is a Beta(1, M − 1) draw, mean 1/M
        x, y = noise(42, 2, 120, n_averages, 16)
        over_axis = coherence(x, y, 1.0, average=1, window=None, detrend=None)
        assert_mean(over_axis.coherence[:, 1:-1], 1.0 / n_averages)
        x, y = (values.reshape(120, -1) for values in (x, y))
        over_segments = coherence(
            x, y, 1.0, segment=16, overlap=0.0, window=None, detrend=None
        )
        assert_matches_oracle(over_segments.coherence, over_axis.coherence, rtol=1e-12)

    def test_a_known_coherence(self) -> None:
        shared, noise_x = noise(43, 2, 200, 4096)
        out = coherence(shared, shared + noise_x, 1.0, segment=256, average=0)
        # half of the variance is shared; 200 × 31 transforms
        assert_matches_oracle(out.coherence, np.full(129, 0.5), rtol=0.0, atol=0.04)

    def test_a_lagged_or_scaled_copy_is_coherent(self) -> None:
        x = noise(44, 30, 16)
        for y in (np.roll(x, 2, axis=-1), -3.0 * x):
            out = coherence(x, y, 1.0, average=0, window=None, detrend=None)
            assert_matches_oracle(out.coherence, np.ones(9), rtol=1e-12)

    def test_average_axes(self) -> None:
        shared, x, y = noise(45, 3, 4, 5, 16, 3)
        x, y = x + shared, y + shared
        both = coherence(x, y, 1.0, axis=2, average=(0, -1))
        flat = [np.moveaxis(a, 2, -1).swapaxes(0, 1).reshape(5, -1, 16) for a in (x, y)]
        expected = coherence(*flat, 1.0, average=1)
        assert both.coherence.shape == (5, 9)
        assert_matches_oracle(both.coherence, expected.coherence, rtol=1e-12)

    def test_no_power_is_nan(self) -> None:
        x = noise(46, 6, 16)
        out = coherence(x, np.zeros(16), 1.0, average=0)
        assert np.isnan(out.coherence).all()

    def test_small_signals_in_float32(self) -> None:
        x, y = noise(47, 2, 8, 16)
        expected = coherence(x, y, 1.0, average=0).coherence
        small = [(1e-15 * a).astype(np.float32) for a in (x, y)]
        out = coherence(*small, 1.0, average=0).coherence
        assert_matches_oracle(out, expected, rtol=1e-4)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any, y: Any) -> Any:
            out = coherence(x, y, 0.3, segment=8, average=0)
            return out.wavenumber, out.coherence

        x, y = noise(48, 2, 3, 2, 20)
        y = y + x
        assert_backends_agree(of, x, y, rtol=1e-11, rtol32=1e-4)
        assert_traces(of, x[None], y[None], rtol=1e-11)
        assert_refuses_dataarrays(of, x, y)
        assert_grad_matches_fd(
            lambda x: coherence(x, y[:, 0], 1.0, average=0).coherence, x[:, 0]
        )

    @needs_jax
    def test_gradient_where_another_bin_is_undefined(self) -> None:
        # a constant has power in bin 0 only
        y = noise(49, 4, 8)

        def of(x: Any) -> Any:
            return coherence(x, y, 1.0, average=0, window=None, detrend=None).coherence

        assert np.isnan(of(np.ones((4, 8)))[1:]).all()
        assert_finite_gradient(of, np.ones((4, 8)))

    def test_errors(self) -> None:
        x = noise(50, 3, 12)
        for options in ({}, {"segment": 12}, {"segment": 8, "overlap": 0.0}):
            with pytest.raises(ValidationError, match="at least 2 transforms"):
                coherence(x, x, 1.0, **options)
        with pytest.raises(ValidationError, match="at least 2 transforms"):
            coherence(x[:1], x[:1], 1.0, average=0)
        for average in (1, (0, 0), (0, -2)):
            with pytest.raises(ValidationError, match="'average' must not"):
                coherence(x, x, 1.0, average=average)
        with pytest.raises(ValidationError, match="no axis 2"):
            coherence(x, x, 1.0, average=2)


class TestRotarySpectrum:
    @pytest.mark.parametrize("n", [2, 15, 16])
    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("detrend", ["constant", "linear", None])
    def test_parseval(self, n: int, window: str | None, detrend: Any) -> None:
        u, v = noise(51, 2, 3, n)
        out = rotary_spectrum(u, v, 0.5, window=window, detrend=detrend)
        w = np.ones(n) if window is None else hann(n)
        kinds = {"constant": "constant", "linear": "linear"}
        bare = [
            a if detrend is None else signal.detrend(a, type=kinds[detrend])
            for a in (u, v)
        ]
        expected = sum(np.mean((w * a) ** 2, axis=-1) for a in bare) / np.mean(w**2)
        both = out.counter_clockwise + out.clockwise
        total = np.sum(both, axis=-1) * step_of(out.wavenumber)
        assert_matches_oracle(total, expected, rtol=1e-12, atol=1e-25)

    def test_the_transform_of_the_complex_series(self) -> None:
        for n in (12, 13):
            u, v = noise(52, 2, n)
            power = np.abs(np.fft.fft(u + 1j * v)) ** 2 * 0.5 / n
            out = rotary_spectrum(u, v, 0.5, detrend=None)
            inner = slice(1, (n + 1) // 2)
            assert_matches_oracle(
                out.counter_clockwise[inner], power[inner], rtol=1e-12
            )
            backwards = np.roll(power[::-1], 1)
            assert_matches_oracle(out.clockwise[inner], backwards[inner], rtol=1e-12)
            half = 0.5 * power[0]
            assert out.clockwise[0] == pytest.approx(half, rel=1e-12)
            assert out.counter_clockwise[0] == pytest.approx(half, rel=1e-12)
            if n % 2 == 0:
                assert out.clockwise[-1] == pytest.approx(power[n // 2], rel=1e-12)
                assert out.counter_clockwise[-1] == 0.0

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_a_rotation_is_in_one_half(self, sign: float) -> None:
        angle = 2.0 * np.pi * 3.0 * np.arange(20) / 20.0
        u, v = 2.0 * np.cos(angle), sign * 2.0 * np.sin(angle)
        out = rotary_spectrum(u, v, 0.25)
        expected = np.zeros(11)
        expected[3] = 4.0
        halves = (out.counter_clockwise, out.clockwise)
        this, other = halves if sign > 0.0 else halves[::-1]
        step = step_of(out.wavenumber)
        assert_matches_oracle(this * step, expected, rtol=0.0, atol=1e-13)
        assert_matches_oracle(other * step, 0.0 * expected, rtol=0.0, atol=1e-13)

    def test_stored_xrops(self) -> None:
        stored = xrops_values("spectra")
        out = rotary_spectrum(
            stored["series_u"], stored["series_v"], float(stored["series_spacing"])
        )
        assert_matches_oracle(
            out.wavenumber[1:], stored["rotary_wavenumber"], rtol=1e-12
        )
        for part, name in (
            (out.counter_clockwise, "rotary_psd_ccw"),
            (out.clockwise, "rotary_psd_cw"),
        ):
            assert_matches_oracle(part[:, 1:], stored[name], rtol=1e-10, atol=1e-13)

    def test_axis_and_broadcasting(self) -> None:
        u, v = noise(53, 12, 3), noise(54, 12, 1)
        out = rotary_spectrum(u, v, 1.0, axis=0)
        one = rotary_spectrum(u[:, 2], v[:, 0], 1.0)
        assert out.clockwise.shape == (3, 7)
        assert_matches_oracle(out.clockwise[2], one.clockwise, rtol=1e-12, atol=1e-28)

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [9, 10])
    def test_backends_and_precision(self, n: int) -> None:
        def of(u: Any, v: Any) -> Any:
            out = rotary_spectrum(u, v, 0.3)
            return out.wavenumber, 1.0 + out.counter_clockwise, 1.0 + out.clockwise

        u, v = noise(55, 2, 2, n)
        assert_backends_agree(of, u, v)
        assert_traces(of, u[None], v[None])
        assert_refuses_dataarrays(of, u, v)
        assert_grad_matches_fd(lambda u: of(u, v[0])[1], u[0])

    def test_errors(self) -> None:
        x = noise(56, 8)
        with pytest.raises(ValidationError, match="spacing"):
            rotary_spectrum(x, x, -1.0)
        with pytest.raises(ValidationError, match="no axis"):
            rotary_spectrum(x, x, 1.0, axis=1)
        with pytest.raises(ValidationError, match="'detrend'"):
            rotary_spectrum(x, x, 1.0, detrend="mean")


FLUXES = {"energy": kinetic_energy_flux, "enstrophy": enstrophy_flux}
BOXES = {
    "even": ((16, 16), (1.0, 1.0)),
    "odd": ((9, 11), (0.5, 0.5)),
    "oblong": ((12, 7), (0.5, 2.0)),
}


def coefficients(x: Any) -> np.ndarray:
    return np.fft.fft2(x) / x.shape[-1] / x.shape[-2]


def angular(shape: Any, spacing: Any) -> tuple[np.ndarray, np.ndarray]:
    ky, kx = (
        2.0 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(shape, spacing, strict=True)
    )
    return ky[:, None], kx[None, :]


def below_nyquist(shape: Any) -> np.ndarray:
    """The modes that have an odd derivative."""
    rows, columns = (np.fft.fftfreq(n, 1.0 / n) for n in shape)
    half_y, half_x = (n / 2 for n in shape)
    return (np.abs(rows)[:, None] < half_y) & (np.abs(columns)[None, :] < half_x)


def stirred(seed: int, shape: Any, spacing: Any) -> tuple[np.ndarray, np.ndarray]:
    """A periodic non-divergent velocity with every mode below the Nyquist ones."""
    ky, kx = angular(shape, spacing)
    psi_hat = np.where(below_nyquist(shape), np.fft.fft2(noise(seed, *shape)), 0.0)
    u = np.fft.ifft2(-1j * ky * psi_hat).real
    v = np.fft.ifft2(1j * kx * psi_hat).real
    return u, v


def triads(u: Any, v: Any, advected: Any, spacing: Any) -> np.ndarray:
    """T of each mode, −Re ∑ q̂(k)* (û(p)·i(k − p)) q̂(k − p), mode by mode."""
    shape = u.shape
    ky, kx = angular(shape, spacing)
    rows, columns = (np.fft.fftfreq(n, 1.0 / n).astype(int) for n in shape)
    u_hat, v_hat = coefficients(u), coefficients(v)
    below = below_nyquist(shape)
    modes = [(j, i) for j, i in np.ndindex(*shape) if below[j, i]]
    where = {(rows[j], columns[i]): (j, i) for j, i in modes}
    out = np.zeros(shape)
    for k in modes:
        for p in modes:
            q = where.get((rows[k[0]] - rows[p[0]], columns[k[1]] - columns[p[1]]))
            if q is None:
                continue
            speed = u_hat[p] * 1j * kx[0, q[1]] + v_hat[p] * 1j * ky[q[0], 0]
            for c_hat in advected:
                out[k] -= (np.conj(c_hat[k]) * speed * c_hat[q]).real
    return out


def vorticity(u: Any, v: Any, spacing: Any) -> np.ndarray:
    ky, kx = angular(u.shape, spacing)
    return 1j * kx * coefficients(v) - 1j * ky * coefficients(u)


def advected_by(name: str, u: Any, v: Any, spacing: Any) -> tuple[Any, ...]:
    if name == "enstrophy":
        return (vorticity(u, v, spacing),)
    return coefficients(u), coefficients(v)


@pytest.mark.parametrize("name", FLUXES)
class TestFluxes:
    @pytest.mark.parametrize(("shape", "spacing"), BOXES.values(), ids=BOXES)
    def test_the_transfer_sums_to_zero(
        self, name: str, shape: Any, spacing: Any
    ) -> None:
        # 3 to 14 % of the absolute sum without the finer grid
        u, v = stirred(60, shape, spacing)
        out = FLUXES[name](u, v, spacing)
        scale = np.sum(np.abs(out.transfer))
        assert abs(np.sum(out.transfer)) < 1e-13 * scale
        assert abs(out.flux[0]) < 1e-13 * scale
        # Π(kₙ) − Π(kₙ₊₁) = T(kₙ)
        ends = np.append(out.flux, 0.0)
        assert_matches_oracle(
            ends[:-1] - ends[1:], out.transfer, rtol=0.0, atol=1e-13 * scale
        )

    @pytest.mark.parametrize("shape", [(6, 7), (7, 6)])
    def test_the_sum_over_triads(self, name: str, shape: Any) -> None:
        spacing = (0.5, 0.8)
        u, v = stirred(61, shape, spacing)
        expected = triads(u, v, advected_by(name, u, v, spacing), spacing)
        ky, kx = angular(shape, spacing)
        out = FLUXES[name](u, v, spacing)
        step = 2.0 * np.pi * step_of(out.wavenumber)
        ring = np.rint(np.hypot(ky, kx) / step).astype(int)
        by_ring = np.bincount(ring.ravel(), expected.ravel(), out.transfer.size)
        scale = float(np.max(np.abs(by_ring)))
        assert_matches_oracle(out.transfer, by_ring, rtol=0.0, atol=1e-13 * scale)

    def test_a_divergent_velocity_too(self, name: str) -> None:
        spacing = (0.5, 0.8)
        u, v = noise(62, 2, 5, 7)
        expected = triads(u, v, advected_by(name, u, v, spacing), spacing)
        out = FLUXES[name](u, v, spacing)
        total = np.sum(expected)
        assert np.sum(out.transfer) == pytest.approx(total, rel=1e-11)
        assert abs(total) > 1e-3 * np.sum(np.abs(expected))

    def test_the_rings_are_those_of_the_isotropic_spectrum(self, name: str) -> None:
        u, v = stirred(63, (12, 7), (0.5, 2.0))
        out = FLUXES[name](u, v, (0.5, 2.0))
        spectrum = isotropic_spectrum(u, (0.5, 2.0))
        assert_matches_oracle(out.wavenumber, spectrum.wavenumber, rtol=0.0)
        assert out.transfer.shape == spectrum.power.shape

    def test_a_finer_grid_changes_nothing(self, name: str) -> None:
        y, x = 2.0 * np.pi * np.mgrid[:8, :8] / 8.0
        fine_y, fine_x = 2.0 * np.pi * np.mgrid[:20, :20] / 20.0

        def flow(y: Any, x: Any) -> tuple[Any, Any]:
            u = np.sin(y) + np.sin(x + y) + 2.0 * np.sin(x + 2.0 * y + 1.0)
            return u, -np.sin(x + y) - np.sin(x + 2.0 * y + 1.0)

        coarse = FLUXES[name](*flow(y, x), 2.0 * np.pi / 8.0)
        fine = FLUXES[name](*flow(fine_y, fine_x), 2.0 * np.pi / 20.0)
        n = coarse.transfer.size
        assert_matches_oracle(fine.transfer[:n], coarse.transfer, rtol=0, atol=1e-13)

    def test_the_nyquist_modes_are_left_out(self, name: str) -> None:
        u, v = stirred(64, (8, 8), (1.0, 1.0))
        bare = FLUXES[name](u, v, 1.0).transfer
        checks = 3.0 * (-1.0) ** np.arange(8)
        more = FLUXES[name](u + checks, v + checks[:, None], 1.0).transfer
        assert_matches_oracle(more, bare, rtol=0.0, atol=1e-11)

    def test_a_taper_and_a_trend(self, name: str) -> None:
        u, v = noise(65, 2, 9, 10)
        w = hann(9, 10)
        flux = FLUXES[name]
        out = flux(u, v, 1.0, window="hann", detrend="constant")
        u, v = (w * (a - a.mean()) for a in (u, v))
        assert_matches_oracle(out.transfer, flux(u, v, 1.0).transfer, rtol=1e-10)

    @pytest.mark.slow
    def test_backends_and_precision(self, name: str) -> None:
        def of(u: Any, v: Any) -> Any:
            out = FLUXES[name](u, v, (0.5, 0.75))
            # relative to about the largest, as they cross zero
            return out.wavenumber, 100.0 + out.transfer, 100.0 + out.flux

        u, v = np.stack([stirred(s, (6, 7), (0.5, 0.75)) for s in (66, 67)], axis=1)
        assert_backends_agree(of, u, v)
        assert_traces(of, u[None], v[None])
        assert_refuses_dataarrays(of, u, v)
        assert_grad_matches_fd(lambda u: of(u, v[0])[2], u[0])

    def test_errors(self, name: str) -> None:
        with pytest.raises(ValidationError, match="at least 2 samples"):
            FLUXES[name](np.ones(8), np.ones(8), 1.0)
        with pytest.raises(ValidationError, match="spacing"):
            FLUXES[name](np.ones((4, 4)), np.ones((4, 4)), (1.0, -1.0))


def power_law(slope: float, n: int = 40) -> Spectrum:
    k = 1e-6 * np.arange(n)
    power = np.zeros(n)
    power[1:] = 3.0 * k[1:] ** slope
    return Spectrum(wavenumber=k, power=power)


class TestSpectralSlope:
    @pytest.mark.parametrize("slope", [-3.0, -5.0 / 3.0, 0.0, 1.0])
    def test_a_power_law(self, slope: float) -> None:
        out = spectral_slope(power_law(slope), 2e-6, 3e-5)
        assert out == pytest.approx(slope, abs=1e-12)

    def test_only_the_band_counts(self) -> None:
        spectrum = power_law(-3.0)
        k = spectrum.wavenumber
        broken = np.where((k < 4.5e-6) | (k > 2.05e-5), 7.0, spectrum.power)
        out = spectral_slope(Spectrum(k, broken), 4.5e-6, 2.05e-5)
        assert out == pytest.approx(-3.0, abs=1e-12)

    def test_least_squares(self) -> None:
        k = np.arange(1.0, 20.0)
        power = np.exp(noise(70, 3, 19)) * k**-2.0
        out = spectral_slope(Spectrum(k, power), 3.0, 15.0)
        band = slice(2, 15)
        for got, one in zip(out, power, strict=True):
            fit = np.polyfit(np.log(k[band]), np.log(one[band]), 1)
            assert got == pytest.approx(fit[0], rel=1e-12)

    def test_undefined(self) -> None:
        spectrum = power_law(-3.0)
        # one bin, none, and a bin without power
        assert np.isnan(spectral_slope(spectrum, 4.9e-6, 5.1e-6))
        assert np.isnan(spectral_slope(spectrum, 1.0, 2.0))
        power = spectrum.power.copy()
        power[6] = 0.0
        assert np.isnan(spectral_slope(Spectrum(spectrum.wavenumber, power), 0, 1e-5))
        # k = 0 is never in a band
        assert np.isfinite(spectral_slope(spectrum, 0.0, 1e-5))

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(k: Any, power: Any) -> Any:
            return spectral_slope(Spectrum(k, power), 0.15, 0.75)

        k = 0.1 * np.arange(10)
        power = np.exp(noise(71, 2, 10)) / (0.1 + k) ** 3
        assert_backends_agree(of, k, power, rtol32=1e-4)
        assert_traces(of, k[None], power[None])
        assert_refuses_dataarrays(of, k, power)
        assert_grad_matches_fd(lambda power: of(k, power), power)
        # a batch with a bin without power does not spoil the other one
        power[0, 4] = 0.0
        where = np.arange(2)[:, None] == 1
        assert_finite_gradient(lambda power: of(k, power), power, where=where)

    @pytest.mark.slow
    @needs_jax
    def test_the_slope_of_a_field_inside_a_loss(self) -> None:
        import jax

        def of(x: Any) -> Any:
            return spectral_slope(isotropic_spectrum(x, 0.5), 0.2, 0.9)

        x = noise(72, 12, 12)
        assert_matches_oracle(jax.jit(of)(x), of(x), rtol=1e-10)
        assert np.isfinite(jax.grad(of)(jax.numpy.asarray(x))).all()


class TestIntegralScale:
    def test_one_bin(self) -> None:
        k = 0.1 * np.arange(6)
        power = np.where(np.arange(6) == 4, 2.0, 0.0)
        for moment in (1, 2, 3):
            out = integral_scale(Spectrum(k, power), moment=moment)
            assert out == pytest.approx(2.5, rel=1e-12)

    def test_a_gaussian(self) -> None:
        # P = exp(−k²/2σ²) on a one-sided axis: mean k is σ √(2/π), mean k² is σ²
        sigma = 0.3
        k = np.linspace(0.0, 6.0, 60001)
        spectrum = Spectrum(k, np.exp(-0.5 * (k / sigma) ** 2))
        first = integral_scale(spectrum)
        assert first == pytest.approx(np.sqrt(np.pi / 2.0) / sigma, rel=2e-4)
        second = integral_scale(spectrum, moment=2)
        assert second == pytest.approx(1.0 / sigma, rel=2e-4)

    def test_undefined(self) -> None:
        k = 0.1 * np.arange(4)
        power = np.zeros((2, 4))
        power[0, 0] = 1.0
        assert np.isnan(integral_scale(Spectrum(k, power))).all()
        with pytest.raises(ValidationError, match="'moment'"):
            integral_scale(Spectrum(k, power), moment=0)

    @pytest.mark.slow
    @pytest.mark.parametrize("moment", [1, 2])
    def test_backends_and_precision(self, moment: int) -> None:
        def of(k: Any, power: Any) -> Any:
            return integral_scale(Spectrum(k, power), moment=moment)

        k, power = 0.1 * np.arange(7), 1.0 + noise(73, 2, 7) ** 2
        assert_backends_agree(of, k, power)
        assert_traces(of, k[None], power[None])
        assert_refuses_dataarrays(of, k, power)
        assert_grad_matches_fd(lambda power: of(k, power), power)


MOTHERS = {
    "morlet": ("morlet", 6.0),
    "morlet 8": ("morlet", 8.0),
    "paul": ("paul", 4.0),
    "paul 2": ("paul", 2.0),
    "hat": ("dog", 2.0),
    "dog 1": ("dog", 1.0),
    "dog 6": ("dog", 6.0),
}


def mother_in_space(name: str, param: float, eta: np.ndarray) -> np.ndarray:
    """ψ₀(η) of Torrence and Compo's Table 1, without the iᵐ of a Paul."""
    if name == "morlet":
        return np.pi**-0.25 * np.exp(1j * param * eta - 0.5 * eta**2)
    m = int(param)
    if name == "paul":
        norm = 2.0**m * math.factorial(m)
        norm /= np.sqrt(np.pi * math.factorial(2 * m))
        return norm * (1.0 - 1j * eta) ** -(m + 1.0)
    # (−1)^(m+1) dᵐ/dηᵐ exp(−η²/2) = −Heₘ(η) exp(−η²/2)
    hermite = np.polynomial.hermite_e.hermeval(eta, [0.0] * m + [1.0])
    return -hermite * np.exp(-0.5 * eta**2) / np.sqrt(math.gamma(m + 0.5)) + 0j


def octaves(shortest: float, longest: float, voices: int = 8) -> np.ndarray:
    count = int(np.floor(voices * np.log2(longest / shortest))) + 1
    return shortest * 2.0 ** (np.arange(count) / voices)


@pytest.mark.parametrize(("name", "param"), MOTHERS.values(), ids=MOTHERS)
class TestMothers:
    def test_the_definition_in_space(self, name: str, param: float) -> None:
        # W(s, n) = ∑ xₘ (Δt/s)^½ ψ₀((m − n) Δt/s)*
        n, step = 512, 0.5
        x = noise(80, n)
        scales = step * np.asarray([12.0, 16.0, 20.0])
        out = wavelet_transform(x, step, scales, mother=name, param=param, detrend=None)
        lag = step * (np.arange(n)[None, :] - np.arange(n)[:, None])
        # well below the Nyquist frequency, where a wavelet is cut, and above the
        # lags at which a Paul of order m, which falls as (s/Δ)^(m+1), would wrap
        atol = 5e-3 if name == "paul" and param < 4.0 else 2e-6
        for s, got in zip(scales, out.coefficients, strict=True):
            psi = mother_in_space(name, param, lag / s)
            expected = np.sqrt(step / s) * np.conj(psi) @ x
            for part in (np.real, np.imag):
                assert_matches_oracle(part(got), part(expected), rtol=0, atol=atol)

    def test_a_periodic_wave(self, name: str, param: float) -> None:
        n, step, cycles = 60, 0.25, 5
        omega = 2.0 * np.pi * cycles / (n * step)
        x = np.cos(omega * step * np.arange(n))
        scales = np.asarray([0.3, 0.7, 2.0])
        out = wavelet_transform(
            x, step, scales, mother=name, param=param, periodic=True
        )
        # ψ̂₀ as the integral of ψ₀ exp(−iωη) / √(2π)
        eta = np.linspace(-400.0, 400.0, 800001)
        for s, got in zip(scales, out.coefficients, strict=True):
            psi = mother_in_space(name, param, eta)
            wave = np.exp(-1j * s * omega * eta)
            hat = np.trapezoid(psi * wave, eta) / np.sqrt(2.0 * np.pi)
            turn = np.exp(1j * omega * step * np.arange(n))
            expected = 0.5 * np.conj(hat) * turn
            if name == "dog":
                # a real wavelet sees both halves of a cosine
                expected = expected + np.conj(expected)
            expected = expected * np.sqrt(2.0 * np.pi * s / step)
            # a truncated Paul integral, and a Morlet's 1e-8 at ω < 0
            for part in (np.real, np.imag):
                assert_matches_oracle(part(got), part(expected), rtol=0.0, atol=2e-5)

    def test_white_noise_has_its_variance(self, name: str, param: float) -> None:
        x = 1.5 * noise(81, 400, 64)
        out = wavelet_transform(
            x, 2.0, [8.0], mother=name, param=param, detrend=None, periodic=True
        )
        # the middle of each series: neighbours share draws. An energy of 1, and
        # the aliasing of a discrete spectrum: below 2 % at four samples
        power = out.power[:, 0, 32]
        assert np.mean(power) == pytest.approx(1.5**2, rel=0.25)
        assert_mean(power, 1.5**2)

    def test_the_inverse(self, name: str, param: float) -> None:
        step = 0.25
        t = step * np.arange(400)
        x = np.sin(2.0 * np.pi * t / 8.0) * np.exp(-(((t - 45.0) / 15.0) ** 2))
        x = x - x.mean()
        scales = octaves(step / 8.0, 100.0)
        out = wavelet_transform(x, step, scales, mother=name, param=param)
        if name == "dog" and param == 1.0:
            with pytest.raises(ValidationError, match="has no inverse"):
                inverse_wavelet_transform(out)
            return
        back = inverse_wavelet_transform(out)
        # an envelope of one scale: no power beyond these, but for a mean that
        # zero padding makes
        assert_matches_oracle(back, x, rtol=0.0, atol=2e-3)

    def test_the_period_is_where_a_wave_peaks(self, name: str, param: float) -> None:
        step, period = 0.25, 10.0
        n = 400
        x = np.cos(2.0 * np.pi * step * np.arange(n) / period)
        scales = octaves(0.5, 40.0, voices=64)
        out = wavelet_transform(
            x, step, scales, mother=name, param=param, periodic=True
        )
        peak = out.period[np.argmax(np.mean(out.power, axis=-1))]
        assert peak == pytest.approx(period, rel=0.015)


class TestWaveletTransform:
    def test_the_published_constants(self) -> None:
        # Table 2 gives C_δ and ψ₀(0) to three digits
        from diagnostix._src.spectra.mothers import Mother

        for name, published in (
            ("morlet", 0.776 * np.pi**-0.25),
            ("paul", 1.132 * 1.079),
            ("dog", 3.541 * 0.867),
        ):
            constant = Mother.of(name, None).inverse_constant()
            # the hat's is 2 % off: it was summed over the scales of a real series
            assert constant == pytest.approx(published, rel=0.03)
        for m, published in ((4.0, 1.132 * 1.079), (6.0, 0.776 * np.pi**-0.25)):
            name = "paul" if m == 4.0 else "morlet"
            constant = Mother.of(name, m).inverse_constant()
            assert constant == pytest.approx(published, rel=4e-3)

    def test_padding_keeps_the_ends_apart(self) -> None:
        # an event at the end is not felt at the start
        x = np.zeros(64)
        x[-3] = 1.0
        scales = [4.0, 6.0]
        padded = wavelet_transform(x, 1.0, scales, detrend=None)
        wrapped = wavelet_transform(x, 1.0, scales, detrend=None, periodic=True)
        assert np.max(padded.power[:, :3]) < 1e-12
        assert np.max(wrapped.power[:, :3]) > 1e-3

    def test_the_cone(self) -> None:
        scales = [1.0, 2.0, 100.0]
        x = noise(82, 20)
        for name, reach in (("morlet", 2.0**0.5), ("paul", 0.5**0.5)):
            out = wavelet_transform(x, 0.5, scales, mother=name)
            position = 0.5 * np.arange(20)
            distance = np.minimum(position, 9.5 - position)
            expected = distance >= reach * np.asarray(scales)[:, None]
            assert_matches_oracle(out.inside_cone, expected, rtol=0.0)
            assert not out.inside_cone[-1].any()
        out = wavelet_transform(x, 0.5, scales, periodic=True)
        assert out.inside_cone.all()

    @pytest.mark.parametrize("name", ["morlet", "paul", "hat"])
    def test_a_spike_at_an_end_has_fallen_by_e2_at_the_cone(self, name: str) -> None:
        mother = MOTHERS[name][0]
        spike = np.zeros(2048)
        spike[0] = 1.0
        out = wavelet_transform(spike, 1.0, [80.0], mother=mother, detrend=None)
        first = int(np.argmax(out.inside_cone[0]))
        ratio = out.power[0, first] / out.power[0, 0]
        # 1.5⁻⁵ for a Paul of order 4; the next sample is up to 1/80 of s away
        expected = 1.5**-5 if name == "paul" else np.exp(-2.0)
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_axis_and_batch(self) -> None:
        x = noise(83, 2, 24, 3)
        out = wavelet_transform(x, 1.0, [2.0, 4.0], axis=1)
        one = wavelet_transform(x[1, :, 2], 1.0, [2.0, 4.0])
        assert out.coefficients.shape == (2, 3, 2, 24)
        assert_matches_oracle(out.power[1, 2], one.power, rtol=1e-12)

    def test_what_is_removed(self) -> None:
        x, line = noise(84, 40), 5.0 + 0.5 * np.arange(40)
        scales = [2.0, 5.0]
        plain = wavelet_transform(x, 1.0, scales, detrend="linear").power
        with_line = wavelet_transform(x + line, 1.0, scales, detrend="linear").power
        assert_matches_oracle(with_line, plain, rtol=0.0, atol=1e-11)
        same = wavelet_transform(x - x.mean(), 1.0, scales, detrend=None).power
        assert_matches_oracle(
            wavelet_transform(x, 1.0, scales).power, same, rtol=0.0, atol=1e-12
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["morlet", "hat", "dog 1"])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_backends_and_precision(self, name: str, periodic: bool) -> None:
        mother, param = MOTHERS[name]

        def of(x: Any) -> Any:
            out = wavelet_transform(
                x, 0.5, [1.0, 2.5], mother=mother, param=param, periodic=periodic
            )
            return *two_parts(out.coefficients), out.power + 1.0

        x = noise(85, 2, 12)
        assert_backends_agree(of, x)
        assert_traces(of, x[None])
        assert_refuses_dataarrays(of, x)
        assert_grad_matches_fd(lambda x: of(x)[2], x[0])

    @needs_jax
    def test_only_the_coefficients_are_traced(self) -> None:
        import jax

        out = wavelet_transform(noise(86, 16), 1.0, [2.0, 4.0])
        leaves, tree = jax.tree.flatten(out)
        assert len(leaves) == 1
        again = jax.tree.unflatten(tree, leaves)
        assert again.scales == (2.0, 4.0)
        # a transform goes into a compiled function, its scales static
        traced = jax.jit(inverse_wavelet_transform)(out)
        assert_matches_oracle(traced, inverse_wavelet_transform(out), rtol=1e-11)

    def test_errors(self) -> None:
        x = noise(87, 16)
        for scales in ([0.0, 1.0], [2.0, 1.0], [1.0, 1.0]):
            with pytest.raises(ValidationError, match="positive and ascending"):
                wavelet_transform(x, 1.0, scales)
        for scales in ([], [[1.0]], [np.nan], 2.0):
            with pytest.raises(ValidationError, match="1-D and finite"):
                wavelet_transform(x, 1.0, scales)
        with pytest.raises(ValidationError, match="'mother' must be one of"):
            wavelet_transform(x, 1.0, [2.0], mother="haar")
        for mother, param in (("paul", 2.5), ("dog", 0.0), ("morlet", np.inf)):
            with pytest.raises(ValidationError, match="'param' of"):
                wavelet_transform(x, 1.0, [2.0], mother=mother, param=param)
        with pytest.raises(ValidationError, match="spacing"):
            wavelet_transform(x, 0.0, [2.0])
        with pytest.raises(ValidationError, match="no axis"):
            wavelet_transform(x, 1.0, [2.0], axis=1)
        with pytest.raises(ValidationError, match="at least 2 samples"):
            wavelet_transform(x[:1], 1.0, [2.0])


class TestInverse:
    def test_a_band_of_two_waves(self) -> None:
        t = 0.5 * np.arange(256)
        slow, fast = np.cos(2.0 * np.pi * t / 32.0), np.cos(2.0 * np.pi * t / 4.0)
        scales = octaves(0.1, 120.0)
        out = wavelet_transform(slow + fast, 0.5, scales, periodic=True)
        kept = inverse_wavelet_transform(out, band=(12.0, 1e9))
        # the tail of each wave in the other's band: exp(−½ (6 − 2.25)²)
        assert_matches_oracle(kept, slow, rtol=0.0, atol=1e-3)
        kept = inverse_wavelet_transform(out, band=(0.0, 12.0))
        assert_matches_oracle(kept, fast, rtol=0.0, atol=1e-3)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any) -> Any:
            out = wavelet_transform(x, 0.5, [1.0, 2.0, 4.0])
            return 9.0 + inverse_wavelet_transform(out)

        x = noise(88, 2, 12)
        assert_backends_agree(of, x)
        assert_traces(of, x[None])
        assert_grad_matches_fd(of, x[0])

    def test_errors(self) -> None:
        x = noise(89, 16)
        for scales in ([2.0], [1.0, 2.0, 3.0]):
            out = wavelet_transform(x, 1.0, scales)
            with pytest.raises(ValidationError, match="geometric sequence"):
                inverse_wavelet_transform(out)
        out = wavelet_transform(x, 1.0, [2.0, 4.0], param=3.0)
        with pytest.raises(ValidationError, match="has no inverse"):
            inverse_wavelet_transform(out)


class TestWaveletSignificance:
    @pytest.mark.parametrize("name", ["morlet", "paul", "hat"])
    @pytest.mark.parametrize("confidence", [0.8, 0.95])
    def test_white_noise_passes_as_often_as_it_should(
        self, name: str, confidence: float
    ) -> None:
        mother, param = MOTHERS[name]
        x = noise(90, 4000, 64)
        out = wavelet_transform(
            x, 1.0, [4.0], mother=mother, param=param, detrend=None, periodic=True
        )
        threshold = wavelet_significance(out, 1.0, confidence=confidence)
        # one coefficient of each series, so that the draws are independent
        above = out.power[:, 0, 10] > threshold[0]
        assert_mean(above, 1.0 - confidence)

    @pytest.mark.parametrize("alpha", [0.7, -0.4])
    def test_red_noise_too(self, alpha: float) -> None:
        n, count = 128, 4000
        drive = noise(91, count, n + 50)
        x = np.zeros_like(drive)
        for i in range(1, n + 50):
            x[:, i] = alpha * x[:, i - 1] + drive[:, i]
        x = x[:, 50:]
        variance = 1.0 / (1.0 - alpha**2)
        out = wavelet_transform(x, 1.0, [8.0], detrend=None)
        threshold = wavelet_significance(out, variance, lag1=alpha)
        # a wavelet is broad, so its mean background is near that at its period
        share = np.mean(out.power[:, 0, 64] > threshold[0])
        assert share == pytest.approx(0.05, abs=0.02)

    def test_the_formula(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        x = noise(92, 2, 32)
        scales = [2.0, 4.0, 8.0]
        for mother, degrees in (("morlet", 2), ("dog", 1)):
            out = wavelet_transform(x, 0.5, scales, mother=mother)
            variance, lag1 = np.asarray([1.0, 4.0]), np.asarray([0.3, 0.6])
            got = wavelet_significance(out, variance, lag1=lag1, confidence=0.9)
            angle = 2.0 * np.pi * 0.5 / out.period
            a = lag1[:, None]
            red = (1.0 - a**2) / (1.0 + a**2 - 2.0 * a * np.cos(angle))
            quantile = stats.chi2.ppf(0.9, degrees) / degrees
            assert_matches_oracle(got, variance[:, None] * red * quantile, rtol=1e-10)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        def of(x: Any, variance: Any, lag1: Any) -> Any:
            out = wavelet_transform(x, 0.5, [1.0, 2.5])
            return wavelet_significance(out, variance, lag1=lag1)

        x, variance, lag1 = noise(93, 2, 12), np.ones(2), np.asarray([0.2, 0.5])
        assert_backends_agree(of, x, variance, lag1)
        assert_traces(of, x[None], variance[None], lag1[None])
        assert_refuses_dataarrays(of, x, variance, lag1)
        assert_grad_matches_fd(lambda lag1: of(x, variance, lag1), lag1)
        # numbers take the precision of the transform
        out = wavelet_transform(x.astype(np.float32), 0.5, [1.0])
        assert wavelet_significance(out, 1.0).dtype == np.float32

    def test_errors(self) -> None:
        out = wavelet_transform(noise(94, 8), 1.0, [2.0])
        for confidence in (0.0, 1.0, 95.0):
            with pytest.raises(ValidationError, match="'confidence'"):
                wavelet_significance(out, 1.0, confidence=confidence)


class TestWaveletPower2D:
    SCALES = octaves(2.2, 30.0)

    def area(self, power: Any, wavenumber: Any) -> Any:
        return -np.trapezoid(power, wavenumber, axis=0)

    @pytest.mark.parametrize("mode", [(8, 0), (0, 8), (5, 6), (-6, 5), (7, -4)])
    def test_a_wave_has_its_variance_everywhere(self, mode: Any) -> None:
        y, x = np.mgrid[:48, :48]
        wave = 2.0 * np.cos(2.0 * np.pi * (mode[0] * y + mode[1] * x) / 48.0)
        out = wavelet_power_2d(wave, 1.0, self.SCALES, periodic=True)
        area = self.area(out.power, out.wavenumber)
        # the trapezoid rule over eight voices
        assert_matches_oracle(area, np.full((48, 48), 2.0), rtol=0.02)
        # the same whatever its direction, but for the length of the wave
        peak = out.wavenumber[np.argmax(out.power[:, 0, 0])]
        assert peak == pytest.approx(np.hypot(*mode) / 48.0, rel=0.08)

    def test_isotropy(self) -> None:
        # three waves of |k| = 5 Δk: (5, 0), (3, 4) and (−4, 3)
        y, x = np.mgrid[:40, :40]
        powers = []
        for j, i in ((5, 0), (3, 4), (-4, 3)):
            wave = np.cos(2.0 * np.pi * (j * y + i * x) / 40.0)
            out = wavelet_power_2d(wave, 0.5, [3.0, 4.0, 5.0], periodic=True)
            powers.append(np.mean(out.power, axis=(-2, -1)))
        assert_matches_oracle(powers[1], powers[0], rtol=1e-5)
        assert_matches_oracle(powers[2], powers[0], rtol=1e-5)

    def test_white_noise_is_the_isotropic_spectrum(self) -> None:
        # a white field's isotropic density is 2π k σ² Δx Δy, and its smoothed one
        # within the wavelet's width of it: k is raised by 1/(8π²) of itself
        x = noise(95, 40, 32, 32)
        scales = [3.0, 4.0, 6.0]
        out = wavelet_power_2d(x, 1.0, scales, periodic=True, detrend=None)
        mean = np.mean(out.power, axis=(-2, -1))
        for k, density in zip(out.wavenumber, mean.T, strict=True):
            assert np.mean(density) == pytest.approx(2.0 * np.pi * k, rel=0.06)

    def test_a_patch_is_found_where_it_is(self) -> None:
        y, x = np.mgrid[:60, :60]
        envelope = np.exp(-((y - 40.0) ** 2 + (x - 18.0) ** 2) / (2.0 * 6.0**2))
        patch = envelope * np.cos(2.0 * np.pi * x / 6.0)
        out = wavelet_power_2d(patch, 1.0, [4.0, 6.0, 9.0])
        scale, row, column = np.unravel_index(np.argmax(out.power), out.power.shape)
        assert scale == 1
        assert abs(row - 40) <= 1
        assert abs(column - 18) <= 1
        assert np.max(out.power[:, :10]) < 1e-3 * np.max(out.power)

    def test_spacings(self) -> None:
        # a field and its transpose, on a grid with two spacings
        x = noise(96, 14, 20)
        out = wavelet_power_2d(x, (2.0, 0.5), [9.0, 12.0])
        swapped = wavelet_power_2d(x.T, (0.5, 2.0), [9.0, 12.0])
        assert_matches_oracle(np.swapaxes(swapped.power, -1, -2), out.power, rtol=1e-9)

    def test_the_cone(self) -> None:
        out = wavelet_power_2d(noise(97, 12, 20), (1.0, 0.5), [1.0, 2.0, 9.0])
        assert out.inside_cone.shape == out.power.shape == (3, 12, 20)
        # √2 from the rows' edges and the columns': rows 2 … 9, columns 3 … 16
        expected = np.zeros((12, 20), dtype=bool)
        expected[2:10, 3:17] = True
        assert_matches_oracle(out.inside_cone[0], expected, rtol=0.0)
        assert not out.inside_cone[-1].any()
        out = wavelet_power_2d(noise(98, 12, 20), 1.0, [9.0], periodic=True)
        assert out.inside_cone.all()

    @pytest.mark.slow
    @pytest.mark.parametrize("periodic", [True, False])
    def test_backends_and_precision(self, periodic: bool) -> None:
        def of(x: Any) -> Any:
            out = wavelet_power_2d(
                x, (0.5, 0.75), [1.5, 2.0], n_angles=4, periodic=periodic
            )
            return out.wavenumber, 1.0 + out.power

        x = noise(99, 2, 6, 5)
        assert_backends_agree(of, x)
        assert_traces(of, x[None])
        assert_refuses_dataarrays(of, x)
        assert_grad_matches_fd(lambda x: of(x)[1], x[0])

    def test_errors(self) -> None:
        x = noise(100, 8, 8)
        with pytest.raises(ValidationError, match="'n_angles'"):
            wavelet_power_2d(x, 1.0, [2.0], n_angles=0)
        with pytest.raises(ValidationError, match="positive and ascending"):
            wavelet_power_2d(x, 1.0, [-2.0])
        with pytest.raises(ValidationError, match="at least 2 samples"):
            wavelet_power_2d(x[0], 1.0, [2.0])
