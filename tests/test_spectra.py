"""Spectra: how the variance of a field is spread over scales."""

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
    isotropic_spectrum,
    power_spectrum,
    power_spectrum_2d,
    rotary_spectrum,
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
