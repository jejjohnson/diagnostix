"""Climatologies."""

from __future__ import annotations

import datetime
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
from _oracles import xrops_values
from hypothesis import given, strategies as st

from diagnostix import ValidationError
from diagnostix._src.climate.window import window_sum
from diagnostix.climate import (
    Climatology,
    calendar_day,
    climatology,
    harmonic_climatology,
    smooth_climatology,
)


NAN = np.nan


def days(start: str, stop: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The year, month and day of every date in a half-open range."""
    dates = np.arange(start, stop, dtype="datetime64[D]").astype(datetime.date)
    parts = np.asarray([(date.year, date.month, date.day) for date in dates])
    return parts[:, 0], parts[:, 1], parts[:, 2]


def cycle(day: Any, period: float = 366.0) -> np.ndarray:
    """A mean and three harmonics."""
    angle = 2.0 * np.pi * np.asarray(day) / period
    return (
        10.0
        + 8.0 * np.cos(angle)
        - 3.0 * np.sin(angle)
        + 1.5 * np.sin(2.0 * angle)
        + 0.7 * np.cos(3.0 * angle)
    )


class TestCalendarDay:
    def test_every_date_keeps_its_slot(self) -> None:
        # the table of a leap year, read by every year
        _, month, day = days("2024-01-01", "2025-01-01")
        assert_matches_oracle(calendar_day(month, day), np.arange(1, 367), rtol=0.0)
        for year in (2021, 2023, 1900):
            _, month, day = days(f"{year}-01-01", f"{year + 1}-01-01")
            expected = np.delete(np.arange(1, 367), 59)
            assert_matches_oracle(calendar_day(month, day), expected, rtol=0.0)

    def test_the_dates_around_the_leap_day(self) -> None:
        out = calendar_day([2, 2, 3, 12], [28, 29, 1, 31])
        assert out.tolist() == [59, 60, 61, 366]
        assert out.dtype == np.int64

    def test_arguments_broadcast(self) -> None:
        out = calendar_day(np.arange(1, 13)[:, None], np.asarray([1, 15]))
        assert out.shape == (12, 2)
        assert out[2].tolist() == [61, 75]

    @pytest.mark.xarray
    def test_dataarrays_keep_their_labels(self) -> None:
        import xarray as xr

        time = xr.DataArray(
            np.arange("2023-02-27", "2023-03-03", dtype="datetime64[D]").astype(
                "datetime64[ns]"
            ),
            dims="time",
            name="time",
        )
        time = time.assign_coords(time=time)
        out = calendar_day(time.dt.month, time.dt.day)
        assert out.dims == ("time",)
        assert out.values.tolist() == [58, 59, 61, 62]

    @pytest.mark.parametrize(("month", "value"), [(0, "0"), (13, "13")])
    def test_no_such_month(self, month: int, value: str) -> None:
        with pytest.raises(ValidationError, match=f"in 1 to 12, got {value}"):
            calendar_day([1, month], 1)

    @pytest.mark.parametrize(("month", "day"), [(2, 30), (4, 31), (1, 0), (12, 32)])
    def test_no_such_day(self, month: int, day: int) -> None:
        with pytest.raises(ValidationError, match=f"got {day} of month {month}"):
            calendar_day([1, month], [1, day])

    def test_not_integers(self) -> None:
        with pytest.raises(ValidationError, match="'day' must be of integers"):
            calendar_day(1, 1.0)
        with pytest.raises(ValidationError, match="'month' must be of integers"):
            calendar_day([1.0], 1)


class TestClimatology:
    def test_the_first_of_march_has_one_slot(self) -> None:
        # xrops gives 0.75 on one slot and 0.25 on the next (geotoolz#774)
        _, month, day = days("2001-01-01", "2009-01-01")
        x = ((month == 3) & (day == 1)).astype(float)
        out = climatology(x, calendar_day(month, day))
        expected = np.zeros(366)
        expected[60] = 1.0
        assert_matches_oracle(out.mean, expected, rtol=0.0)
        assert out.count[59] == 2 and out.count[60] == 8
        assert out.n_slots == 366

    def test_stored_xrops_values(self) -> None:
        stored = xrops_values("climate")
        out = climatology(stored["values"], stored["month"], n_slots=12)
        assert_matches_oracle(out.mean, stored["monthly_climatology"], rtol=1e-12)

    def test_base_excludes_its_complement(self) -> None:
        year, month, day = days("2001-01-01", "2005-01-01")
        slot = calendar_day(month, day)
        x = np.random.default_rng(0).normal(size=(2, year.size))
        early = year < 2003
        out = climatology(np.where(early, x, 1e9), slot, base=early)
        expected = climatology(x[:, early], slot[early])
        assert_matches_oracle(out.mean, expected.mean, rtol=0.0)
        assert_matches_oracle(out.count, expected.count, rtol=0.0)
        # both years have 365 days
        assert np.isnan(out.mean[:, 59]).all() and (out.count[:, 59] == 0).all()

    def test_an_invalid_sample_changes_nothing(self) -> None:
        rng = np.random.default_rng(1)
        slot = rng.integers(1, 13, size=100)
        x = rng.normal(size=(3, 100))
        valid = rng.random(x.shape) > 0.2
        out = climatology(x, slot, n_slots=12, valid=valid)
        spoiled = np.where(valid, x, NAN)
        again = climatology(spoiled, slot, n_slots=12, valid=valid)
        assert_matches_oracle(again.mean, out.mean, rtol=0.0)
        for row in range(3):
            for month in range(12):
                use = valid[row] & (slot == month + 1)
                assert out.count[row, month] == use.sum()
                expected = np.asarray(x[row, use].mean())
                assert_matches_oracle(
                    out.mean[row, month], expected, rtol=1e-12, atol=1e-15
                )

    def test_a_nan_in_a_valid_sample(self) -> None:
        out = climatology([1.0, NAN, 3.0], [1, 1, 2], n_slots=2)
        assert_matches_oracle(out.mean, np.asarray([NAN, 3.0]), rtol=0.0)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        rng = np.random.default_rng(2)
        slot = rng.integers(1, 5, size=30)
        x = rng.normal(size=(2, 3, 30))
        expected = climatology(x, slot, n_slots=4)
        out = climatology(np.moveaxis(x, -1, axis), slot, n_slots=4, axis=axis)
        assert_matches_oracle(np.moveaxis(out.mean, axis, -1), expected.mean, rtol=0.0)
        assert_matches_oracle(
            np.moveaxis(out.count, axis, -1), expected.count, rtol=0.0
        )

    def test_backends_and_precision(self) -> None:
        rng = np.random.default_rng(3)
        slot = rng.integers(1, 13, size=40)
        x, valid = rng.random((2, 40)), rng.random((2, 40)) > 0.2

        def of(x: Any, valid: Any) -> Any:
            out = climatology(x, slot, n_slots=12, valid=valid)
            return out.mean, out.count

        assert_backends_agree(of, x, valid)

    def test_refuses_dataarrays(self) -> None:
        slot = np.asarray([1, 2, 1])
        assert_refuses_dataarrays(lambda x: climatology(x, slot, n_slots=2), np.ones(3))

    @needs_jax
    def test_the_record_crosses_jit(self) -> None:
        import jax

        slot = np.asarray([1, 2, 1])
        out = jax.jit(lambda x: climatology(x, slot, n_slots=2))(np.arange(3.0))
        assert isinstance(out, Climatology)
        assert out.n_slots == 2
        assert_matches_oracle(out.mean, np.asarray([1.0, 1.0]), rtol=0.0)

    @pytest.mark.parametrize("slot", [[0, 1], [1, 13]])
    def test_slots_out_of_range(self, slot: list[int]) -> None:
        with pytest.raises(ValidationError, match="'slot' must be in 1 to 12, got"):
            climatology(np.ones(2), slot, n_slots=12)

    @pytest.mark.parametrize("slot", [[1.0, 2.0], [[1, 2]], 1])
    def test_slots_that_are_not_integers(self, slot: Any) -> None:
        with pytest.raises(ValidationError, match="'slot' must be a 1-D array of int"):
            climatology(np.ones(2), slot, n_slots=12)

    @pytest.mark.parametrize("base", [[1, 0], [True], np.ones((2, 1), dtype=bool)])
    def test_a_base_that_is_not_one_boolean_per_sample(self, base: Any) -> None:
        with pytest.raises(ValidationError, match="'base' must be boolean, one per"):
            climatology(np.ones(2), [1, 2], base=base)

    def test_no_slots(self) -> None:
        with pytest.raises(ValidationError, match="'n_slots' must be at least 1"):
            climatology(np.ones(1), [1], n_slots=0)

    def test_another_number_of_samples(self) -> None:
        with pytest.raises(ValidationError, match="must hold the 2 samples"):
            climatology(np.ones(3), [1, 2], n_slots=2)


class TestHarmonicClimatology:
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_a_known_cycle_is_recovered(self, with_valid: bool) -> None:
        # eight years with two thirds of the days, and no 29 February at all
        _, month, day = days("2001-01-01", "2009-01-01")
        slot = calendar_day(month, day)
        rng = np.random.default_rng(0)
        keep = (rng.random(slot.size) > 0.35) & (slot != 60)
        slot = slot[keep]
        x = np.stack([cycle(slot), 2.0 * cycle(slot) - 5.0])
        options: dict[str, Any] = {}
        if with_valid:
            valid = rng.random(x.shape) > 0.1
            x = np.where(valid, x, NAN)
            options["valid"] = valid
        out = harmonic_climatology(x, slot, **options)
        full = cycle(np.arange(1, 367))
        expected = np.stack([full, 2.0 * full - 5.0])
        assert_matches_oracle(out.mean, expected, rtol=0.0, atol=1e-12)
        assert (out.count[:, 59] == 0).all()
        expected = climatology(x, slot, **options)
        assert_matches_oracle(out.count, expected.count, rtol=0.0)

    def test_the_two_fits_agree(self) -> None:
        rng = np.random.default_rng(1)
        slot = rng.integers(1, 13, size=200)
        x = rng.normal(size=(4, 200))
        shared = harmonic_climatology(x, slot, n_slots=12, n_harmonics=2)
        each = harmonic_climatology(
            x, slot, n_slots=12, n_harmonics=2, valid=np.ones(x.shape, dtype=bool)
        )
        assert_matches_oracle(each.mean, shared.mean, rtol=0.0, atol=1e-13)

    def test_it_is_weighted_by_the_counts(self) -> None:
        rng = np.random.default_rng(2)
        slot = rng.integers(1, 9, size=60)
        x = rng.normal(size=60)
        out = harmonic_climatology(x, slot, n_slots=8, n_harmonics=1)
        # least squares over the samples themselves
        angle = 2.0 * np.pi * slot / 8.0
        basis = np.stack([np.ones(60), np.cos(angle), np.sin(angle)], axis=-1)
        a = np.linalg.lstsq(basis, x, rcond=None)[0]
        angle = 2.0 * np.pi * np.arange(1, 9) / 8.0
        expected = a[0] + a[1] * np.cos(angle) + a[2] * np.sin(angle)
        assert_matches_oracle(out.mean, expected, rtol=0.0, atol=1e-13)

    def test_no_harmonics_is_the_mean(self) -> None:
        x = np.asarray([1.0, 2.0, 6.0])
        out = harmonic_climatology(x, [1, 1, 3], n_slots=4, n_harmonics=0)
        assert_matches_oracle(out.mean, np.full(4, 3.0), rtol=1e-14)

    def test_as_many_harmonics_as_fit_interpolate(self) -> None:
        x = np.random.default_rng(3).normal(size=7)
        out = harmonic_climatology(x, np.arange(1, 8), n_slots=7, n_harmonics=3)
        assert_matches_oracle(out.mean, x, rtol=0.0, atol=1e-13)

    def test_a_point_with_too_few_slots_is_nan(self) -> None:
        rng = np.random.default_rng(4)
        slot = np.tile(np.arange(1, 13), 3)
        x = rng.normal(size=(3, 36))
        valid = np.ones(x.shape, dtype=bool)
        valid[0] = False
        valid[1] = slot <= 4
        out = harmonic_climatology(x, slot, n_slots=12, n_harmonics=2, valid=valid)
        assert np.isnan(out.mean).all(axis=-1).tolist() == [True, True, False]
        assert np.isfinite(out.mean[2]).all()

    @pytest.mark.parametrize("axis", [0, 1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        rng = np.random.default_rng(5)
        slot = np.tile(np.arange(1, 13), 3)
        x = rng.normal(size=(2, 4, 36))
        for valid in (None, rng.random(x.shape) > 0.1):
            expected = harmonic_climatology(x, slot, n_slots=12, valid=valid)
            moved = None if valid is None else np.moveaxis(valid, -1, axis)
            out = harmonic_climatology(
                np.moveaxis(x, -1, axis), slot, n_slots=12, axis=axis, valid=moved
            )
            assert_matches_oracle(
                np.moveaxis(out.mean, axis, -1), expected.mean, rtol=1e-12
            )

    def test_float32_on_daily_data(self) -> None:
        # 1.2e-5 relative to amplitudes of order 10, on eight years of days
        _, month, day = days("2001-01-01", "2009-01-01")
        slot = calendar_day(month, day)
        x = cycle(slot).astype(np.float32)
        for valid in (None, np.ones(x.shape, dtype=bool)):
            out = harmonic_climatology(x, slot, valid=valid)
            assert out.mean.dtype == np.float32
            expected = cycle(np.arange(1, 367)).astype(np.float32)
            assert_matches_oracle(out.mean, expected, rtol=0.0, atol=2e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("with_valid", [False, True])
    def test_backends_and_precision(self, with_valid: bool) -> None:
        rng = np.random.default_rng(6)
        slot = np.tile(np.arange(1, 13), 4)
        x = 10.0 + rng.random((2, 48))
        valid = rng.random(x.shape) > 0.1

        def of(x: Any, valid: Any) -> Any:
            out = harmonic_climatology(
                x, slot, n_slots=12, valid=valid if with_valid else None
            )
            return out.mean, out.count

        assert_backends_agree(of, x, valid, rtol=1e-11)

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        rng = np.random.default_rng(7)
        slot = np.tile(np.arange(1, 9), 3)
        x = rng.normal(size=(3, 24))
        valid = rng.random(x.shape) > 0.1
        valid[2] = False

        def of(x: Any, valid: Any) -> Any:
            return harmonic_climatology(
                x, slot, n_slots=8, n_harmonics=1, valid=valid
            ).mean

        assert_traces(of, x, valid, rtol=1e-10)
        assert_grad_matches_fd(lambda x: of(x, valid[:2]), x[:2])
        assert_finite_gradient(lambda x: of(x, valid), np.where(valid, x, NAN))
        assert_grad_matches_fd(lambda x: of(x, None), x)

    def test_refuses_dataarrays(self) -> None:
        slot = np.arange(1, 8)
        assert_refuses_dataarrays(
            lambda x: harmonic_climatology(x, slot, n_slots=7), np.ones(7)
        )

    def test_too_few_slots_with_data(self) -> None:
        with pytest.raises(
            ValidationError, match="5 slots need data for 2 harmonics, got 4"
        ):
            harmonic_climatology(np.ones(4), [1, 2, 3, 4], n_slots=12, n_harmonics=2)

    @pytest.mark.parametrize("n_harmonics", [-1, 6])
    def test_too_many_harmonics(self, n_harmonics: int) -> None:
        with pytest.raises(ValidationError, match="2K \\+ 1 at most 12"):
            harmonic_climatology(
                np.ones(12), np.arange(1, 13), n_slots=12, n_harmonics=n_harmonics
            )


def equal_counts(mean: np.ndarray) -> Climatology:
    return Climatology(
        mean=mean, count=np.full(mean.shape, 8, dtype=np.int32), n_slots=mean.shape[-1]
    )


class TestSmoothClimatology:
    def test_a_constant_is_a_constant(self) -> None:
        out = smooth_climatology(equal_counts(np.full(366, 3.5)), 61)
        assert_matches_oracle(out.mean, np.full(366, 3.5), rtol=1e-14)

    @pytest.mark.parametrize("window", [1, 3, 61, 365])
    def test_a_sinusoid_is_attenuated(self, window: int) -> None:
        # by the Dirichlet kernel sin(W π / D) / (W sin(π / D))
        angle = 2.0 * np.pi * np.arange(1, 367) / 366.0
        out = smooth_climatology(equal_counts(np.cos(angle - 0.4)), window)
        half = np.pi / 366.0
        factor = np.sin(window * half) / (window * np.sin(half))
        expected = factor * np.cos(angle - 0.4)
        assert_matches_oracle(out.mean, expected, rtol=0.0, atol=1e-14)

    def test_it_is_weighted_by_the_counts(self) -> None:
        rng = np.random.default_rng(0)
        mean, count = rng.normal(size=(2, 12)), rng.integers(0, 4, size=(2, 12))
        mean = np.where(count > 0, mean, NAN)
        given_ = Climatology(mean=mean, count=count.astype(np.int32), n_slots=12)
        out = smooth_climatology(given_, 5)
        for row in range(2):
            for slot in range(12):
                near = [(slot + shift) % 12 for shift in range(-2, 3)]
                used = [k for k in near if count[row, k] > 0]
                expected = np.average(mean[row, used], weights=count[row, used])
                assert_matches_oracle(
                    out.mean[row, slot], np.asarray(expected), rtol=1e-12, atol=1e-15
                )
        assert_matches_oracle(out.count, given_.count, rtol=0.0)

    def test_an_empty_window_is_nan(self) -> None:
        out = climatology([1.0], [1], n_slots=7)
        smooth = smooth_climatology(out, 3).mean
        assert np.isnan(smooth).tolist() == [False, False] + [True] * 4 + [False]

    def test_slots_on_another_axis(self) -> None:
        mean = np.random.default_rng(1).normal(size=(3, 12))
        expected = smooth_climatology(equal_counts(mean), 3)
        turned = Climatology(
            mean=mean.T, count=np.full((12, 3), 8, dtype=np.int32), n_slots=12
        )
        out = smooth_climatology(turned, 3, axis=0)
        assert_matches_oracle(out.mean.T, expected.mean, rtol=1e-14)

    def test_backends_and_precision(self) -> None:
        mean = 1.0 + np.random.default_rng(2).random((2, 12))
        count = np.random.default_rng(3).integers(1, 4, size=(2, 12)).astype(np.int32)
        assert_backends_agree(
            lambda mean, count: (
                smooth_climatology(
                    Climatology(mean=mean, count=count, n_slots=12), 5
                ).mean
            ),
            mean,
            count,
        )

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        mean = np.random.default_rng(4).normal(size=(3, 12))
        count = np.ones((3, 12), dtype=np.int32)
        count[:, 3:6] = 0
        mean[:, 3:6] = NAN

        def of(mean: Any, count: Any) -> Any:
            out = Climatology(mean=mean, count=count, n_slots=12)
            return smooth_climatology(out, 3).mean

        assert_traces(of, mean, count)
        assert_finite_gradient(lambda mean: of(mean, count), mean)

    @pytest.mark.parametrize("window", [0, -3, 4])
    def test_a_window_that_is_not_odd(self, window: int) -> None:
        with pytest.raises(ValidationError, match="'window' must be odd and positive"):
            smooth_climatology(equal_counts(np.ones(12)), window)

    def test_a_window_longer_than_the_year(self) -> None:
        with pytest.raises(ValidationError, match="exceed the period of 12, got 13"):
            smooth_climatology(equal_counts(np.ones(12)), 13)

    def test_an_axis_without_the_slots(self) -> None:
        with pytest.raises(ValidationError, match="must hold the 12 slots"):
            smooth_climatology(equal_counts(np.ones((2, 12))), 3, axis=0)


class TestWindowSum:
    @given(
        st.integers(1, 20),
        st.integers(0, 12).map(lambda half: 2 * half + 1),
        st.booleans(),
    )
    def test_against_a_loop(self, n: int, window: int, periodic: bool) -> None:
        if periodic and window > n:
            return
        x = np.arange(1.0, n + 1.0) ** 2
        half = window // 2
        expected = []
        for i in range(n):
            near = range(i - half, i + half + 1)
            if periodic:
                expected.append(sum(x[k % n] for k in near))
            else:
                expected.append(sum(x[k] for k in near if 0 <= k < n))
        # whole numbers, so every sum is exact
        out = window_sum(np, x, window, periodic=periodic)
        assert_matches_oracle(out, np.asarray(expected), rtol=0.0)

    def test_a_nan_spoils_its_windows_only(self) -> None:
        x = np.ones(20)
        x[10] = NAN
        out = window_sum(np, x, 5, periodic=False)
        assert np.flatnonzero(np.isnan(out)).tolist() == [8, 9, 10, 11, 12]
