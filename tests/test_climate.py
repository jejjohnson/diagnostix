"""Climatologies, anomalies, trends and running means."""

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
    Trend,
    add_climatology,
    anomaly,
    calendar_day,
    climatology,
    detrend,
    harmonic_climatology,
    linear_trend,
    running_mean,
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


class TestAnomaly:
    def test_stored_xrops_values(self) -> None:
        stored = xrops_values("climate")
        monthly = climatology(stored["values"], stored["month"], n_slots=12)
        out = anomaly(stored["values"], monthly, stored["month"])
        assert_matches_oracle(out, stored["monthly_anomaly"], rtol=1e-12, atol=1e-14)

    def test_it_is_undone(self) -> None:
        rng = np.random.default_rng(0)
        slot = rng.integers(1, 13, size=80)
        # samples outside the fit too
        x = 5.0 + rng.normal(size=(3, 80))
        base = np.arange(80) < 50
        for cycle_ in (
            climatology(x, slot, n_slots=12, base=base),
            harmonic_climatology(x, slot, n_slots=12, n_harmonics=2, base=base),
        ):
            out = add_climatology(anomaly(x, cycle_, slot), cycle_, slot)
            assert_matches_oracle(out, x, rtol=1e-12)

    def test_a_field_that_is_its_climatology(self) -> None:
        slot = np.tile(np.arange(1, 13), 3)
        x = np.stack([np.sin(slot), np.cos(slot)])
        cycle_ = climatology(x, slot, n_slots=12)
        out = anomaly(x, cycle_, slot)
        assert_matches_oracle(out, np.zeros_like(x), rtol=0.0, atol=1e-15)

    def test_an_empty_slot_is_nan(self) -> None:
        cycle_ = climatology([1.0, 2.0], [1, 2], n_slots=3)
        out = anomaly([1.0, 5.0, 5.0], cycle_, [1, 2, 3])
        assert_matches_oracle(out, np.asarray([0.0, 3.0, NAN]), rtol=0.0)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        rng = np.random.default_rng(1)
        slot = rng.integers(1, 5, size=30)
        x = rng.normal(size=(2, 3, 30))
        cycle_ = climatology(x, slot, n_slots=4)
        expected = anomaly(x, cycle_, slot)
        turned = climatology(np.moveaxis(x, -1, axis), slot, n_slots=4, axis=axis)
        out = anomaly(np.moveaxis(x, -1, axis), turned, slot, axis=axis)
        assert_matches_oracle(np.moveaxis(out, axis, -1), expected, rtol=0.0)

    @pytest.mark.parametrize("fn", [anomaly, add_climatology])
    def test_backends_and_precision(self, fn: Any) -> None:
        slot = np.asarray([1, 2, 3, 1])
        mean = np.asarray([[1.0, 2.0, 4.0], [3.0, 1.0, 1.0]])
        count = np.ones((2, 3), dtype=np.int32)

        def of(x: Any, mean: Any, count: Any) -> Any:
            return fn(x, Climatology(mean=mean, count=count, n_slots=3), slot)

        x = 10.0 + np.random.default_rng(2).random((2, 4))
        assert_backends_agree(of, x, mean, count)

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        slot = np.asarray([1, 2, 3, 1])
        rng = np.random.default_rng(3)
        mean, x = rng.normal(size=(3, 3)), rng.normal(size=(3, 4))
        count = np.ones((3, 3), dtype=np.int32)

        def of(x: Any, mean: Any, count: Any) -> Any:
            return anomaly(x, Climatology(mean=mean, count=count, n_slots=3), slot)

        assert_traces(of, x, mean, count)
        assert_grad_matches_fd(lambda mean: of(x, mean, count), mean)

    def test_refuses_dataarrays(self) -> None:
        cycle_ = climatology([1.0, 2.0], [1, 2], n_slots=2)
        assert_refuses_dataarrays(lambda x: anomaly(x, cycle_, [1, 2, 1]), np.ones(3))

    def test_samples_and_slots_that_differ(self) -> None:
        cycle_ = climatology([1.0, 2.0], [1, 2], n_slots=2)
        with pytest.raises(ValidationError, match="must hold the 2 samples"):
            anomaly(np.ones(3), cycle_, [1, 2])
        with pytest.raises(ValidationError, match="'slot' must be in 1 to 2, got 3"):
            add_climatology(np.ones(2), cycle_, [1, 3])
        with pytest.raises(ValidationError, match="must hold the 2 slots"):
            anomaly(np.ones((2, 2)), cycle_, [1, 2], axis=1)


#: Uneven times, as days.
DAYS = np.cumsum(np.random.default_rng(9).uniform(0.5, 3.0, size=40))


def parts(trend: Trend) -> tuple[Any, ...]:
    return trend.slope, trend.intercept, trend.centre


class TestLinearTrend:
    def test_a_line_with_gaps_and_uneven_times(self) -> None:
        rng = np.random.default_rng(0)
        valid = rng.random((3, DAYS.size)) > 0.3
        slope = np.asarray([0.5, -2.0, 0.0])
        x = np.where(valid, 7.0 + slope[:, None] * DAYS, NAN)
        out = linear_trend(x, DAYS, valid=valid)
        assert_matches_oracle(out.slope, slope, rtol=1e-12, atol=1e-12)
        for row in range(3):
            # the centre is each point's own
            centre = DAYS[valid[row]].mean()
            assert_matches_oracle(out.centre[row], np.asarray(centre), rtol=1e-14)
        assert_matches_oracle(out.intercept, 7.0 + slope * out.centre, rtol=1e-12)

    def test_against_polyfit(self) -> None:
        rng = np.random.default_rng(1)
        x, weights = rng.normal(size=(4, DAYS.size)), rng.uniform(0.1, 2.0, DAYS.size)
        out = linear_trend(x, DAYS, weights=weights)
        for row in range(4):
            # polyfit's weights multiply the residuals before they are squared
            slope, level = np.polyfit(DAYS, x[row], 1, w=np.sqrt(weights))
            assert_matches_oracle(out.slope[row], np.asarray(slope), rtol=1e-11)
            at_zero = out.intercept[row] - out.slope[row] * out.centre[row]
            assert_matches_oracle(at_zero, np.asarray(level), rtol=1e-10)

    def test_the_scale_of_the_weights_cancels(self) -> None:
        x = np.random.default_rng(2).normal(size=DAYS.size)
        weights = np.random.default_rng(3).random(DAYS.size)
        out = parts(linear_trend(x, DAYS, weights=weights))
        again = parts(linear_trend(x, DAYS, weights=1e6 * weights))
        for value, expected in zip(again, out, strict=True):
            assert_matches_oracle(value, expected, rtol=1e-11)

    @pytest.mark.parametrize("on_jax", [False, pytest.param(True, marks=needs_jax)])
    def test_float32_with_days_since_1950(self, on_jax: bool) -> None:
        days_since = 27000.0 + np.arange(3650.0)
        x = (3.0 + 1e-3 * (days_since - 27000.0)).astype(np.float32)
        if on_jax:
            import jax.numpy as jnp

            x = jnp.asarray(x)
        out = linear_trend(x, days_since)
        assert all(part.dtype == np.float32 for part in parts(out))
        assert_matches_oracle(
            np.asarray(out.slope), np.asarray(1e-3, np.float32), rtol=1e-5
        )

    @pytest.mark.parametrize("seconds", [0.0, 1.7e9])
    def test_one_time_has_no_slope(self, seconds: float) -> None:
        time = seconds + np.asarray([0.1, 0.2, 0.7, 0.2])
        valid = np.asarray([[False] * 4, [False, True] + [False] * 2, [True] * 4])
        # two samples at one time
        valid = np.concatenate([valid, [[False, True, False, True]]])
        x = np.arange(4.0) * np.ones((4, 1))
        out = linear_trend(x, time, valid=valid, weights=[0.3, 0.7, 0.1, 0.9])
        assert np.isnan(out.slope).tolist() == [True, True, False, True]
        assert np.isnan(out.intercept).tolist() == [True, False, False, False]
        assert np.isnan(out.centre).tolist() == [True, False, False, False]

    def test_a_weight_of_zero_is_not_used(self) -> None:
        out = linear_trend([1.0, 5.0], [0.0, 1.0], weights=[1.0, 0.0])
        assert np.isnan(out.slope)

    def test_a_nan_in_a_valid_sample(self) -> None:
        x = np.ones((2, DAYS.size))
        x[1, 5] = NAN
        out = linear_trend(x, DAYS)
        assert np.isnan(out.slope).tolist() == [False, True]

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2, 3, DAYS.size))
        valid = rng.random(x.shape) > 0.2
        expected = parts(linear_trend(x, DAYS, valid=valid))
        turned = [np.moveaxis(array, -1, axis) for array in (x, valid)]
        out = parts(linear_trend(turned[0], DAYS, valid=turned[1], axis=axis))
        for value, want in zip(out, expected, strict=True):
            assert_matches_oracle(value, want, rtol=1e-12)

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        rng = np.random.default_rng(5)
        x = 2.0 + 0.5 * DAYS + rng.random((2, DAYS.size))
        weights, valid = rng.uniform(0.5, 1.0, DAYS.size), rng.random(x.shape) > 0.2

        def of(x: Any, weights: Any, valid: Any) -> Any:
            return parts(linear_trend(x, DAYS, weights=weights, valid=valid))

        assert_backends_agree(of, x, weights, valid, rtol=1e-11)
        # times of the library itself
        assert_backends_agree(lambda x, t: parts(linear_trend(x, t)), x, DAYS)

    @needs_jax
    @pytest.mark.slow
    def test_transforms_and_gradients(self) -> None:
        rng = np.random.default_rng(6)
        x, valid = rng.normal(size=(3, 10)), rng.random((3, 10)) > 0.2
        time = np.stack([DAYS[:10]] * 3)
        assert_traces(
            lambda x, t, valid: parts(linear_trend(x, t, valid=valid)),
            x,
            time,
            valid,
            rtol=1e-10,
        )
        assert_grad_matches_fd(lambda x: linear_trend(x, DAYS[:10]).slope, x)
        valid[0] = False
        valid[1] = np.arange(10) == 4
        assert_finite_gradient(
            lambda x: linear_trend(x, DAYS[:10], valid=valid).slope,
            np.where(valid, x, NAN),
        )

    def test_refuses_dataarrays(self) -> None:
        x = np.ones((2, DAYS.size))
        assert_refuses_dataarrays(
            lambda x, t, w: linear_trend(x, t, weights=w).slope, x, DAYS, x
        )

    @pytest.mark.parametrize("time", [DAYS[:-1], DAYS[None], 1.0])
    def test_times_that_are_not_one_per_sample(self, time: Any) -> None:
        with pytest.raises(ValidationError, match="'time' must be one time per sample"):
            linear_trend(np.ones((2, DAYS.size)), time)

    def test_no_such_axis(self) -> None:
        with pytest.raises(ValidationError, match="no axis 1 for the samples"):
            linear_trend(np.ones(3), np.arange(3.0), axis=1)


class TestDetrend:
    def test_it_is_idempotent(self) -> None:
        rng = np.random.default_rng(0)
        x = 100.0 + 3.0 * DAYS + rng.normal(size=(3, DAYS.size))
        weights, valid = rng.random(DAYS.size), rng.random(x.shape) > 0.2
        options = {"weights": weights, "valid": valid}
        once = detrend(x, DAYS, **options)
        twice = detrend(once, DAYS, **options)
        assert_matches_oracle(twice, once, rtol=0.0, atol=1e-12)
        assert_matches_oracle(
            linear_trend(once, DAYS, **options).slope,
            np.zeros(3),
            rtol=0.0,
            atol=1e-12,
        )

    def test_it_is_the_residual_of_the_line(self) -> None:
        x = np.random.default_rng(1).normal(size=(2, DAYS.size))
        out = linear_trend(x, DAYS)
        line = out.intercept[:, None] + out.slope[:, None] * (
            DAYS - out.centre[:, None]
        )
        assert_matches_oracle(detrend(x, DAYS), x - line, rtol=0.0, atol=1e-13)

    def test_float32_with_days_since_1950(self) -> None:
        days_since = 27000.0 + np.arange(3650.0)
        x = (3.0 + 1e-3 * (days_since - 27000.0)).astype(np.float32)
        out = detrend(x, days_since)
        assert out.dtype == np.float32
        # x is rounded to 2.4e-7 up to 4, and to 4.8e-7 above
        assert_matches_oracle(out, np.zeros_like(x), rtol=0.0, atol=3e-6)

    def test_a_point_without_a_trend_is_nan(self) -> None:
        valid = np.asarray([[True, False, False], [True, True, True]])
        out = detrend(np.ones((2, 3)), [0.0, 1.0, 2.0], valid=valid)
        assert np.isnan(out).all(axis=-1).tolist() == [True, False]

    @pytest.mark.parametrize("axis", [0, 1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        x = np.random.default_rng(2).normal(size=(2, 3, DAYS.size))
        out = detrend(np.moveaxis(x, -1, axis), DAYS, axis=axis)
        assert_matches_oracle(
            np.moveaxis(out, axis, -1), detrend(x, DAYS), rtol=0.0, atol=1e-13
        )

    @pytest.mark.slow
    def test_backends_and_precision(self) -> None:
        x = 10.0 + 8.0 * np.random.default_rng(3).random((2, DAYS.size))
        # away from zero, for float32's relative precision
        assert_backends_agree(
            lambda x, c: detrend(x, DAYS) + c, x, np.full(x.shape, 99.0)
        )

    @needs_jax
    def test_gradients(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2, 6))
        assert_grad_matches_fd(lambda x: detrend(x, DAYS[:6]), x)
        valid = np.asarray([[True] * 6, [True] + [False] * 5])
        assert_finite_gradient(
            lambda x: detrend(x, DAYS[:6], valid=valid), np.where(valid, x, NAN)
        )


class TestRunningMean:
    @pytest.mark.parametrize("periodic", [False, True])
    @pytest.mark.parametrize("window", [1, 3, 7, 19])
    def test_against_a_loop_with_gaps(self, window: int, periodic: bool) -> None:
        rng = np.random.default_rng(0)
        x, valid = rng.normal(size=(3, 20)), rng.random((3, 20)) > 0.4
        valid[2, 4:14] = False
        out = running_mean(
            np.where(valid, x, NAN), window, valid=valid, periodic=periodic
        )
        half = window // 2
        for row in range(3):
            for i in range(20):
                near = np.arange(i - half, i + half + 1)
                near = near % 20 if periodic else near[(near >= 0) & (near < 20)]
                used = near[valid[row, near]]
                expected = x[row, used].mean() if used.size else NAN
                assert_matches_oracle(
                    out[row, i], np.asarray(expected), rtol=1e-12, atol=1e-14
                )
        assert window > 10 or np.isnan(out[2]).any()

    def test_a_window_longer_than_the_series_is_truncated(self) -> None:
        x = np.asarray([1.0, 2.0, 6.0])
        assert_matches_oracle(running_mean(x, 11), np.full(3, 3.0), rtol=1e-15)

    def test_a_constant_is_a_constant(self) -> None:
        for periodic in (False, True):
            out = running_mean(np.full(365, 0.1), 31, periodic=periodic)
            assert_matches_oracle(out, np.full(365, 0.1), rtol=1e-14)

    def test_a_nan_in_a_valid_sample(self) -> None:
        x = np.ones(9)
        x[4] = NAN
        out = running_mean(x, 3)
        assert np.flatnonzero(np.isnan(out)).tolist() == [3, 4, 5]

    @pytest.mark.parametrize("axis", [0, 1])
    def test_samples_on_another_axis(self, axis: int) -> None:
        rng = np.random.default_rng(1)
        x, valid = rng.normal(size=(2, 3, 20)), rng.random((2, 3, 20)) > 0.2
        expected = running_mean(x, 5, valid=valid)
        turned = [np.moveaxis(array, -1, axis) for array in (x, valid)]
        out = running_mean(turned[0], 5, valid=turned[1], axis=axis)
        assert_matches_oracle(np.moveaxis(out, axis, -1), expected, rtol=1e-14)

    @pytest.mark.parametrize("periodic", [False, True])
    def test_backends_and_precision(self, periodic: bool) -> None:
        rng = np.random.default_rng(2)
        x, valid = 1.0 + rng.random((2, 12)), rng.random((2, 12)) > 0.2
        assert_backends_agree(
            lambda x, valid: running_mean(x, 5, valid=valid, periodic=periodic),
            x,
            valid,
        )

    @needs_jax
    def test_transforms_and_gradients(self) -> None:
        rng = np.random.default_rng(3)
        x, valid = rng.normal(size=(3, 12)), rng.random((3, 12)) > 0.3
        valid[0, 3:10] = False
        assert_traces(lambda x, valid: running_mean(x, 5, valid=valid), x, valid)
        assert_grad_matches_fd(lambda x: running_mean(x, 5), x)
        assert_finite_gradient(
            lambda x: running_mean(x, 5, valid=valid), np.where(valid, x, NAN)
        )

    def test_refuses_dataarrays(self) -> None:
        assert_refuses_dataarrays(lambda x: running_mean(x, 3), np.ones(5))

    @pytest.mark.parametrize("window", [0, 2, -1])
    def test_a_window_that_is_not_odd(self, window: int) -> None:
        with pytest.raises(ValidationError, match="'window' must be odd and positive"):
            running_mean(np.ones(5), window)

    def test_a_window_longer_than_the_period(self) -> None:
        with pytest.raises(ValidationError, match="exceed the period of 5, got 7"):
            running_mean(np.ones(5), 7, periodic=True)

    def test_no_such_axis(self) -> None:
        with pytest.raises(ValidationError, match="no axis 2 for the samples"):
            running_mean(np.ones(5), 3, axis=2)
