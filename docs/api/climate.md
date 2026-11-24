# Climate

A field's annual cycle, and what is left without it.

Samples are grouped by slots of the year, integers 1 … `n_slots`. A daily climatology uses `calendar_day`, a leap-year calendar on which every date has one slot in every year: 29 February is 60 and 1 March is 61, always. Grouped by the day of the year instead, 1 March moves between two slots and every later date with it. Months are slots 1 … 12, and any other labels (seasons, weekdays, the days of a 360-day calendar) are the caller's.

```python
from diagnostix.climate import calendar_day, climatology, harmonic_climatology

day = calendar_day(month, day_of_month)
# x: (T, Ny, Nx); the years before 2020 only, so the score of 2020 is honest
daily = climatology(x, day, axis=0, base=year < 2020)
# a mean and three harmonics, defined on every day
cycle = harmonic_climatology(x, day, axis=0, base=year < 2020)
```

`anomaly` takes a climatology away from each sample and `add_climatology` gives it back, exactly. `linear_trend` and `detrend` centre the times before they square them, so days since 1950 do for float32 samples, and give no significance: that of a correlated series is a modelling choice. `running_mean` has an odd window, skips invalid samples, and is truncated at the ends unless the series wraps.

```python
from diagnostix.climate import add_climatology, anomaly, detrend

# x′ = x − x̄(d), without its trend
departure = detrend(anomaly(x, cycle, day, axis=0), days_since_1950, axis=0)
# back to x
x = add_climatology(anomaly(x, cycle, day, axis=0), cycle, day, axis=0)
```

The grouping is static: slots and the base period are concrete, and under JAX only the samples are traced. `valid` says which samples count (True = use); a NaN in a valid sample gives NaN.

29 February has a quarter of the samples of its neighbours, and a quarter of their weight in the harmonic fit and in `smooth_climatology`.

::: diagnostix.climate
    options:
      show_root_heading: false
      show_source: false
