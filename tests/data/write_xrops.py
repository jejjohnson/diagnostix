"""Write, or check, the xrops values that diagnostix's tests compare against.

xrops imports diagnostix (it will), so it cannot be a test dependency; its
values are stored once, float64, with the commit in each file's name. Only what
xrops gets right is here: the table in jejjohnson/diagnostix#16 says what is
left out and why.

Run it in a geotoolz checkout at that commit, from that checkout's root, with
diagnostix's own root as the output::

    git checkout --detach a8b2f17
    uv run --package xrtoolz-ops --with cyclopts --with loguru \
        python <diagnostix>/tests/data/write_xrops.py --check
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import cyclopts
import numpy as np
import xarray as xr
from loguru import logger


#: The geotoolz commit whose xrops these values come from.
COMMIT = "a8b2f17"

HERE = Path(__file__).parent

type Arrays = dict[str, np.ndarray]


def _smooth(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """A smooth random field: a few cosines of random amplitude and phase."""
    axes = np.meshgrid(*(np.linspace(0.0, 1.0, n) for n in shape), indexing="ij")
    out = np.zeros(shape)
    for _ in range(6):
        wavenumbers = rng.integers(0, 4, size=len(shape))
        phase = 2.0 * np.pi * sum(k * a for k, a in zip(wavenumbers, axes, strict=True))
        out += rng.normal() * np.cos(phase + rng.uniform(0.0, 2.0 * np.pi))
    return out


def kinematics() -> Arrays:
    """Advection, frontogenesis, geostrophic residual, shear and curvature."""
    from xrops import ocn
    from xrops.metrics.physical import geostrophic_balance_error

    rng = np.random.default_rng(0)
    ny, nx = 24, 30
    out: Arrays = {}

    # --- a plane, uniform metres ---
    x, y = 2.0e3 * np.arange(nx), 3.0e3 * np.arange(ny)
    u, v = 0.3 * _smooth(rng, ny, nx), 0.3 * _smooth(rng, ny, nx)
    scalar = 290.0 + _smooth(rng, ny, nx)
    height = 0.1 * _smooth(rng, ny, nx)
    coriolis = 1.0e-4
    plane = xr.Dataset(
        {
            name: (("y", "x"), values)
            for name, values in (("u", u), ("v", v), ("c", scalar), ("ssh", height))
        },
        coords={"x": x, "y": y},
    )
    options = {"dims": ("x", "y"), "geometry": "cartesian"}
    out |= {
        "plane_x": x,
        "plane_y": y,
        "plane_u": u,
        "plane_v": v,
        "plane_scalar": scalar,
        "plane_height": height,
        "plane_coriolis": np.asarray(coriolis),
    }
    out["plane_advection"] = ocn.advection(plane, "c", **options)["c_advection"].values
    out["plane_frontogenesis"] = ocn.frontogenesis(plane, "c", **options)[
        "c_frontogenesis"
    ].values
    out["plane_shear_vorticity"] = ocn.shear_vorticity(plane, **options)[
        "vort_shear"
    ].values
    out["plane_curvature_vorticity"] = ocn.curvature_vorticity(plane, **options)[
        "vort_curv"
    ].values
    # a calm point, where both components are 0 by convention
    calm = plane.copy(deep=True)
    calm["u"][5, 7] = calm["v"][5, 7] = 0.0
    out["plane_calm_shear_vorticity"] = ocn.shear_vorticity(calm, **options)[
        "vort_shear"
    ].values
    out["plane_calm_curvature_vorticity"] = ocn.curvature_vorticity(calm, **options)[
        "vort_curv"
    ].values
    residual = geostrophic_balance_error(plane, f=coriolis, **options)
    out["plane_residual_u"] = residual["r_u"].values
    out["plane_residual_v"] = residual["r_v"].values

    # --- a sphere, degrees; scalars only, because xrops differentiates a vector's
    # components as scalars (jejjohnson/geotoolz#749) ---
    lon, lat = np.linspace(-40.0, 18.0, nx), np.linspace(-25.0, 32.5, ny)
    sphere = xr.Dataset(
        {name: (("lat", "lon"), plane[name].values) for name in ("u", "v", "c", "ssh")},
        coords={"lon": lon, "lat": lat},
    )
    out |= {"sphere_lon": lon, "sphere_lat": lat}
    out["sphere_advection"] = ocn.advection(sphere, "c")["c_advection"].values
    out["sphere_coriolis"] = ocn.coriolis_parameter(sphere["lat"]).values
    residual = geostrophic_balance_error(sphere)
    out["sphere_residual_u"] = residual["r_u"].values
    out["sphere_residual_v"] = residual["r_v"].values
    from xrops import grad

    out["gravity"] = np.asarray(grad.GRAVITY)
    return out


def columns() -> Arrays:
    """N², column integrals, the boundary-layer height, and the mixed layer's level."""
    from xrops import atm, ocn

    rng = np.random.default_rng(1)
    out: Arrays = {}

    # --- the ocean: uniform depths, wet columns only ---
    depth = 5.0 * np.arange(30)
    density = 1025.0 + np.cumsum(0.02 * rng.random((6, depth.size)), axis=-1)
    ocean = xr.Dataset({"rho": (("column", "depth"), density)}, coords={"depth": depth})
    out |= {"ocean_depth": depth, "ocean_density": density}
    out["ocean_n_squared"] = ocn.brunt_vaisala_frequency(ocean, rho0=1025.0, g=9.80665)[
        "n_squared"
    ].values
    # xrops gives a level, with no interpolation: an answer within one spacing
    out["ocean_mixed_layer_level"] = ocn.mixed_layer_depth(ocean)["mld"].values

    # --- the atmosphere: uneven heights, shared and per column ---
    height = np.sort(rng.uniform(10.0, 3000.0, size=(6, 25)), axis=-1)
    theta_v = 295.0 + 0.004 * height + 0.2 * rng.random(height.shape)
    wind_u = 5.0 + 0.004 * height + 0.5 * rng.random(height.shape)
    wind_v = 0.001 * height + 0.5 * rng.random(height.shape)
    # a calm level, with no shear against the surface
    wind_u[2, 3], wind_v[2, 3] = wind_u[2, 0], wind_v[2, 0]
    # a column that never reaches the critical number: neutral
    theta_v[5] = 295.0
    air = xr.Dataset(
        {
            "theta_v": (("column", "level"), theta_v),
            "u": (("column", "level"), wind_u),
            "v": (("column", "level"), wind_v),
            "z_agl": (("column", "level"), height),
        }
    )
    out |= {
        "air_height": height,
        "air_theta_v": theta_v,
        "air_u": wind_u,
        "air_v": wind_v,
    }
    out["air_boundary_layer_height"] = atm.pbl_height_bulk_richardson(
        air, theta_v="theta_v"
    )["pbl_height"].values
    shared = xr.DataArray(height[0], dims="level")
    for method in ("trapezoid", "sum"):
        out[f"air_integral_per_column_{method}"] = atm.column_integral(
            air["theta_v"], dim="level", coord=air["z_agl"], method=method
        ).values
        out[f"air_integral_shared_{method}"] = atm.column_integral(
            air["theta_v"], dim="level", coord=shared, method=method
        ).values
    return out


def climate() -> Arrays:
    """The monthly climatology and its anomaly."""
    from xrops.anomaly import calculate_anomaly, calculate_climatology

    rng = np.random.default_rng(2)
    # four years, one of them a leap year, sampled every three days
    time = np.arange("2019-01-01", "2023-01-01", 3, dtype="datetime64[D]")
    values = rng.standard_normal((3, time.size))
    months = time.astype("datetime64[M]").astype(int) % 12 + 1
    data = xr.DataArray(
        values,
        dims=("cell", "time"),
        coords={"time": time.astype("datetime64[ns]")},
    )
    monthly = calculate_climatology(data, freq="month").transpose("cell", "month")
    anomaly = calculate_anomaly(data, freq="month").transpose("cell", "time")
    return {
        "values": values,
        "month": months.astype(np.int64),
        "monthly_climatology": monthly.values,
        "monthly_anomaly": anomaly.values,
    }


def spectra() -> Arrays:
    """Power spectra and the rotary spectrum.

    The spectral fluxes are left out: xrops' inverse transform rolls the fields
    by half the domain, which flips the sign of the transfer at every mode whose
    wavenumbers add up to an odd number (jejjohnson/geotoolz#780).
    """
    from xrops.transforms import power_spectrum, rotary_spectrum

    rng = np.random.default_rng(3)
    out: Arrays = {}
    untapered = {"window": None, "window_correction": False, "detrend": "constant"}

    # --- one dimension ---
    spacing = 0.5
    position = spacing * np.arange(60)
    u, v = rng.standard_normal((2, 4, position.size))
    coords = {"x": position}
    series_u = xr.DataArray(u, dims=("sample", "x"), coords=coords, name="u")
    series_v = xr.DataArray(v, dims=("sample", "x"), coords=coords, name="v")
    out |= {"series_spacing": np.asarray(spacing), "series_u": u, "series_v": v}
    for window in (None, "hann"):
        psd = power_spectrum(
            series_u,
            "x",
            window=window,
            window_correction=window is not None,
            detrend="constant",
        )
        out[f"series_psd_{window or 'boxcar'}"] = psd.values
    out["series_frequency"] = psd["freq_x"].values
    rotary = rotary_spectrum(series_u, series_v, dim="x")
    out["rotary_wavenumber"] = rotary["wavenumber"].values
    for name in ("psd_ccw", "psd_cw"):
        out[f"rotary_{name}"] = rotary[name].transpose("sample", "wavenumber").values

    # --- a doubly periodic plane ---
    ny, nx = 24, 24
    x, y = 2.0 * np.arange(nx), 2.0 * np.arange(ny)
    angle_x, angle_y = np.meshgrid(
        2.0 * np.pi * np.arange(nx) / nx, 2.0 * np.pi * np.arange(ny) / ny
    )
    field_u, field_v = np.zeros((ny, nx)), np.zeros((ny, nx))
    # non-divergent: u = −∂ψ/∂y, v = ∂ψ/∂x, with every mode up to wavenumber 4
    length = 2.0 * nx
    for k, m in itertools.product(range(5), range(-4, 5)):
        amplitude, shift = rng.normal(), rng.uniform(0.0, 2.0 * np.pi)
        phase = k * angle_x + m * angle_y + shift
        field_u += amplitude * (2.0 * np.pi * m / length) * np.sin(phase)
        field_v -= amplitude * (2.0 * np.pi * k / length) * np.sin(phase)
    plane = {"y": y, "x": x}
    plane_u = xr.DataArray(field_u, dims=("y", "x"), coords=plane, name="u")
    out |= {"plane_spacing": np.asarray(2.0), "plane_u": field_u, "plane_v": field_v}
    psd = power_spectrum(plane_u, ("y", "x"), **untapered)
    out["plane_psd"] = psd.transpose("freq_y", "freq_x").values
    out["plane_frequency"] = psd["freq_x"].values
    return out


def skill() -> Arrays:
    """Pointwise scores without gaps, the PSD score and the resolved scale."""
    from xrops.metrics.pixel import bias, mae, nrmse, nrmse_score, rmse
    from xrops.metrics.spectral import psd_score, resolved_scale

    rng = np.random.default_rng(4)
    out: Arrays = {}

    # --- pointwise, no gaps ---
    reference = 3.0 + rng.standard_normal((5, 40))
    prediction = reference + 0.3 * rng.standard_normal(reference.shape) + 0.05
    dims = ("cell", "time")
    pair = (
        xr.DataArray(prediction, dims=dims),
        xr.DataArray(reference, dims=dims),
    )
    out |= {"prediction": prediction, "reference": reference}
    for name, score in (
        ("rmse", rmse),
        ("mae", mae),
        ("bias", bias),
        ("nrmse_rms", nrmse),
        ("nrmse_std", nrmse_score),
    ):
        out[name] = score(*pair, dim="time").values

    # --- spectral: an error that grows with wavenumber ---
    n, spacing = 60, 10.0
    position = spacing * np.arange(n)
    frequency = np.fft.rfftfreq(n, spacing)
    phases = np.exp(2j * np.pi * rng.random((2, frequency.size)))
    signal = np.fft.irfft(phases[0] * n * 0.1, n)
    error = np.fft.irfft(phases[1] * n * 0.1 * (frequency / frequency[-1]), n)
    truth = xr.DataArray(signal, dims="x", coords={"x": position}, name="truth")
    estimate = (truth + error).rename("estimate")
    score = psd_score(
        estimate,
        truth,
        psd_dims=["x"],
        window=None,
        window_correction=False,
        detrend="constant",
    )["score"]
    out |= {
        "spectral_spacing": np.asarray(spacing),
        "spectral_truth": signal,
        "spectral_prediction": signal + error,
        "spectral_frequency": score["freq_x"].values,
        "spectral_psd_score": score.values,
        "spectral_resolved_scale": np.asarray(resolved_scale(score, "freq_x")),
    }
    return out


FAMILIES: dict[str, Callable[[], Arrays]] = {
    "climate": climate,
    "columns": columns,
    "kinematics": kinematics,
    "skill": skill,
    "spectra": spectra,
}


def _require_commit() -> None:
    """Refuse to run unless the imported xrops is at `COMMIT`, unchanged.

    Raises:
        SystemExit: If the checkout is at another commit, or has edits.
    """
    import xrops

    source = Path(xrops.__file__).parent

    def git(*arguments: str) -> str:
        run = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return run.stdout.strip()

    head = git("rev-parse", "HEAD")
    edits = git("status", "--porcelain", "--untracked-files=no")
    if not head.startswith(COMMIT) or edits:
        logger.error(f"xrops is at {head[:7]} with edits {edits!r}, not at {COMMIT}")
        sys.exit(1)


app = cyclopts.App(help=__doc__)


@app.default
def main(*, check: bool = False) -> None:
    """Write every family's file, or check that the stored ones are reproduced.

    Args:
        check: Compare with the stored files instead of writing them.
    """
    _require_commit()
    for family, values in FAMILIES.items():
        arrays = {name: np.asarray(array) for name, array in sorted(values().items())}
        path = HERE / f"xrops_{family}_{COMMIT}.npz"
        if not check:
            np.savez_compressed(path, **arrays)
            logger.info(f"{path.name}: {len(arrays)} arrays, {path.stat().st_size} B")
            continue
        with np.load(path) as stored:
            if sorted(stored.files) != sorted(arrays):
                logger.error(f"{path.name}: the names differ")
                sys.exit(1)
            for name, array in arrays.items():
                np.testing.assert_allclose(array, stored[name], rtol=1e-13, atol=0.0)
        logger.info(f"{path.name}: reproduced")


if __name__ == "__main__":
    app()
