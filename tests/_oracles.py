"""Reference results: closed forms, stored xrops values and reference libraries.

A closed form needs only NumPy. A helper that calls a library of the ``oracles``
group imports it with ``pytest.importorskip``, so a test that uses it skips,
rather than errors, where it is missing.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest


__all__ = [
    "XROPS_COMMIT",
    "XROPS_FAMILIES",
    "Motion",
    "Patterns",
    "Profile",
    "functax_eofs",
    "linear_pycnocline",
    "metpy_advection",
    "metpy_frontogenesis",
    "neutral_layer_under_inversion",
    "pure_strain",
    "reliable_ensemble",
    "sinusoid",
    "sklearn_pca",
    "solid_body_plane",
    "solid_body_sphere",
    "two_patterns",
    "xrft_power_spectrum",
    "xrops_values",
    "xskillscore_crps",
]


# --- stored xrops values ----------------------------------------------------

#: The geotoolz commit whose xrops values are stored in ``tests/data``.
XROPS_COMMIT = "a8b2f17"

#: The families that have a file there.
XROPS_FAMILIES = ("climate", "columns", "kinematics", "skill", "spectra")


def xrops_values(family: str) -> dict[str, np.ndarray]:
    """The values stored from xrops, by name (see ``write_xrops.py``)."""
    path = Path(__file__).parent / "data" / f"xrops_{family}_{XROPS_COMMIT}.npz"
    with np.load(path) as stored:
        return dict(stored)


# --- closed forms: motion ---------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Motion:
    """A velocity field and what its gradient is, at the same points."""

    u: np.ndarray
    v: np.ndarray
    vorticity: np.ndarray
    divergence: np.ndarray
    okubo_weiss: np.ndarray


def solid_body_plane(x: Any, y: Any, omega: float) -> Motion:
    """Solid-body rotation on a plane: u = −ωy, v = ωx.

    ζ = 2ω, δ = 0, no strain, so the Okubo–Weiss parameter is −4ω².

    Args:
        x, y: Coordinates (m), broadcast against each other.
        omega: Rotation rate ω (s⁻¹).
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    ones = np.ones_like(x)
    return Motion(
        u=-omega * y,
        v=omega * x,
        vorticity=2.0 * omega * ones,
        divergence=0.0 * ones,
        okubo_weiss=-4.0 * omega**2 * ones,
    )


def solid_body_sphere(lon: Any, lat: Any, axis: Any, radius: float) -> Motion:
    """Solid-body rotation on a sphere, about any axis: u⃗ = ω⃗ × r⃗.

    ζ = 2 ω⃗ · r̂, δ = 0, no strain, so the Okubo–Weiss parameter is −ζ².

    Args:
        lon, lat: Coordinates (degrees), broadcast against each other.
        axis: ω⃗, the rotation vector (s⁻¹), Earth-centred: x towards (0°, 0°),
            z towards the north pole.
        radius: Radius of the sphere (m).
    """
    lam, phi = np.broadcast_arrays(np.deg2rad(lon), np.deg2rad(lat))
    # the local east, north and up directions, last axis x, y, z
    zero = np.zeros_like(lam)
    east = np.stack([-np.sin(lam), np.cos(lam), zero], axis=-1)
    north = np.stack(
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)], axis=-1
    )
    up = np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1
    )
    omega = np.asarray(axis, dtype=np.float64)
    velocity = np.cross(np.broadcast_to(omega, up.shape), radius * up)
    # ζ = 2 ω⃗ · r̂
    vorticity = 2.0 * np.sum(omega * up, axis=-1)
    return Motion(
        u=np.sum(velocity * east, axis=-1),
        v=np.sum(velocity * north, axis=-1),
        vorticity=vorticity,
        divergence=np.zeros_like(vorticity),
        okubo_weiss=-(vorticity**2),
    )


def pure_strain(x: Any, y: Any, alpha: float) -> Motion:
    """Pure strain u = (αx, −αy): ζ = 0, δ = 0, Okubo–Weiss 4α².

    Args:
        x, y: Coordinates (m), broadcast against each other.
        alpha: Strain rate α (s⁻¹).
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    zeros = np.zeros_like(x)
    return Motion(
        u=alpha * x,
        v=-alpha * y,
        vorticity=zeros,
        divergence=zeros,
        okubo_weiss=4.0 * alpha**2 * np.ones_like(x),
    )


# --- closed forms: columns --------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Columns on levels (last axis), and the depth or height to find in them."""

    level: np.ndarray
    fields: dict[str, np.ndarray]
    expected: float


def linear_pycnocline(
    depth: Any,
    *,
    top: float = 10.0,
    slope: float = 0.002,
    threshold: float = 0.03,
    surface: float = 1025.0,
) -> Profile:
    """Density uniform above ``top`` and linear below: ρ = ρ₀ + s (z − z_top).

    The density exceeds its value at the top of the gradient by Δρ at depth
    z_top + Δρ/s: 25 m for the defaults.

    Args:
        depth: Levels (m), positive down.
        top: Depth of the top of the gradient z_top (m).
        slope: Density gradient s below it (kg m⁻⁴).
        threshold: Δρ (kg m⁻³).
        surface: Density above ρ₀ (kg m⁻³).
    """
    z = np.asarray(depth, dtype=np.float64)
    density = surface + slope * np.maximum(z - top, 0.0)
    return Profile(z, {"density": density}, top + threshold / slope)


def neutral_layer_under_inversion(
    height: Any,
    *,
    inversion: float = 800.0,
    lapse: float = 0.005,
    shear: float = 0.01,
    theta: float = 300.0,
    gravity: float = 9.80665,
    critical: float = 0.25,
) -> Profile:
    """θ_v uniform below an inversion and linear above it, under a linear wind.

    With the surface level at z₀, u − u_s = s (z − z₀) and θ_v = θ + γ (z − z_i)₊,
    the bulk Richardson number is

        Ri_b(z) = g γ (z − z_i)₊ / (θ s² (z − z₀)).

    The expected height is where the straight line between the two levels that
    bracket the critical number crosses it, as a diagnostic on levels finds it.

    Args:
        height: Levels (m), the surface first.
        inversion: Height of the inversion z_i (m).
        lapse: Gradient γ of θ_v above it (K m⁻¹).
        shear: Wind shear s (s⁻¹).
        theta: θ_v below the inversion (K).
        gravity: g (m s⁻²).
        critical: The critical bulk Richardson number.
    """
    z = np.asarray(height, dtype=np.float64)
    fields = {
        "theta_v": theta + lapse * np.maximum(z - inversion, 0.0),
        "u": 2.0 + shear * (z - z[0]),
        "v": np.full_like(z, 1.0),
    }
    # the surface level has no shear, and is never a crossing
    above = z[1:]
    richardson = np.concatenate(
        [
            [0.0],
            gravity
            * lapse
            * np.maximum(above - inversion, 0.0)
            / (theta * shear**2 * (above - z[0])),
        ]
    )
    upper = int(np.argmax(richardson >= critical))
    if upper == 0:
        raise ValueError("no level reaches the critical number")
    lower = upper - 1
    fraction = (critical - richardson[lower]) / (richardson[upper] - richardson[lower])
    return Profile(z, fields, float(z[lower] + fraction * (z[upper] - z[lower])))


# --- closed forms: spectra, patterns, ensembles -----------------------------


def sinusoid(
    n: int, spacing: float, cycles: int, amplitude: float
) -> tuple[np.ndarray, float, float]:
    """A sinusoid of ``cycles`` whole periods: variance a²/2 at one wavenumber.

    Args:
        n: Number of samples.
        spacing: Distance between them.
        cycles: Whole periods in n samples, between 1 and n/2 − 1.
        amplitude: a.

    Returns:
        The samples, the wavenumber (cycles per unit distance) and a²/2.
    """
    wavenumber = cycles / (n * spacing)
    position = spacing * np.arange(n)
    values = amplitude * np.cos(2.0 * np.pi * wavenumber * position + 0.7)
    return values, wavenumber, amplitude**2 / 2.0


@dataclasses.dataclass(frozen=True, slots=True)
class Patterns:
    """A field built from orthogonal patterns, and what its EOFs are, up to sign."""

    field: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    variances: np.ndarray


def two_patterns(n_samples: int = 64, n_cells: int = 40) -> Patterns:
    """Two orthonormal patterns with uncorrelated, zero-mean amplitudes.

    Args:
        n_samples: Samples in time, first axis of the field.
        n_cells: Cells in space, second axis.

    Returns:
        The field (samples × cells), the patterns (2 × cells, orthonormal), their
        amplitudes (samples × 2) and the amplitudes' sample variances, descending.
    """
    cell = np.arange(n_cells) / n_cells
    modes = np.stack(
        [
            np.sqrt(2.0 / n_cells) * np.cos(2.0 * np.pi * cell),
            np.sqrt(2.0 / n_cells) * np.sin(4.0 * np.pi * cell),
        ]
    )
    time = np.arange(n_samples) / n_samples
    amplitudes = np.stack(
        [3.0 * np.sin(2.0 * np.pi * 3.0 * time), np.cos(2.0 * np.pi * 5.0 * time)],
        axis=-1,
    )
    # whole periods: each mean is 0 and the sample variance is a²/2 · T/(T − 1)
    variances = np.asarray([9.0, 1.0]) / 2.0 * n_samples / (n_samples - 1)
    return Patterns(amplitudes @ modes, modes, amplitudes, variances)


def reliable_ensemble(
    n_members: int, n_samples: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """A reliable Gaussian ensemble: the truth is one more draw of its members.

    The fair CRPS has expectation 1/√π, and the spread–skill ratio, with its
    (E + 1)/E factor, is 1.

    Args:
        n_members: E, last axis of the ensemble.
        n_samples: Independent cases.
        seed: Seed of the draws.

    Returns:
        The ensemble (samples × members), the truth (samples) and 1/√π.
    """
    rng = np.random.default_rng(seed)
    ensemble = rng.standard_normal((n_samples, n_members))
    return ensemble, rng.standard_normal(n_samples), 1.0 / math.sqrt(math.pi)


# --- reference libraries ----------------------------------------------------


def metpy_advection(scalar: Any, u: Any, v: Any, dx: float, dy: float) -> np.ndarray:
    """MetPy's −u·∇c on a uniform plane, fields ending in (y, x)."""
    calc = pytest.importorskip("metpy.calc")
    units = pytest.importorskip("metpy.units").units
    out = calc.advection(
        np.asarray(scalar) * units.kelvin,
        u=np.asarray(u) * units("m/s"),
        v=np.asarray(v) * units("m/s"),
        dx=dx * units.m,
        dy=dy * units.m,
    )
    return np.asarray(out.to("K/s").magnitude)


def metpy_frontogenesis(theta: Any, u: Any, v: Any, dx: float, dy: float) -> np.ndarray:
    """MetPy's Petterssen frontogenesis on a uniform plane, (y, x) last."""
    calc = pytest.importorskip("metpy.calc")
    units = pytest.importorskip("metpy.units").units
    out = calc.frontogenesis(
        np.asarray(theta) * units.kelvin,
        np.asarray(u) * units("m/s"),
        np.asarray(v) * units("m/s"),
        dx=dx * units.m,
        dy=dy * units.m,
    )
    return np.asarray(out.to("K/m/s").magnitude)


def xrft_power_spectrum(
    values: Any, spacing: float, *, window: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """xrft's two-sided power spectral density along the last axis, demeaned.

    Args:
        values: Samples on the last axis.
        spacing: Distance between them.
        window: A SciPy window name, or ``None`` for none.

    Returns:
        The wavenumbers (cycles per unit distance, ascending) and the density.
    """
    xrft = pytest.importorskip("xrft")
    xr = pytest.importorskip("xarray")
    values = np.asarray(values)
    dims = (*(f"d{axis}" for axis in range(values.ndim - 1)), "x")
    array = xr.DataArray(
        values, dims=dims, coords={"x": spacing * np.arange(values.shape[-1])}
    )
    with warnings.catch_warnings():
        # xrft 1.0 calls the deprecated DataArray.drop
        warnings.simplefilter("ignore", FutureWarning)
        out = xrft.power_spectrum(
            array,
            dim=["x"],
            detrend="constant",
            window=window,
            window_correction=window is not None,
            scaling="density",
        )
    out = out.transpose(*dims[:-1], "freq_x")
    return out["freq_x"].values, out.values


def xskillscore_crps(observation: Any, ensemble: Any) -> np.ndarray:
    """xskillscore's CRPS of each case, the biased one (no fair correction).

    Args:
        observation: One value per case.
        ensemble: Cases × members.
    """
    with warnings.catch_warnings():
        # properscoring has an invalid escape sequence in a docstring
        warnings.simplefilter("ignore", SyntaxWarning)
        xs = pytest.importorskip("xskillscore")
    xr = pytest.importorskip("xarray")
    out = xs.crps_ensemble(
        xr.DataArray(np.asarray(observation), dims="case"),
        xr.DataArray(np.asarray(ensemble), dims=("case", "member")),
        dim=[],
    )
    return out.values


def sklearn_pca(field: Any, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """scikit-learn's unweighted EOFs.

    Args:
        field: Samples × cells.
        n_modes: Modes to keep.

    Returns:
        The patterns (modes × cells), the amplitudes (samples × modes) and the
        variance of each mode.
    """
    decomposition = pytest.importorskip("sklearn.decomposition")
    model = decomposition.PCA(n_components=n_modes, svd_solver="full")
    amplitudes = model.fit_transform(np.asarray(field))
    return model.components_, amplitudes, model.explained_variance_


def functax_eofs(field: Any, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """functax's unweighted EOFs: the patterns (modes × cells) and the variances."""
    eigen = pytest.importorskip("functax.basis.eigen")
    field = np.asarray(field)
    patterns, singular_values = eigen.eof_basis(field, n_modes=n_modes)
    variances = np.asarray(singular_values) ** 2 / (field.shape[0] - 1)
    return np.asarray(patterns).T, variances
