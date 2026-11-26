"""Separation, cumulative separation and skill of trajectories, and dispersion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _backends import (
    EAGER_KINDS,
    assert_backends_agree,
    assert_refuses_dataarrays,
    needs_jax,
)
from geophysix.earth import SPHERE, WGS84, great_circle_distance, sphere
from pipekit_array.testing import (
    assert_finite_gradient,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)

from diagnostix import ValidationError
from diagnostix.trajectories import (
    absolute_dispersion,
    cumulative_separation,
    pair_dispersion,
    separation,
    trajectory_skill,
)


DISTANCES = pytest.mark.parametrize("distance", ["euclidean", "sphere"])
SCORES = (separation, cumulative_separation, trajectory_skill)


def _observed(distance: str) -> np.ndarray:
    """A day of hourly positions, across the date line on the sphere."""
    t = np.linspace(0.0, 1.0, 25)
    if distance == "euclidean":
        return 1e3 * np.column_stack([t, np.sin(3.0 * t)])
    longitude = np.mod(179.5 + t + 180.0, 360.0) - 180.0
    return np.column_stack([longitude, 40.0 + 0.5 * t])


def _simulated(distance: str) -> np.ndarray:
    """It drifts away from the same start."""
    t = np.linspace(0.0, 1.0, 25)
    drift = {"euclidean": 100.0, "sphere": 0.01}[distance]
    path = _observed(distance).copy()
    path[:, 1] += drift * t
    return path


def _by_hand(left: Any, right: Any, distance: str) -> float:
    if distance == "sphere":
        return float(great_circle_distance(*left, *right))
    return float(np.linalg.norm(left - right))


def _score_by_hand(simulated: Any, observed: Any, distance: str) -> float:
    """∑ d_k / ∑ l_k, one time at a time."""
    total = scale = length = 0.0
    last = None
    for x, y in zip(simulated, observed, strict=True):
        if not np.isfinite(y).all():
            continue
        if last is not None:
            length += _by_hand(last, y, distance)
        last = y
        total += _by_hand(x, y, distance)
        scale += length
    return total / scale


def _walk(n_times: int = 5, n_particles: int = 6, dim: int = 2) -> np.ndarray:
    rng = np.random.default_rng(3)
    steps = rng.normal(size=(n_times, n_particles, dim))
    return np.cumsum(steps, axis=0)


def _globe(n_times: int = 5, n_particles: int = 6) -> np.ndarray:
    rng = np.random.default_rng(4)
    start = np.column_stack(
        [rng.uniform(-180.0, 180.0, n_particles), rng.uniform(-60.0, 60.0, n_particles)]
    )
    steps = rng.normal(scale=2.0, size=(n_times, n_particles, 2))
    path = start + np.cumsum(steps, axis=0)
    path[..., 0] = np.mod(path[..., 0] + 180.0, 360.0) - 180.0
    return path


# --- separation and skill: values -----------------------------------------------


@DISTANCES
def test_the_values_match_a_loop(distance: str) -> None:
    simulated, observed = _simulated(distance), _observed(distance)
    expected = [
        _by_hand(x, y, distance) for x, y in zip(simulated, observed, strict=True)
    ]
    found = separation(simulated, observed, distance=distance)
    assert_matches_oracle(found, np.array(expected), rtol=1e-12)
    s = _score_by_hand(simulated, observed, distance)
    found = cumulative_separation(simulated, observed, distance=distance)
    assert_matches_oracle(found, s, rtol=1e-12)
    found = trajectory_skill(simulated, observed, distance=distance)
    assert_matches_oracle(found, 1.0 - s, rtol=1e-12)
    found = trajectory_skill(simulated, observed, distance=distance, tolerance=2.0 * s)
    assert_matches_oracle(found, 0.5, rtol=1e-12)


@DISTANCES
@pytest.mark.parametrize(
    "missing", [[0], [0, 1, 2], [7], [3, 4, 5, 11], [24], [0, 12, 23, 24]]
)
def test_a_missing_observation_is_left_out_and_bridged(
    distance: str, missing: list[int]
) -> None:
    simulated, gaps = _simulated(distance), _observed(distance)
    gaps[missing] = np.nan
    # one column is enough to be missing
    gaps[15, 0] = np.nan
    expected = _score_by_hand(simulated, gaps, distance)
    found = cumulative_separation(simulated, gaps, distance=distance)
    assert_matches_oracle(found, expected, rtol=1e-12)
    apart = separation(simulated, gaps, distance=distance)
    assert np.flatnonzero(np.isnan(apart)).tolist() == sorted([*missing, 15])
    # what is simulated there does not matter, whatever it is
    simulated[missing] = np.nan
    again = cumulative_separation(simulated, gaps, distance=distance)
    assert_matches_oracle(again, expected, rtol=1e-12)


@DISTANCES
def test_a_simulated_nan_is_not_hidden(distance: str) -> None:
    simulated, observed = _simulated(distance), _observed(distance)
    simulated[7] = np.nan
    assert np.isnan(separation(simulated, observed, distance=distance)[7])
    assert np.isnan(cumulative_separation(simulated, observed, distance=distance))
    assert np.isnan(trajectory_skill(simulated, observed, distance=distance))


def test_the_skill_is_zero_beyond_the_tolerance() -> None:
    observed = _observed("euclidean")
    far = observed + np.array([0.0, 1e6])
    assert float(cumulative_separation(far, observed)) > 1.0
    assert float(trajectory_skill(far, observed)) == 0.0
    assert float(trajectory_skill(observed, observed)) == 1.0


def test_an_observation_that_does_not_move_needs_a_length() -> None:
    observed = np.ones((5, 2))
    simulated = observed.copy()
    simulated[1:, 0] += 2.0
    assert np.isinf(cumulative_separation(simulated, observed))
    assert float(trajectory_skill(simulated, observed)) == 0.0
    # 8 m over 5 lengths of 4 m
    found = cumulative_separation(simulated, observed, min_length=4.0)
    assert_matches_oracle(found, 0.4, rtol=1e-14)
    found = trajectory_skill(simulated, observed, min_length=4.0)
    assert_matches_oracle(found, 0.6, rtol=1e-14)


def test_no_observation_is_nan() -> None:
    simulated = _simulated("euclidean")
    nothing = np.full_like(simulated, np.nan)
    assert np.isnan(cumulative_separation(simulated, nothing))
    assert np.isnan(trajectory_skill(simulated, nothing))


def test_three_columns() -> None:
    observed = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 6.0]])
    simulated = observed + np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    assert_matches_oracle(cumulative_separation(simulated, observed), 3 / 7, rtol=1e-14)


def test_integers_do_not_truncate_the_observations() -> None:
    simulated = np.array([[0, 0], [10, 0], [20, 0]])
    observed = np.array([[0.0, 0.0], [10.5, 0.0], [np.nan, 0.0]])
    found = separation(simulated, observed)
    assert_matches_oracle(found, np.array([0.0, 0.5, np.nan]), rtol=1e-14)


def test_another_sphere_changes_the_separation_and_not_the_score() -> None:
    simulated, observed = _simulated("sphere"), _observed("sphere")
    large = separation(simulated, observed, distance="sphere")
    small = separation(simulated, observed, distance="sphere", earth=sphere(1.0))
    assert_matches_oracle(small * SPHERE.semi_major, large, rtol=1e-12)
    large = cumulative_separation(simulated, observed, distance="sphere")
    small = cumulative_separation(
        simulated, observed, distance="sphere", earth=sphere(1.0)
    )
    assert_matches_oracle(small, large, rtol=1e-12)


@DISTANCES
def test_one_reference_serves_a_batch(distance: str) -> None:
    observed = _observed(distance)
    observed[4] = np.nan
    drift = _simulated(distance) - _observed(distance)
    batch = np.stack([_observed(distance) + scale * drift for scale in (0.0, 1.0, 3.0)])
    found = cumulative_separation(batch, observed[None], distance=distance)
    expected = [
        cumulative_separation(member, observed, distance=distance) for member in batch
    ]
    assert_matches_oracle(found, np.stack(expected), rtol=1e-13, atol=1e-15)


# --- separation and skill: backends and gradients -------------------------------


@pytest.mark.slow
@DISTANCES
@pytest.mark.parametrize("score", SCORES, ids=lambda fn: fn.__name__)
def test_backends_agree(distance: str, score: Any) -> None:
    observed = np.stack([_observed(distance), _observed(distance)])
    observed[1, 4] = np.nan
    # apart at the start too: a distance of 0 is rounding, which differs under jit
    apart = _simulated(distance) + np.array([0.0, 0.02])
    simulated = np.stack([apart, 2.0 * apart - _observed(distance)])

    def of(x: Any, y: Any) -> Any:
        return score(x, y, distance=distance)

    # float32 resolves 1e-7 rad, 0.6 m on the Earth, of separations of kilometres
    rtol32 = 2e-3 if distance == "sphere" else 1e-4
    assert_backends_agree(of, simulated, observed, kinds=EAGER_KINDS, rtol32=rtol32)
    assert_traces(of, simulated[None], observed[None])
    assert_refuses_dataarrays(of, simulated, observed)


@needs_jax
@DISTANCES
def test_gradients_match_finite_differences(distance: str) -> None:
    """With a shared starting point and a missing observation."""
    observed = _observed(distance)
    observed[4] = np.nan
    first = observed[:1]
    eps = {"euclidean": 1e-4, "sphere": 1e-7}[distance]

    def of_simulated(later: Any) -> Any:
        # the start is shared, where the distance has a kink
        xp = later.__array_namespace__()
        simulated = xp.concat([xp.asarray(first), later])
        return cumulative_separation(simulated, observed, distance=distance)

    assert_grad_matches_fd(of_simulated, _simulated(distance)[1:], eps=eps, rtol=1e-6)


@needs_jax
@DISTANCES
def test_the_gradient_is_finite_at_a_perfect_forecast(distance: str) -> None:
    observed = _observed(distance)
    observed[4] = np.nan
    perfect = _observed(distance)
    for score in (cumulative_separation, trajectory_skill):
        assert_finite_gradient(
            lambda x, s=score: s(x, observed, distance=distance), perfect
        )


@needs_jax
def test_a_drifter_that_does_not_move_spoils_no_gradient() -> None:
    """One good drifter and one stationary one, with a shared parameter."""
    import jax
    import jax.numpy as jnp

    moving, still = _observed("euclidean"), np.ones((25, 2))
    observed = jnp.asarray(np.stack([moving, still]))
    direction = jnp.array([1.0, 2.0])
    for score in (cumulative_separation, trajectory_skill):

        def total(drift: Any, score: Any = score) -> Any:
            scores = score(observed + drift * direction, observed)
            return jnp.where(jnp.isfinite(scores), scores, 0.0).sum()

        def alone(drift: Any, score: Any = score) -> Any:
            return score(jnp.asarray(moving) + drift * direction, jnp.asarray(moving))

        gradient = jax.grad(total)(3.0)
        assert_matches_oracle(gradient, float(jax.grad(alone)(3.0)), rtol=1e-12)
        assert float(gradient) != 0.0


# --- dispersion -------------------------------------------------------------------


def test_a_strain_separates_a_pair_exponentially() -> None:
    """x(t) = x₀ e^{rt}, y(t) = y₀ e^{−rt}."""
    rate, times = 0.1, np.linspace(0.0, 10.0, 11)
    start = np.random.default_rng(5).random((8, 2))
    growth = np.exp(rate * times)
    trajectory = np.stack([start * np.array([g, 1.0 / g]) for g in growth])
    pairs = np.array([[0, 1], [2, 3], [4, 7]])
    gap = start[pairs[:, 0]] - start[pairs[:, 1]]
    expected = [np.mean((gap[:, 0] * g) ** 2 + (gap[:, 1] / g) ** 2) for g in growth]
    found = pair_dispersion(trajectory, pairs)
    assert_matches_oracle(found, np.array(expected), rtol=1e-12)


def test_a_pair_across_the_date_line_has_the_short_distance() -> None:
    trajectory = np.array([[[179.5, 0.0], [-179.5, 0.0]]])
    found = pair_dispersion(trajectory, np.array([[0, 1]]), distance="sphere")
    expected = np.deg2rad(1.0) * SPHERE.semi_major
    assert_matches_oracle(np.sqrt(found), np.array([expected]), rtol=1e-9)
    small = pair_dispersion(
        trajectory, np.array([[0, 1]]), distance="sphere", earth=sphere(1.0)
    )
    assert_matches_oracle(np.sqrt(small), np.array([np.deg2rad(1.0)]), rtol=1e-9)


@DISTANCES
def test_both_dispersions_match_a_loop(distance: str) -> None:
    values = _globe() if distance == "sphere" else _walk(dim=3)
    # two missing positions, and a particle whose start is missing
    values[2, 1] = np.nan
    values[3, 4, 0] = np.nan
    values[0, 5] = np.nan
    pairs = np.array([[0, 1], [4, 2], [3, 3]])

    def squared(left: np.ndarray, right: np.ndarray) -> float:
        if distance == "sphere":
            return float(great_circle_distance(*left, *right)) ** 2
        return float(np.sum((left - right) ** 2))

    absolute, relative = [], []
    for k in range(5):
        kept = [p for p in range(6) if np.isfinite(values[[0, k], p]).all()]
        absolute.append(np.mean([squared(values[k, p], values[0, p]) for p in kept]))
        kept = [(p, q) for p, q in pairs if np.isfinite(values[k, [p, q]]).all()]
        relative.append(np.mean([squared(values[k, p], values[k, q]) for p, q in kept]))
    found = absolute_dispersion(values, distance=distance)
    assert_matches_oracle(found, np.array(absolute), rtol=1e-12)
    found = pair_dispersion(values, pairs, distance=distance)
    assert_matches_oracle(found, np.array(relative), rtol=1e-12)


def test_a_time_with_nothing_known_is_nan() -> None:
    trajectory = _walk()
    trajectory[2] = np.nan
    found = absolute_dispersion(trajectory)
    assert np.isnan(found).tolist() == [False, False, True, False, False]
    found = pair_dispersion(trajectory, np.array([[0, 1]]))
    assert np.isnan(found).tolist() == [False, False, True, False, False]


def test_a_missing_particle_changes_nothing() -> None:
    trajectory = _walk()
    missing = trajectory.copy()
    missing[:, 5] = np.nan
    found = absolute_dispersion(missing)
    assert_matches_oracle(found, absolute_dispersion(trajectory[:, :5]), rtol=1e-14)


@pytest.mark.slow
@DISTANCES
def test_the_dispersions_on_every_backend(distance: str) -> None:
    trajectory = _globe(4, 6) if distance == "sphere" else _walk(4, 6)
    pairs = np.array([[0, 1], [2, 5]])

    # at the start the distance is rounding, which differs under jit
    def absolute(x: Any) -> Any:
        return absolute_dispersion(x, distance=distance)[1:]

    def relative(x: Any, p: Any) -> Any:
        return pair_dispersion(x, p, distance=distance)

    rtol32 = 2e-3 if distance == "sphere" else 1e-4
    assert_backends_agree(absolute, trajectory, rtol32=rtol32)
    assert_backends_agree(relative, trajectory, pairs, rtol32=rtol32)
    assert_refuses_dataarrays(absolute, trajectory)


@needs_jax
@DISTANCES
def test_dispersion_gradients_match_finite_differences(distance: str) -> None:
    trajectory = _globe(3, 4) if distance == "sphere" else _walk(3, 4)
    # one missing position, and the start, where the distance is zero
    trajectory[2, 1] = np.nan
    pairs = np.array([[0, 1], [2, 3]])
    scale = 1e-10 if distance == "sphere" else 1.0

    def both(positions: Any) -> Any:
        absolute = absolute_dispersion(positions, distance=distance)
        relative = pair_dispersion(positions, pairs, distance=distance)
        return scale * (absolute + relative)

    assert_grad_matches_fd(both, trajectory, eps=1e-5, rtol=1e-6)


# --- what is refused --------------------------------------------------------------


@pytest.mark.parametrize("score", SCORES, ids=lambda fn: fn.__name__)
def test_scores_refuse(score: Any) -> None:
    simulated, observed = _simulated("euclidean"), _observed("euclidean")
    with pytest.raises(ValidationError, match="distance"):
        score(simulated, observed, distance="manhattan")
    with pytest.raises(ValidationError, match="latitudes"):
        score(simulated, observed, distance="sphere")
    with pytest.raises(ValidationError, match="same times and columns"):
        score(simulated, observed[:5])
    with pytest.raises(ValidationError, match="same times and columns"):
        score(simulated, np.zeros((25, 3)))
    globe = _observed("sphere")
    with pytest.raises(ValidationError, match="must be a sphere"):
        score(globe, globe, distance="sphere", earth=WGS84)


def test_the_bounds_are_checked() -> None:
    simulated, observed = _simulated("euclidean"), _observed("euclidean")
    for value in (-1.0, np.inf, np.nan):
        with pytest.raises(ValidationError, match="min_length"):
            cumulative_separation(simulated, observed, min_length=value)
    for value in (0.0, -1.0, np.inf):
        with pytest.raises(ValidationError, match="tolerance"):
            trajectory_skill(simulated, observed, tolerance=value)
    with pytest.raises(ValidationError, match="scalar"):
        cumulative_separation(simulated, observed, min_length=np.ones(25))


def test_dispersions_refuse() -> None:
    trajectory = _walk()
    with pytest.raises(ValidationError, match="distance"):
        absolute_dispersion(trajectory, distance="great-circle")
    with pytest.raises(ValidationError, match="longitude and latitude"):
        absolute_dispersion(_walk(dim=3), distance="sphere")
    with pytest.raises(ValidationError, match="latitudes"):
        absolute_dispersion(100.0 + trajectory, distance="sphere")
    with pytest.raises(ValidationError, match="sphere"):
        absolute_dispersion(_globe(), distance="sphere", earth=WGS84)
    with pytest.raises(ValidationError, match="time, particle, column"):
        absolute_dispersion(trajectory[0])
    with pytest.raises(ValidationError, match=r"\(pair, 2\)"):
        pair_dispersion(trajectory, np.array([0, 1]))
    for pairs in ([[0, 6]], [[-1, 0]]):
        with pytest.raises(ValidationError, match="'pairs' must name"):
            pair_dispersion(trajectory, np.array(pairs))
    with pytest.raises(ValidationError, match="integers"):
        pair_dispersion(trajectory, np.array([[0.0, 1.0]]))
