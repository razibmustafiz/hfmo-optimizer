"""Tests for hfmo.optimizer.

These are sanity/regression tests for the package, not a benchmark suite.
They check: convergence on a trivial problem, reproducibility given a seed,
bound feasibility, both bounds-calling-conventions, and that the
evaluation counter and non-vectorized objective path both work correctly.
"""

import numpy as np
import pytest

from hfmo import HFMO, minimize


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def sphere_scalar(x: np.ndarray) -> float:
    return float(np.sum(x**2))


class TestConvergence:
    def test_converges_on_sphere(self):
        result = minimize(sphere, bounds=[(-10, 10)] * 5, n_particles=40, n_iterations=200, seed=0)
        assert result.best_f < 1.0
        assert result.best_x.shape == (5,)

    def test_history_is_monotonically_non_increasing(self):
        result = minimize(sphere, bounds=[(-10, 10)] * 5, n_particles=40, n_iterations=100, seed=1)
        # best-so-far can only improve or stay flat, never get worse
        assert np.all(np.diff(result.history) <= 1e-12)

    def test_final_history_entry_matches_best_f(self):
        result = minimize(sphere, bounds=[(-10, 10)] * 5, n_particles=30, n_iterations=50, seed=2)
        assert result.history[-1] == pytest.approx(result.best_f)


class TestReproducibility:
    def test_same_seed_gives_identical_result(self):
        kwargs = dict(bounds=[(-10, 10)] * 8, n_particles=25, n_iterations=60, seed=42)
        r1 = minimize(sphere, **kwargs)
        r2 = minimize(sphere, **kwargs)
        assert r1.best_f == r2.best_f
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        np.testing.assert_array_equal(r1.history, r2.history)

    def test_different_seeds_can_differ(self):
        kwargs = dict(bounds=[(-10, 10)] * 8, n_particles=25, n_iterations=60)
        r1 = minimize(sphere, seed=1, **kwargs)
        r2 = minimize(sphere, seed=2, **kwargs)
        # Not a strict requirement of correctness, but if this ever starts
        # failing it likely means the seed isn't actually being threaded
        # through every source of randomness.
        assert r1.best_f != r2.best_f or not np.array_equal(r1.best_x, r2.best_x)

    def test_generator_instance_accepted_as_seed(self):
        rng = np.random.default_rng(7)
        result = minimize(sphere, bounds=[(-10, 10)] * 5, n_particles=20, n_iterations=30, seed=rng)
        assert np.isfinite(result.best_f)


class TestBounds:
    def test_pairs_list_convention(self):
        opt = HFMO(sphere, bounds=[(-5, 5)] * 6, n_particles=20, n_iterations=10, seed=0)
        assert opt.D == 6
        result = opt.optimize()
        assert np.all(result.best_x >= -5) and np.all(result.best_x <= 5)

    def test_low_high_arrays_convention(self):
        low = np.array([-1.0, -2.0, -3.0])
        high = np.array([1.0, 2.0, 3.0])
        opt = HFMO(sphere, bounds=(low, high), n_particles=20, n_iterations=10, seed=0)
        assert opt.D == 3
        result = opt.optimize()
        assert np.all(result.best_x >= low) and np.all(result.best_x <= high)

    def test_scalar_low_high_requires_dim(self):
        with pytest.raises(ValueError):
            HFMO(sphere, bounds=(-1.0, 1.0), n_particles=10, n_iterations=5)  # no dim given

        opt = HFMO(sphere, bounds=(-1.0, 1.0), dim=4, n_particles=10, n_iterations=5, seed=0)
        assert opt.D == 4

    def test_all_candidates_stay_within_bounds_throughout(self):
        opt = HFMO(sphere, bounds=[(-2, 2)] * 4, n_particles=30, n_iterations=1, seed=0)
        for t in range(1, 40):
            opt.step(t)
            assert np.all(opt.X >= opt.X_min) and np.all(opt.X <= opt.X_max)

    def test_malformed_bounds_raise(self):
        with pytest.raises(ValueError):
            HFMO(sphere, bounds=[(-1, 1, 2)], n_particles=10, n_iterations=5)  # wrong tuple width


class TestEvaluationCounting:
    def test_n_evaluations_is_positive_and_tracked(self):
        result = minimize(sphere, bounds=[(-5, 5)] * 5, n_particles=20, n_iterations=15, seed=0)
        assert result.n_evaluations > 0
        # At minimum, initialization (N) plus one evaluation per iteration
        # for whichever phase ran must have been counted.
        assert result.n_evaluations >= 20 + 15

    def test_non_vectorized_objective(self):
        result = minimize(
            sphere_scalar,
            bounds=[(-5, 5)] * 4,
            n_particles=15,
            n_iterations=20,
            vectorized=False,
            seed=0,
        )
        assert np.isfinite(result.best_f)
        assert result.n_evaluations > 0


class TestValidation:
    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            HFMO(sphere, bounds=[(-5, 5)] * 3, beta=3.0, n_particles=10, n_iterations=5)

    def test_too_few_particles_raises(self):
        with pytest.raises(ValueError):
            HFMO(sphere, bounds=[(-5, 5)] * 3, n_particles=1, n_iterations=5)
