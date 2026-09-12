"""
Comparing HFMO against scipy.optimize.differential_evolution under a
matched function-evaluation budget.

This is the honest way to compare two population-based optimizers: by
evaluation count, not by iteration count or wall-clock time alone, since
different algorithms spend different numbers of evaluations per iteration.
HFMO's HFMOResult.n_evaluations makes this straightforward to check.
"""

import time

import numpy as np
from scipy.optimize import differential_evolution

from hfmo import minimize


def rastrigin(x: np.ndarray) -> np.ndarray:
    A = 10
    n = x.shape[-1]
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)


def rastrigin_scalar(x: np.ndarray) -> float:
    return float(rastrigin(x[None, :])[0])


def main() -> None:
    dim = 20
    bounds = [(-5.12, 5.12)] * dim
    n_particles = 50
    n_iterations = 300
    target_evals = n_particles * n_iterations  # a rough shared budget

    print(f"Problem: {dim}-D Rastrigin, target budget ~{target_evals} evaluations\n")

    # --- HFMO -------------------------------------------------------------
    t0 = time.perf_counter()
    hfmo_result = minimize(
        rastrigin, bounds=bounds, n_particles=n_particles, n_iterations=n_iterations, seed=0
    )
    hfmo_time = time.perf_counter() - t0

    # --- scipy.optimize.differential_evolution, matched maxiter -----------
    # DE's budget is (maxiter + 1) * popsize * dim evaluations (roughly);
    # we pick maxiter so the two runs land in the same ballpark.
    de_maxiter = max(1, target_evals // (15 * dim))
    t0 = time.perf_counter()
    de_result = differential_evolution(
        rastrigin_scalar, bounds=bounds, maxiter=de_maxiter, popsize=15, seed=0, polish=False
    )
    de_time = time.perf_counter() - t0

    print(f"{'Optimizer':<12}{'best_f':>14}{'n_evaluations':>16}{'time (s)':>12}")
    print(f"{'HFMO':<12}{hfmo_result.best_f:>14.4f}{hfmo_result.n_evaluations:>16}{hfmo_time:>12.3f}")
    print(f"{'DE (scipy)':<12}{de_result.fun:>14.4f}{de_result.nfev:>16}{de_time:>12.3f}")
    print(
        "\nTreat this as a single anecdotal run, not a benchmark claim -- "
        "re-run with several seeds and report mean +/- std before drawing "
        "any conclusion from it."
    )


if __name__ == "__main__":
    main()
