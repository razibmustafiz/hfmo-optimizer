"""
Basic usage of HFMO.

Two equivalent ways to run it: the one-shot ``minimize`` function, and the
``HFMO`` class directly when you want access to the optimizer object
afterwards (e.g. to inspect ``n_evaluations`` or run more iterations).
"""

import numpy as np

from hfmo import HFMO, minimize


def rastrigin(x: np.ndarray) -> np.ndarray:
    """A standard multimodal benchmark function, global minimum f(0,...,0)=0."""
    A = 10
    n = x.shape[1]
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def main() -> None:
    # --- One-shot, scipy.optimize-style ---------------------------------
    result = minimize(
        rastrigin,
        bounds=[(-5.12, 5.12)] * 10,   # 10-D Rastrigin, one (low, high) pair per dimension
        n_particles=60,
        n_iterations=500,
        seed=0,                        # always set a seed for reproducible results
    )
    print("minimize():")
    print(f"  best_f          = {result.best_f:.6f}")
    print(f"  n_evaluations   = {result.n_evaluations}")
    print(f"  n_iterations    = {result.n_iterations}")
    print(f"  best_x[:3]      = {result.best_x[:3]}")

    # --- Class-based, for more control -----------------------------------
    optimizer = HFMO(
        func=rastrigin,
        bounds=[(-5.12, 5.12)] * 10,
        n_particles=60,
        n_iterations=500,
        seed=0,
    )
    result2 = optimizer.optimize()
    print("\nHFMO class (same seed -> identical result):")
    print(f"  best_f == result.best_f : {result2.best_f == result.best_f}")

    # Run more iterations on the same, already-optimized state if you want.
    extra = optimizer.optimize(n_iterations=100)
    print(f"\nAfter 100 more iterations: best_f = {extra.best_f:.6f}, "
          f"total n_evaluations = {optimizer.n_evaluations}")


if __name__ == "__main__":
    main()
