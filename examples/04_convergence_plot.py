"""
Plotting the convergence history, and using it for a simple early-stopping
callback -- a common requirement inside a larger pipeline where you don't
want to burn the full iteration budget once the search has clearly
plateaued.

Requires matplotlib (pip install matplotlib, or `pip install hfmo-optimizer[examples]`).
"""

import numpy as np

from hfmo import HFMO


def ackley(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    return (
        -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20
        + np.e
    )


def main() -> None:
    optimizer = HFMO(ackley, bounds=[(-32.768, 32.768)] * 10, n_particles=50, n_iterations=1, seed=0)

    # Manual loop with early stopping: run in chunks and stop once the
    # best-so-far fitness hasn't improved meaningfully in a while.
    chunk = 20
    patience_chunks = 5
    history = []
    stalled_chunks = 0

    for _ in range(50):  # up to 50 * chunk = 1000 iterations
        result = optimizer.optimize(n_iterations=chunk)
        history.append(result.history)
        improved = len(history) < 2 or (history[-2][-1] - history[-1][-1]) > 1e-6
        stalled_chunks = 0 if improved else stalled_chunks + 1
        if stalled_chunks >= patience_chunks:
            print(f"Stopped early after {len(history) * chunk} iterations (no improvement).")
            break

    full_history = np.concatenate(history)
    print(f"Final best_f = {optimizer.G_fitness:.6f} "
          f"after {len(full_history)} iterations, {optimizer.n_evaluations} evaluations")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see the convergence plot)")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(full_history)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far fitness (log scale)")
    plt.title("HFMO convergence on 10-D Ackley")
    plt.tight_layout()
    plt.savefig("hfmo_convergence.png", dpi=150)
    print("Saved plot to hfmo_convergence.png")


if __name__ == "__main__":
    main()
