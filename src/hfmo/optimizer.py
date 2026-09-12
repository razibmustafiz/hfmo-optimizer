"""
hfmo.optimizer
==============

Reference implementation of Hilsha Fish Migration Optimization (HFMO).

This is a corrected, vectorized re-implementation of the research prototype
used in the accompanying manuscript. See ``CHANGELOG.md`` at the repository
root for exactly what was fixed relative to that prototype, and the README
for the algorithm description, the biological motivation, and citations for
every component mechanism it composes.

At a glance, HFMO maintains a population ("shoal") of candidate solutions
and cycles them through four phases, selected each iteration by a scalar
salinity signal S(t):

    S(t) = |sin(pi * t / T_max)| * exp(-gamma * t / T_max)

    S(t) > 0.7        -> Marine phase       (Levy-flight global search)
    0.3 < S(t) <= 0.7  -> Migration phase    (PSO-style directed search)
    S(t) <= 0.3        -> Spawning phase     (Gaussian local refinement)
    S(t) > S(t-1)      -> Return phase       (elite-guided reinitialization,
                                               layered independently of the
                                               three phases above)

``S(t)`` is a deterministic function of the iteration index alone; it is a
fixed annealing-style schedule, not a state-dependent controller. This is
stated plainly because the schedule is the actual point of the algorithm:
each phase update is itself a well-known mechanism (see README "Attribution"
section), and HFMO's contribution is their composition under this schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import gamma as _gamma_fn

ArrayLike = Union[np.ndarray, Sequence[float]]
BoundsLike = Union[
    Tuple[ArrayLike, ArrayLike],           # (low, high) each shape (D,) or scalar
    Sequence[Tuple[float, float]],          # [(low_1, high_1), ..., (low_D, high_D)]
]


@dataclass
class HFMOResult:
    """Result of an :meth:`HFMO.optimize` run.

    Attributes
    ----------
    best_x : np.ndarray, shape (D,)
        Best solution found.
    best_f : float
        Fitness of ``best_x``.
    history : np.ndarray, shape (n_iterations,)
        Best-so-far fitness at the end of each iteration. Use this for
        convergence plots or early-stopping logic in a pipeline.
    n_evaluations : int
        Total number of objective-function evaluations consumed. Track this
        when comparing against other optimizers under a fixed budget --
        iteration count alone is not a fair basis for comparison, since
        different phases (and different algorithms) evaluate different
        numbers of candidates per iteration.
    n_iterations : int
        Number of iterations run.
    """

    best_x: np.ndarray
    best_f: float
    history: np.ndarray
    n_evaluations: int
    n_iterations: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HFMOResult(best_f={self.best_f!r}, "
            f"n_evaluations={self.n_evaluations}, "
            f"n_iterations={self.n_iterations})"
        )


def _parse_bounds(bounds: BoundsLike, dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse the two supported calling conventions for ``bounds``:

    - a **tuple** ``(low, high)`` of scalars or shape-``(D,)`` arrays --
      pass ``dim`` explicitly if ``low``/``high`` are scalars; or
    - a **list** (or array) of ``(low_i, high_i)`` pairs, one per
      dimension, exactly as :func:`scipy.optimize.differential_evolution`
      expects -- e.g. ``[(-100, 100)] * D``. ``dim`` is inferred as the
      number of pairs and does not need to be given.

    Returns ``(low, high, dim)`` as shape-``(D,)`` float arrays.
    """
    if isinstance(bounds, tuple) and len(bounds) == 2:
        low, high = bounds
        low = np.atleast_1d(np.asarray(low, dtype=float))
        high = np.atleast_1d(np.asarray(high, dtype=float))
        if low.size == 1 or high.size == 1:
            if dim is None:
                raise ValueError(
                    "bounds=(low, high) with scalar low/high requires dim= "
                    "to be given explicitly (there is no way to infer the "
                    "dimensionality from a scalar pair)"
                )
            low = np.full(dim, low.item()) if low.size == 1 else low
            high = np.full(dim, high.item()) if high.size == 1 else high
        if low.shape != high.shape:
            raise ValueError("bounds=(low, high): low and high must have the same shape")
        return low, high, low.shape[0]

    # scipy-style: a sequence of (low_i, high_i) pairs, one per dimension.
    pairs = np.asarray(bounds, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "bounds must be either a tuple (low, high) of scalars/arrays "
            "(with dim given if scalar), or a sequence of (low_i, high_i) "
            "pairs, one per dimension, e.g. [(-100, 100)] * D as in "
            "scipy.optimize.differential_evolution"
        )
    return pairs[:, 0].copy(), pairs[:, 1].copy(), pairs.shape[0]


class HFMO:
    """Hilsha Fish Migration Optimization.

    Parameters
    ----------
    func : callable
        Objective function to **minimize**. By default it must be
        vectorized: given an array of shape ``(n, D)`` it returns an array
        of shape ``(n,)``. If your function only accepts a single
        1-D vector, pass ``vectorized=False`` and HFMO will call it once
        per candidate internally.
    bounds : (low, high) or sequence of (low_i, high_i)
        Search-space bounds. Either a pair of arrays/scalars ``(low, high)``
        together with ``dim``, or a scipy-style list of per-dimension pairs
        such as ``[(-100, 100)] * D``.
    dim : int, optional
        Dimensionality. Required only when ``bounds`` is given as a scalar
        ``(low, high)`` pair; inferred automatically from a per-dimension
        bounds list.
    n_particles : int, default 50
        Population size (number of candidate solutions per iteration).
    n_iterations : int, default 500
        Number of iterations to run in :meth:`optimize`.
    w_start, w_end : float, default 0.9, 0.4
        Inertia weight at the start/end of the run, linearly interpolated
        during the migration phase (Shi & Eberhart, 1998).
    c1, c2 : float, default 1.7, 1.7
        Cognitive and social coefficients for the migration phase's
        velocity update.
    s_min : float, default 0.1
        Salinity threshold below which the spawning phase's mutation scale
        is computed; also the lower edge of the spawning regime.
    beta : float, default 1.5
        Levy-flight stability exponent, beta in (1, 2].
    alpha_scale : float, default 0.01
        Scales the Levy step relative to the search-space width.
    eta : float, default 0.1
        Scales the Gaussian perturbation used to reinitialize replaced
        individuals in the return phase.
    gamma : float, default 0.2
        Exponential decay rate of the salinity schedule, in [0, 0.5].
    epsilon : float, default 1e-6
        Small constant preventing division by zero when weighting personal
        bests by fitness in the migration phase.
    vectorized : bool, default True
        Whether ``func`` accepts a batch of candidates at once. Set to
        ``False`` if it only accepts a single 1-D vector and returns a
        scalar; HFMO will then evaluate candidates one at a time.
    seed : int or numpy.random.Generator, optional
        Seed (or generator) for reproducible runs. **Always set this** when
        comparing HFMO against other algorithms or reporting results --
        without it, results are not exactly reproducible.

    Notes
    -----
    Each phase update is a well-known mechanism, composed here under a
    single deterministic salinity schedule -- see the README's
    "Attribution" section for the primary source of each:

    - Marine phase: a Levy-flight step generated via Mantegna's algorithm,
      as used in Cuckoo Search.
    - Migration phase: the standard PSO velocity update with a linearly
      decaying inertia weight, whose social term is a fitness-weighted
      average over *all* personal bests (as in Fully Informed PSO), not
      just the single global best.
    - Spawning phase: a Gaussian local-search mutation scaled by distance
      to the incumbent best, as in bare-bones PSO, with greedy acceptance.
    - Return phase: elite-guided reinitialization of the worst individuals,
      a pattern common to scout-bee and migration-based algorithms.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: BoundsLike,
        dim: Optional[int] = None,
        n_particles: int = 50,
        n_iterations: int = 500,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.7,
        c2: float = 1.7,
        s_min: float = 0.1,
        beta: float = 1.5,
        alpha_scale: float = 0.01,
        eta: float = 0.1,
        gamma: float = 0.2,
        epsilon: float = 1e-6,
        vectorized: bool = True,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        low, high, dim = _parse_bounds(bounds, dim)
        if not (1.0 < beta <= 2.0):
            raise ValueError(f"beta (Levy exponent) must be in (1, 2]; got {beta}")
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")

        self.N = n_particles
        self.D = dim
        self.X_min = low.reshape(1, dim)
        self.X_max = high.reshape(1, dim)
        self.T_max = n_iterations

        self._raw_func = func
        self._vectorized = vectorized

        self.w_start, self.w_end = w_start, w_end
        self.c1, self.c2 = c1, c2
        self.s_min = s_min
        self.beta = beta
        self.alpha_scale = alpha_scale
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        self.alpha = self.alpha_scale * (self.X_max - self.X_min)

        # Mantegna's algorithm scale factor for beta-stable Levy steps.
        # NOTE: this scale is the *standard deviation* passed to the normal
        # sampler below (np.random.Generator.normal's second argument is a
        # standard deviation, not a variance) -- an earlier research
        # prototype squared it by mistake, which drew steps from the wrong
        # distribution. See CHANGELOG.md.
        self.sigma_u = (
            _gamma_fn(1 + self.beta) * np.sin(np.pi * self.beta / 2)
            / (_gamma_fn((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)
        self.sigma_v = 1.0

        self.n_evaluations = 0

        self.X = self.X_min + self.rng.uniform(0, 1, (self.N, self.D)) * (self.X_max - self.X_min)
        self.V = 0.1 * (self.X_min + self.rng.uniform(0, 1, (self.N, self.D)) * (self.X_max - self.X_min))
        self.fitness = self._evaluate(self.X)
        self.P = self.X.copy()
        self.P_fitness = self.fitness.copy()
        g_idx = int(np.argmin(self.P_fitness))
        self.G = self.P[g_idx].copy()
        self.G_fitness = float(self.P_fitness[g_idx])
        self.S_prev = 0.0
        self._t = 0  # persists across optimize() calls, so a second call
        # continues the salinity schedule rather than restarting it at t=1

    # ------------------------------------------------------------------ #
    # Objective-function plumbing
    # ------------------------------------------------------------------ #
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the objective on a batch of candidates and track NFE."""
        if X.shape[0] == 0:
            return np.empty(0)
        if self._vectorized:
            fitness = np.asarray(self._raw_func(X), dtype=float)
        else:
            fitness = np.array([float(self._raw_func(row)) for row in X])
        self.n_evaluations += X.shape[0]
        return fitness

    def _clamp(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.X_min, self.X_max)

    # ------------------------------------------------------------------ #
    # Salinity schedule
    # ------------------------------------------------------------------ #
    def _update_salinity(self, t: int) -> float:
        return float(np.abs(np.sin(np.pi * t / self.T_max)) * np.exp(-self.gamma * t / self.T_max))

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #
    def _generate_levy_batch(self, n: int) -> np.ndarray:
        u = self.rng.normal(0.0, self.sigma_u, size=(n, self.D))
        v = self.rng.normal(0.0, self.sigma_v, size=(n, self.D))
        return u / (np.abs(v) ** (1.0 / self.beta))

    def _marine_phase(self) -> None:
        """Global exploration via Levy flights (vectorized over the whole
        population; costs one objective evaluation of size N)."""
        step = self._generate_levy_batch(self.N)
        self.X = self._clamp(self.X + self.alpha * step)
        self.V = self.alpha * step
        self.fitness = self._evaluate(self.X)

    def _migration_phase(self, t: int) -> None:
        """Directed search: a PSO velocity update whose social term is a
        fitness-weighted average over all personal bests.

        The weighted mean of ``P`` is the same for every particle in a
        given iteration, so it is computed once in O(N*D) rather than
        recomputed per particle in O(N^2*D) (a research-prototype
        inefficiency fixed here; see CHANGELOG.md)::

            C_i = sum_j w_j (P_j - X_i) / sum_j w_j
                = weighted_mean(P) - X_i
        """
        w = self.w_start - (self.w_start - self.w_end) * (t / self.T_max)
        omega = 1.0 / (self.epsilon + self.P_fitness)
        weighted_mean_P = (omega[:, None] * self.P).sum(axis=0) / omega.sum()
        C = weighted_mean_P[None, :] - self.X

        r1 = self.rng.uniform(0, 1, (self.N, self.D))
        r2 = self.rng.uniform(0, 1, (self.N, self.D))
        self.V = w * self.V + self.c1 * r1 * (self.P - self.X) + self.c2 * r2 * C
        self.X = self._clamp(self.X + self.V)
        self.fitness = self._evaluate(self.X)

    def _spawning_phase(self) -> None:
        """Local exploitation: a greedy Gaussian mutation applied only to
        above-median individuals. Only evaluates the individuals it
        perturbs, rather than the whole population."""
        median_fitness = np.median(self.fitness)
        idx = np.where(self.fitness < median_fitness)[0]
        if idx.size == 0:
            return

        dist_to_G = np.linalg.norm(self.X[idx] - self.G, axis=1, keepdims=True)
        # abs(): a negative (S_min - S_prev) leaves a zero-mean Gaussian's
        # distribution unchanged, but a signed "scale" is confusing to
        # read; magnitude is what determines the mutation's spread.
        scale = np.abs(self.s_min - self.S_prev) * dist_to_G * self.rng.uniform(0, 1, (idx.size, 1))
        spawn = self._clamp(self.X[idx] + scale * self.rng.normal(size=(idx.size, self.D)))
        spawn_fitness = self._evaluate(spawn)

        improved = spawn_fitness < self.fitness[idx]
        better_idx = idx[improved]
        self.X[better_idx] = spawn[improved]
        self.fitness[better_idx] = spawn_fitness[improved]

    def _return_phase(self, S: float) -> None:
        """Diversification: replace the worst ``floor(N*S)`` individuals
        with Gaussian perturbations of the incumbent best."""
        n_replace = int(self.N * S)
        if n_replace == 0:
            return
        worst_idx = np.argsort(self.fitness)[-n_replace:]
        new_X = self._clamp(
            self.G[None, :] + self.eta * self.rng.normal(size=(n_replace, self.D)) * (self.X_max - self.X_min)
        )
        self.X[worst_idx] = new_X
        self.fitness[worst_idx] = self._evaluate(new_X)

    def _update_bests(self) -> None:
        improved = self.fitness < self.P_fitness
        self.P[improved] = self.X[improved]
        self.P_fitness[improved] = self.fitness[improved]
        g_idx = int(np.argmin(self.P_fitness))
        if self.P_fitness[g_idx] < self.G_fitness:
            self.G = self.P[g_idx].copy()
            self.G_fitness = float(self.P_fitness[g_idx])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(self, t: int) -> float:
        """Run a single iteration (1-indexed) and return the salinity S(t).

        Exposed for pipelines that need fine-grained control -- e.g. to
        interleave HFMO iterations with other work, or to implement
        custom early stopping on ``self.G_fitness`` between calls.
        """
        S = self._update_salinity(t)
        if S > self.S_prev:
            self._return_phase(S)
        if S > 0.7:
            self._marine_phase()
        elif 0.3 < S <= 0.7:
            self._migration_phase(t)
        else:
            self._spawning_phase()
        self._update_bests()
        self.S_prev = S
        return S

    def optimize(self, n_iterations: Optional[int] = None) -> HFMOResult:
        """Run the optimizer and return an :class:`HFMOResult`.

        Parameters
        ----------
        n_iterations : int, optional
            Number of further iterations to run. Defaults to the
            ``n_iterations`` given at construction time on the *first*
            call. Calling :meth:`optimize` again continues the salinity
            schedule from where it left off (the internal iteration
            counter persists across calls) rather than restarting it at
            ``t=1`` -- so ``opt.optimize(300); opt.optimize(200)`` behaves
            like a single 500-iteration run, split in two so you can
            inspect or checkpoint state in between.
        """
        n_iter = n_iterations if n_iterations is not None else self.T_max
        history = np.empty(n_iter, dtype=float)
        for i in range(n_iter):
            self._t += 1
            self.step(self._t)
            history[i] = self.G_fitness
        return HFMOResult(
            best_x=self.G.copy(),
            best_f=self.G_fitness,
            history=history,
            n_evaluations=self.n_evaluations,
            n_iterations=n_iter,
        )


def minimize(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: BoundsLike,
    dim: Optional[int] = None,
    n_particles: int = 50,
    n_iterations: int = 500,
    vectorized: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
    **kwargs,
) -> HFMOResult:
    """One-shot convenience wrapper, in the style of
    :func:`scipy.optimize.minimize` / :func:`scipy.optimize.differential_evolution`.

    Equivalent to constructing an :class:`HFMO` instance and calling
    :meth:`HFMO.optimize`. Prefer the class directly if you need to inspect
    intermediate state, warm-start a run, or call :meth:`HFMO.step` yourself.

    Examples
    --------
    >>> import numpy as np
    >>> from hfmo import minimize
    >>> result = minimize(lambda x: np.sum(x**2, axis=1), bounds=[(-5, 5)] * 5, seed=0)
    >>> result.best_f < 1e-2
    True
    """
    optimizer = HFMO(
        func=func,
        bounds=bounds,
        dim=dim,
        n_particles=n_particles,
        n_iterations=n_iterations,
        vectorized=vectorized,
        seed=seed,
        **kwargs,
    )
    return optimizer.optimize()
