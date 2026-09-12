# Changelog

## [0.1.0] -- 2026-09-12

The PyPI distribution is named `hfmo` (previously planned as `hfmo-optimizer`
before this version's first release; the import name has always been
`hfmo`, i.e. `import hfmo`, `pip install hfmo`).

Initial public release. This is a corrected, vectorized re-implementation of
the research prototype (`HFMO_Algorithm.ipynb`) used during development of
the accompanying manuscript. The algorithm -- the four phases, the salinity
schedule, and every default parameter value -- is unchanged. What changed is
implementation correctness, performance, and reproducibility:

### Fixed

- **Levy-flight sampling drew from the wrong distribution.** The prototype
  called `np.random.normal(0, sigma_u**2, D)`, but a `Generator.normal`'s
  second argument is a *standard deviation*, not a variance. Mantegna's
  algorithm requires `u ~ N(0, sigma_u^2)`, i.e. `scale=sigma_u`. Squaring
  it drew steps with the wrong spread (for the default `beta=1.5`, standard
  deviation ~0.485 instead of the intended ~0.697), so the marine phase was
  not actually taking Levy-stable steps. Fixed in `_generate_levy_batch`.
- **No random-seed support.** The prototype called `np.random` module-level
  functions directly, so a run could not be exactly reproduced. `HFMO` now
  takes a `seed` parameter (an int or an existing `numpy.random.Generator`)
  and threads a single `Generator` through every source of randomness.
- **Dead code in the marine phase.** `self.V[i] = self.X[i] - self.X[i] + step`
  computed an identically-zero term before adding the step -- almost
  certainly a refactoring leftover. Simplified to `self.V = self.alpha * step`.

### Changed (performance, not behavior)

- **`migration_phase` was O(N^2*D); it is now O(N*D).** The per-particle
  "river current" term `C_i = sum_j w_j (P_j - X_i) / sum_j w_j` was
  recomputed for every particle by summing over the whole population again
  each time. Since `C_i = weighted_mean(P) - X_i`, the weighted mean of `P`
  is the same for every particle in a given iteration and only needs to be
  computed once. This is the single biggest performance fix in this release
  and matters most at larger population sizes.
- **Fitness was re-evaluated redundantly.** The prototype called the
  objective function on the entire population twice per iteration
  regardless of which phase ran or whether anything had changed. The
  spawning and return phases now update `self.fitness` only for the
  individuals they actually touch (and only evaluate those individuals in
  the first place); the marine and migration phases -- which do move every
  individual -- still evaluate the full population once. `n_evaluations` on
  the result is the true count, useful for reporting a fixed-budget
  comparison against other optimizers.
- **All four phases are now fully vectorized.** The prototype looped over
  individuals in Python (`for i in range(self.N)`) inside every phase; each
  phase is now a batched NumPy operation.

### Added

- A clean, documented public API: the `HFMO` class for fine-grained control
  (including a `step(t)` method for interleaving iterations with other
  pipeline logic) and a `minimize(func, bounds, ...)` convenience function
  in the style of `scipy.optimize.minimize` / `differential_evolution`.
- Two bounds-calling conventions: `bounds=(low, high)` (arrays or, with
  `dim=`, scalars) or the scipy-style `bounds=[(low_1, high_1), ...]`.
- Support for non-vectorized objective functions via `vectorized=False`.
- An `HFMOResult` with `best_x`, `best_f`, `history` (best-so-far per
  iteration, for convergence plots or early stopping), `n_evaluations`, and
  `n_iterations`.
- Input validation (`beta` range, minimum population size, malformed
  bounds) with clear error messages instead of silent misbehavior.
- A test suite covering convergence, reproducibility, both bounds
  conventions, bound feasibility throughout a run, evaluation counting, and
  the non-vectorized objective path.

### Not changed

- The **salinity schedule** `S(t) = |sin(pi t / T_max)| * exp(-gamma t / T_max)`
  and the phase thresholds are unchanged. Note that it is a deterministic
  function of the iteration index alone -- see the README for what that
  does and does not imply about "adaptivity".
- **Default parameter values** (`w_start=0.9`, `w_end=0.4`, `c1=c2=1.7`,
  `s_min=0.1`, `beta=1.5`, `alpha_scale=0.01`, `eta=0.1`, `gamma=0.2`) are
  unchanged from the prototype.
- The **`spawning_phase` mutation scale**, `(s_min - S_prev)`, can be
  negative; this release takes its absolute value when scaling the
  perturbation for clarity, which leaves the (zero-mean, symmetric)
  distribution of the perturbation itself unchanged. Whether the schedule
  *should* make the spawning-phase mutation shrink monotonically as the
  population approaches `s_min` is a design question, not an implementation
  bug, and is left open -- see the README's "Known limitations" section.
