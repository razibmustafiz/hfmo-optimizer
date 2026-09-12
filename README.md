# Hilsha Fish Migration Optimization (HFMO): A Novel Optimization Algorithm Based on Hilsha Fish Migration Pattern

**Hilsha Fish Migration Optimization (HFMO)** — a phase-scheduled hybrid metaheuristic for
continuous, bound-constrained optimization, inspired by the salinity-driven migration of Hilsha
(*Tenualosa ilisha*) between marine and freshwater habitats in the Bay of Bengal.

```bash
pip install hfmo
```

```python
import numpy as np
from hfmo import minimize

result = minimize(lambda x: np.sum(x**2, axis=1), bounds=[(-100, 100)] * 30, seed=0)
print(result.best_f, result.best_x)
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/razibmustafiz/hfmo-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/razibmustafiz/hfmo-optimizer/actions/workflows/tests.yml)

---

## Contents

- [What is HFMO](#what-is-hfmo)
- [How it works](#how-it-works)
- [Attribution — what's actually novel here](#attribution--whats-actually-novel-here)
- [Installation](#installation)
- [Usage](#usage)
- [Using HFMO in an optimization pipeline](#using-hfmo-in-an-optimization-pipeline)
- [API reference](#api-reference)
- [Parameters](#parameters)
- [Benchmark results](#benchmark-results)
- [Known limitations](#known-limitations)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## What is HFMO

Hilsha exhibit a distinctive anadromous life cycle: they spend most of their life foraging in
the marine waters of the Bay of Bengal, then migrate upstream into the Ganges-Brahmaputra-Meghna
delta to spawn in freshwater as salinity drops, before returning to sea. HFMO uses this cycle as
a template for scheduling a population-based search: a single scalar "salinity" signal, computed
from the iteration index, switches the population between four search behaviors over the course
of a run — a wide-ranging exploratory phase early on, a directed phase, an intense local-refinement
phase, and a periodic diversification phase that resets part of the population.

It is implemented here as a drop-in, `scipy.optimize`-style optimizer: give it a bound-constrained
objective function, get back a best solution and a convergence history.

## How it works

HFMO maintains a population of `N` candidate solutions ("fish") in a `D`-dimensional search space.
Each iteration `t`, a **salinity** value

```
S(t) = |sin(pi * t / T_max)| * exp(-gamma * t / T_max)
```

determines which of four phases runs:

| Condition | Phase | Behavior |
|---|---|---|
| `S(t) > 0.7` | **Marine** | Global exploration via Lévy-flight steps |
| `0.3 < S(t) <= 0.7` | **Migration** | Directed search: a PSO-style velocity update |
| `S(t) <= 0.3` | **Spawning** | Local exploitation: Gaussian mutation of above-median individuals |
| `S(t) > S(t-1)` | **Return** | Diversification: the worst individuals are replaced, layered independently on top of whichever of the three phases above also ran |

Because `t` only ever ranges over `[1, T_max]`, `sin(pi t / T_max)` stays non-negative throughout
a run, so `S(t)` traces exactly **one** rise and fall — not a repeating oscillation. Concretely,
for the defaults (`T_max=500`, `gamma=0.2`): the run opens in the **spawning** phase (`S(1) ≈
0.006`), rises through **migration** by around iteration 50, reaches **marine** by around
iteration 133, peaks near the run's midpoint, falls back through migration by around iteration
351 and back to spawning by around iteration 442. The **return** phase, triggered whenever
`S(t) > S(t-1)`, fires on every one of those rising iterations — roughly the first half of the
run — and never during the falling half. Exact transition points depend on `gamma` and `T_max`;
the shape does not.

**Important characteristic, stated plainly:** `S(t)` is a deterministic function of the iteration
index alone. It does not read population diversity, fitness stagnation, or any other property of
the search state. Every run — on every problem, at every population size — executes the identical
phase sequence at the identical iteration indices, single-hump shape included. This is closer to
a fixed annealing-style schedule than to a state-dependent controller, and you should not expect
the schedule to adapt to how a particular run is actually going. See
[Known limitations](#known-limitations).

## Attribution — what's actually novel here

Each phase update is, on its own, an established mechanism from the metaheuristics literature.
HFMO's contribution is the specific composition of these mechanisms under the salinity schedule
above, not the mechanisms themselves — and we think a package like this should say so plainly
rather than let a biological metaphor imply more originality than there is.

| Phase | Mechanism | Origin |
|---|---|---|
| Marine | Lévy-flight step via Mantegna's algorithm | Mantegna (1994); used in Cuckoo Search, Yang & Deb (2009) |
| Migration | PSO velocity update with linearly-decaying inertia | Kennedy & Eberhart (1995); Shi & Eberhart (1998) |
| Migration (social term) | Fitness-weighted average over *all* personal bests, not just the global best | Fully Informed Particle Swarm, Mendes, Kennedy & Neves (2004) |
| Spawning | Gaussian local-search mutation scaled by distance to the incumbent best | Bare-bones PSO, Kennedy (2003) |
| Return | Elite-guided reinitialization of the worst individuals | Common to scout-bee and migration-based algorithms generally |

## Installation

```bash
# From PyPI
pip install hfmo

# Or directly from GitHub
pip install git+https://github.com/razibmustafiz/hfmo-optimizer.git

# Or, for local development
git clone https://github.com/razibmustafiz/hfmo-optimizer.git
cd hfmo-optimizer
pip install -e ".[dev,examples]"
```

Requires Python 3.8+, NumPy, and SciPy.

## Usage

```python
import numpy as np
from hfmo import minimize

def rastrigin(x: np.ndarray) -> np.ndarray:
    """x: shape (n_particles, D) -> shape (n_particles,)"""
    A = 10
    return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

result = minimize(
    rastrigin,
    bounds=[(-5.12, 5.12)] * 10,   # one (low, high) pair per dimension, as in scipy.optimize
    n_particles=60,
    n_iterations=500,
    seed=0,                         # always set a seed for reproducible results
)

print(result.best_f)        # best fitness found
print(result.best_x)        # the corresponding solution
print(result.history)       # best-so-far fitness at each iteration
print(result.n_evaluations) # total objective-function calls used
```

Use the `HFMO` class directly when you need more control — inspecting intermediate state,
running in chunks with a custom stopping rule, or objective functions that only accept a single
candidate at a time:

```python
from hfmo import HFMO

optimizer = HFMO(rastrigin, bounds=[(-5.12, 5.12)] * 10, n_particles=60, n_iterations=500, seed=0)
result = optimizer.optimize()

# Run more iterations later -- continues the same schedule, doesn't restart it
more = optimizer.optimize(n_iterations=100)
```

See [`examples/`](examples/) for complete, runnable scripts:

- [`01_basic_usage.py`](examples/01_basic_usage.py) — the two calling styles above
- [`02_pipeline_integration.py`](examples/02_pipeline_integration.py) — hyperparameter tuning for a scikit-learn model
- [`03_compare_with_scipy.py`](examples/03_compare_with_scipy.py) — a budget-matched comparison against `scipy.optimize.differential_evolution`
- [`04_convergence_plot.py`](examples/04_convergence_plot.py) — chunked runs, early stopping, and a convergence plot

## Using HFMO in an optimization pipeline

HFMO's public API is deliberately shaped like `scipy.optimize.minimize` /
`scipy.optimize.differential_evolution`, so it drops into the same place in a pipeline:

- **Hyperparameter tuning** — wrap a cross-validated model-scoring function as the objective and
  search a bounded hyperparameter space (see `examples/02_pipeline_integration.py`).
- **Simulation calibration** — minimize the discrepancy between a simulator's output and observed
  data over its input parameters.
- **As one candidate among several** — since `HFMOResult` exposes `n_evaluations`, you can run
  HFMO alongside other optimizers (scipy's, or your own) under an explicitly matched evaluation
  budget and compare fairly, rather than by iteration count or wall-clock time alone (see
  `examples/03_compare_with_scipy.py`).
- **Long-running or checkpointed searches** — call `HFMO.optimize(n_iterations=...)` repeatedly
  (it continues the schedule rather than restarting it) to interleave the search with logging,
  checkpointing, or an early-stopping rule based on `optimizer.G_fitness` (see
  `examples/04_convergence_plot.py`).

If your objective only accepts one candidate at a time rather than a batch, pass
`vectorized=False` and HFMO will call it once per candidate internally.

## API reference

### `hfmo.minimize(func, bounds, dim=None, n_particles=50, n_iterations=500, vectorized=True, seed=None, **kwargs) -> HFMOResult`

One-shot convenience function. Constructs an `HFMO` instance and calls `.optimize()`. Any
additional keyword arguments (`w_start`, `c1`, `beta`, ...) are forwarded to `HFMO.__init__`.

### `hfmo.HFMO(func, bounds, dim=None, n_particles=50, n_iterations=500, ...)`

The optimizer class. Key methods:

- `optimize(n_iterations=None) -> HFMOResult` — run (further) iterations.
- `step(t) -> float` — run a single iteration `t` and return the salinity `S(t)`; for pipelines
  that need to interleave HFMO with other work at the granularity of a single iteration.

Key attributes after construction or during a run: `X` (population positions), `fitness`,
`G` / `G_fitness` (global best and its fitness), `n_evaluations`.

### `hfmo.HFMOResult`

A dataclass: `best_x`, `best_f`, `history` (best-so-far fitness per iteration, as a NumPy array),
`n_evaluations`, `n_iterations`.

### `bounds`

Either convention works:

```python
bounds=[(-5, 5)] * D                       # scipy-style: one (low, high) pair per dimension
bounds=(np.full(D, -5), np.full(D, 5))     # (low_array, high_array)
bounds=(-5, 5); dim=D                      # scalar low/high, dim required
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_particles` | 50 | Population size |
| `n_iterations` | 500 | Iterations to run in `optimize()` |
| `w_start`, `w_end` | 0.9, 0.4 | Inertia weight at the start/end of the run (migration phase) |
| `c1`, `c2` | 1.7, 1.7 | Cognitive / social coefficients (migration phase) |
| `s_min` | 0.1 | Salinity threshold governing the spawning-phase mutation scale |
| `beta` | 1.5 | Lévy-flight stability exponent, in `(1, 2]` |
| `alpha_scale` | 0.01 | Scales the Lévy step relative to the search-space width |
| `eta` | 0.1 | Scales the return-phase reinitialization perturbation |
| `gamma` | 0.2 | Decay rate of the salinity schedule, in `[0, 0.5]` |
| `epsilon` | 1e-6 | Prevents division by zero when fitness-weighting personal bests |
| `vectorized` | `True` | Set `False` if `func` takes one candidate at a time |
| `seed` | `None` | Int or `numpy.random.Generator`, for reproducibility |

## Benchmark results

HFMO was evaluated in the study accompanying this repository on two problem classes:

- **QAPLIB** (ten `chr` instances, `chr12a`–`chr20b`, via a random-key encoding — see the
  manuscript for how a continuous optimizer is mapped onto a permutation problem): HFMO obtained
  the best mean solution quality of five compared metaheuristics (HFMO, CSMA, FSS, HHO, AFMO). In
  the one comparison in that study run under a matched iteration budget — HFMO against Fish School
  Search, both at 10,000 iterations — HFMO was better on solution quality, optimality gap, *and*
  runtime simultaneously.
- **CEC-2015** (five composition/hybrid functions, F10–F14, D=10, 20 runs): HFMO ranked mid-table
  among seven compared algorithms (HFMO, AFMO, MSA, HHO, DAIW-PSO, FSS, CSMA), leading on none of
  the five functions.

We are not aware of any evaluation in which HFMO led on both problem classes simultaneously, and
we don't present it as state-of-the-art on either. The manuscript describing this evaluation in
full — including instance-by-instance results, the encoding used for QAPLIB, and the evaluation
protocol's own limitations — is in preparation; this README will be updated with a link once it
is available.

## Known limitations

- **The salinity schedule is not adaptive.** As described above, `S(t)` depends only on the
  iteration index. Making it responsive to a measured population-diversity or stagnation
  statistic is a natural next step and is not yet implemented.
- **No component of the algorithm is novel in isolation** — see
  [Attribution](#attribution--whats-actually-novel-here). Treat HFMO as a particular scheduling
  strategy over established operators, evaluated empirically, rather than as a new search
  operator in its own right.
- **The closest related algorithm is CSMA** (Calico Salmon Migration Algorithm, Min et al. 2023),
  another four-stage migratory-fish metaheuristic with an adaptive population-energy operator. A
  rigorous component-level comparison between the two has not yet been published.
- **This package implements the continuous optimizer only.** Applying it to combinatorial
  problems (as in the QAPLIB evaluation above) requires an encoding step — a random-key /
  `argsort` decoding was used in that study — which is not included here, since it is specific to
  the target problem rather than part of the core algorithm.
- No claim of general superiority is made or implied, consistent with the No Free Lunch theorems
  for optimization (Wolpert & Macready, 1997): the results above describe two specific problem
  families, not optimization in general.

## Development

```bash
git clone https://github.com/razibmustafiz/hfmo-optimizer.git
cd hfmo-optimizer
pip install -e ".[dev]"
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff). The associated manuscript is in preparation; this file will
be updated with a DOI once it is published.

## License

[MIT](LICENSE) © 2026 Mohammad Razib Mustafiz
