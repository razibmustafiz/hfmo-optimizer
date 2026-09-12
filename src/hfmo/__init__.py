"""
hfmo — Hilsha Fish Migration Optimization
==========================================

A phase-scheduled hybrid metaheuristic for continuous, bound-constrained
optimization, inspired by the salinity-driven migration of Hilsha
(*Tenualosa ilisha*) between marine and freshwater habitats in the Bay of
Bengal.

Quick start
-----------
>>> from hfmo import minimize
>>> import numpy as np
>>> result = minimize(lambda x: np.sum(x**2, axis=1), bounds=[(-100, 100)] * 10, seed=0)
>>> round(result.best_f, 6) < 1e-3
True

See the project README for the algorithm description, parameter reference,
and worked examples: https://github.com/razibmustafiz/hfmo-optimizer
"""

from .optimizer import HFMO, HFMOResult, minimize

__all__ = ["HFMO", "HFMOResult", "minimize"]
__version__ = "0.1.0"
