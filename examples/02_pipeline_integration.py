"""
Using HFMO inside a typical optimization pipeline: hyperparameter tuning.

This is the shape most "optimization pipeline" use cases actually take --
you have some expensive-ish evaluation function (here, cross-validated
model accuracy) and you want to search a bounded parameter space for the
setting that maximizes it. HFMO minimizes, so we minimize the negative
score.

Requires scikit-learn for this specific example (not a package dependency):
    pip install scikit-learn
"""

import numpy as np

from hfmo import minimize


def main() -> None:
    try:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC
    except ImportError:
        print("This example needs scikit-learn: pip install scikit-learn")
        return

    X, y = make_classification(n_samples=300, n_features=20, random_state=0)

    def objective(params: np.ndarray) -> np.ndarray:
        """params: shape (n_particles, 2) -> columns [log10(C), log10(gamma)].

        HFMO's objective must accept a batch and return one score per row,
        but scikit-learn's cross_val_score evaluates one model at a time --
        so we loop internally here and vectorize only at the boundary.
        This is the usual pattern when wrapping a non-vectorized black-box
        evaluator: keep HFMO's calling convention, batch inside your own
        objective instead of setting vectorized=False (which would cost you
        the ability to e.g. joblib-parallelize the inner loop yourself).
        """
        scores = np.empty(params.shape[0])
        for i, (log_C, log_gamma) in enumerate(params):
            model = SVC(C=10**log_C, gamma=10**log_gamma)
            scores[i] = -cross_val_score(model, X, y, cv=3).mean()  # negate: HFMO minimizes
        return scores

    result = minimize(
        objective,
        bounds=[(-3, 3), (-4, 1)],   # [log10(C) in [1e-3, 1e3], log10(gamma) in [1e-4, 1e1]]
        n_particles=12,               # keep small: each candidate costs a 3-fold CV fit
        n_iterations=25,
        seed=0,
    )

    best_C, best_gamma = 10 ** result.best_x
    print(f"Best cross-validated accuracy: {-result.best_f:.4f}")
    print(f"Best C:     {best_C:.4g}")
    print(f"Best gamma: {best_gamma:.4g}")
    print(f"Model evaluations used: {result.n_evaluations}")


if __name__ == "__main__":
    main()
