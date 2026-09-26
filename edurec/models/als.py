"""Matrix factorisation for implicit feedback (Hu, Koren and Volinsky, 2008), written with NumPy.

MARS only records that a learner opened or watched a video, not how much they liked it. Hu et al.
treat every opened video as a preference of 1 with confidence 1 + alpha, and every other video as a
preference of 0 with confidence 1. Learner and video factors are then found with alternating least
squares: fix the videos and solve for each learner exactly, then fix the learners and solve for each
video, and repeat.
"""

import numpy as np

from ..config import SEED
from .base import Recommender, learner_item_matrix


class ImplicitALS(Recommender):
    name = "ALS matrix factorisation"

    def __init__(self, factors: int = 32, regularisation: float = 0.1, alpha: float = 10.0,
                 iterations: int = 15, seed: int = SEED):
        self.factors = factors
        self.regularisation = regularisation
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, train, catalogue):
        matrix, _ = learner_item_matrix(train, catalogue.n_items)
        rng = np.random.default_rng(self.seed)
        users = rng.normal(0, 0.01, (matrix.shape[0], self.factors))
        items = rng.normal(0, 0.01, (matrix.shape[1], self.factors))
        by_user, by_item = matrix.tocsr(), matrix.T.tocsr()
        for _ in range(self.iterations):
            users = self._solve_rows(by_user, items)
            items = self._solve_rows(by_item, users)
        self.item_factors = items
        self.gram = self._gram(items)
        return self

    def _gram(self, fixed):
        return fixed.T @ fixed + self.regularisation * np.eye(self.factors)

    def _solve_one(self, observed, fixed, gram):
        # Only the observed entries change the system: YtCY = YtY + Yt(C - I)Y, and C - I = alpha there.
        y = fixed[observed]
        a = gram + self.alpha * (y.T @ y)
        b = (1.0 + self.alpha) * y.sum(axis=0)
        return np.linalg.solve(a, b)

    def _solve_rows(self, rows, fixed):
        gram = self._gram(fixed)
        solved = np.zeros((rows.shape[0], self.factors))
        for r in range(rows.shape[0]):
            observed = rows.indices[rows.indptr[r]:rows.indptr[r + 1]]
            if len(observed):
                solved[r] = self._solve_one(observed, fixed, gram)
        return solved

    def score(self, history, weights=None):
        history = np.unique(np.asarray(history, dtype=int))
        if len(history) == 0:
            return np.zeros(self.item_factors.shape[0])
        # "Folding in" a learner: the same exact solve as in training, so a new learner can be scored
        # without refitting the whole model.
        learner = self._solve_one(history, self.item_factors, self.gram)
        return self.item_factors @ learner
