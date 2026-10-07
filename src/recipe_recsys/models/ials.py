"""Implicit ALS matrix factorization (Hu, Koren & Volinsky, 2008).

Every user and recipe gets a `factors`-dimensional embedding. The model
minimizes, over *all* user x item pairs (not only observed ones):

    sum_ui c_ui * (p_ui - x_u . y_i)^2 + reg * (|X|^2 + |Y|^2)

where p_ui = 1 if the user cooked the recipe (else 0) and the confidence
c_ui = 1 + alpha for observed pairs (else 1). Unobserved pairs act as weak
negatives, which is what makes this work without explicit negative sampling.

Alternating least squares: with item embeddings fixed, each user's embedding
has a closed-form solution, and vice versa. The trick that keeps this fast is
that the sum over all items splits into a shared Gram matrix Y^T Y plus a
correction over the user's few observed items:

    x_u = (Y^T Y + alpha * Y_u^T Y_u + reg * I)^-1  (1 + alpha) Y_u^T 1
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from recipe_recsys.dataset import InteractionData
from recipe_recsys.models.base import Recommender


class IALSRecommender(Recommender):
    name = "ials"

    def __init__(
        self,
        factors: int = 64,
        reg: float = 0.1,
        alpha: float = 10.0,
        iterations: int = 15,
        seed: int = 42,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, data: InteractionData, verbose: bool = False) -> IALSRecommender:
        rng = np.random.default_rng(self.seed)
        scale = 0.01
        self._user_factors = (rng.standard_normal((data.n_users, self.factors)) * scale).astype(
            np.float32
        )
        self._item_factors = (rng.standard_normal((data.n_items, self.factors)) * scale).astype(
            np.float32
        )

        user_items = data.matrix.tocsr()
        item_users = data.matrix.T.tocsr()
        for iteration in range(self.iterations):
            start = time.perf_counter()
            self._user_factors = self._solve(user_items, self._item_factors)
            self._item_factors = self._solve(item_users, self._user_factors)
            if verbose:
                print(f"  ials iteration {iteration + 1}: {time.perf_counter() - start:.1f}s")
        return self

    def _solve(self, interactions: sp.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        """Closed-form update of every row's embedding given the other side's embeddings."""
        f = self.factors
        gram = fixed.T @ fixed + self.reg * np.eye(f, dtype=np.float32)
        solved = np.zeros((interactions.shape[0], f), dtype=np.float32)
        indptr, indices = interactions.indptr, interactions.indices
        for row in range(interactions.shape[0]):
            observed = indices[indptr[row] : indptr[row + 1]]
            if len(observed) == 0:
                continue
            Y_u = fixed[observed]
            A = gram + self.alpha * (Y_u.T @ Y_u)
            b = (1 + self.alpha) * Y_u.sum(axis=0)
            solved[row] = np.linalg.solve(A, b)
        return solved

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self._user_factors[user_rows] @ self._item_factors.T
