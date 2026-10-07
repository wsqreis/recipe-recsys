"""Item-based collaborative filtering ("people who cooked X also cooked Y")."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recipe_recsys.dataset import InteractionData
from recipe_recsys.models.base import Recommender


class ItemKNNRecommender(Recommender):
    """Cosine similarity between item columns, pruned to the top `neighbors` per item.

    `shrink` dampens similarities computed from very few co-occurrences, which
    are mostly noise in a dataset this sparse.
    """

    name = "itemknn"

    def __init__(self, neighbors: int = 50, shrink: float = 50.0, block_size: int = 2048):
        self.neighbors = neighbors
        self.shrink = shrink
        self._block_size = block_size

    def fit(self, data: InteractionData) -> ItemKNNRecommender:
        X = data.matrix.tocsc()
        XT = X.T.tocsr()
        norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel()).astype(np.float32)

        # The full item x item co-occurrence matrix has ~10^8 non-zeros here, so it
        # is built one block of rows at a time and pruned before the next block.
        blocks = []
        for start in range(0, data.n_items, self._block_size):
            stop = min(start + self._block_size, data.n_items)
            co = (XT[start:stop] @ X).tocoo()
            rows, cols = co.row, co.col
            keep = rows + start != cols  # an item is not its own neighbor
            rows, cols = rows[keep], cols[keep]
            sims = co.data[keep] / (norms[rows + start] * norms[cols] + self.shrink)
            block = sp.csr_matrix(
                (sims.astype(np.float32), (rows, cols)), shape=(stop - start, data.n_items)
            )
            blocks.append(_top_k_per_row(block, self.neighbors))

        self._sim = sp.vstack(blocks, format="csr")
        self._user_items = data.matrix
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return (self._user_items[user_rows] @ self._sim).toarray().astype(np.float32, copy=False)


def _top_k_per_row(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    m = m.tocsr()
    m.sort_indices()
    indptr, data = m.indptr, m.data
    keep = np.zeros(len(data), dtype=bool)
    for i in range(m.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end - start <= k:
            keep[start:end] = True
        else:
            top = np.argpartition(data[start:end], -k)[-k:]
            keep[start + top] = True

    row_idx = np.repeat(np.arange(m.shape[0]), np.diff(indptr))
    return sp.csr_matrix(
        (data[keep], (row_idx[keep], m.indices[keep])), shape=m.shape, dtype=np.float32
    )
