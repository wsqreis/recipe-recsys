"""Content-based recommendation from recipe text embeddings.

A user is the mean text embedding of the last `history` recipes they cooked;
recipes are ranked by cosine similarity to that profile. No interaction data
is needed on the recipe side, so new recipes are scored exactly like old ones.
This is the simple baseline that learned content models have to beat.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recipe_recsys.dataset import InteractionData, ItemPool
from recipe_recsys.models.base import Recommender


class TextProfileRecommender(Recommender):
    name = "text_profile"
    needs_text = True

    def __init__(self, history: int = 30):
        self.history = history

    def fit(self, data: InteractionData) -> TextProfileRecommender:
        if data.item_text is None:
            raise ValueError("text_profile needs text embeddings: call data.with_text(vectors)")
        df = data.interactions.sort_values(["user_idx", "date"], kind="stable")
        if self.history > 0:
            df = df.groupby("user_idx").tail(self.history)

        profiles = np.zeros((data.n_users, data.item_text.shape[1]), dtype=np.float32)
        np.add.at(profiles, df["user_idx"].to_numpy(), data.item_text[df["item_idx"].to_numpy()])
        profiles /= np.linalg.norm(profiles, axis=1, keepdims=True).clip(min=1e-12)
        self._profiles = profiles
        self._item_text = data.item_text
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self._profiles[user_rows] @ self._item_text.T

    def score_histories(self, histories: sp.csr_matrix) -> np.ndarray:
        profiles = histories @ self._item_text
        profiles /= np.linalg.norm(profiles, axis=1, keepdims=True).clip(min=1e-12)
        return profiles @ self._item_text.T

    def score_items(self, user_rows: np.ndarray, pool: ItemPool) -> np.ndarray:
        if pool.text is None:
            raise ValueError("text_profile needs text embeddings for the item pool")
        return self._profiles[user_rows] @ pool.text.T
