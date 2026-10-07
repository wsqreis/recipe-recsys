"""Non-personalized baselines. Any real model has to beat these."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recipe_recsys.dataset import InteractionData
from recipe_recsys.models.base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, data: InteractionData) -> RandomRecommender:
        self._n_items = data.n_items
        self._rng = np.random.default_rng(self.seed)
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self._rng.random((len(user_rows), self._n_items), dtype=np.float32)


class PopularityRecommender(Recommender):
    """Most-cooked recipes overall."""

    name = "popularity"

    def fit(self, data: InteractionData) -> PopularityRecommender:
        self._scores = np.asarray(data.matrix.sum(axis=0), dtype=np.float32).ravel()
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(user_rows), len(self._scores))).copy()


class RecentPopularityRecommender(Recommender):
    """Most-cooked recipes in the last `days` of the training window.

    Trends shift over 18 years of data, so recent popularity is usually a much
    stronger baseline than all-time popularity under a temporal split.
    """

    name = "recent_popularity"

    def __init__(self, days: int = 180):
        self.days = days

    def fit(self, data: InteractionData) -> RecentPopularityRecommender:
        dates = data.interactions["date"]
        recent = data.interactions[dates >= dates.max() - pd.Timedelta(days=self.days)]
        counts = np.bincount(recent["item_idx"], minlength=data.n_items).astype(np.float32)
        # Tiny all-time tie-breaker so items outside the window are still ordered.
        all_time = np.asarray(data.matrix.sum(axis=0), dtype=np.float32).ravel()
        self._scores = counts + all_time / (all_time.max() + 1)
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(user_rows), len(self._scores))).copy()
