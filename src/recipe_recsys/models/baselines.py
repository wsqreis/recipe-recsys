"""Non-personalized baselines. Any real model has to beat these."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recipe_recsys.dataset import InteractionData, ItemPool
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

    def score_items(self, user_rows: np.ndarray, pool: ItemPool) -> np.ndarray:
        # Seeded per batch so the result does not depend on what was scored before
        # (e.g. whether the warm slice ran first).
        rng = np.random.default_rng([self.seed, *user_rows[:1].tolist(), len(user_rows)])
        return rng.random((len(user_rows), len(pool)), dtype=np.float32)


def _days(submitted: pd.Series) -> np.ndarray:
    return (submitted.to_numpy() - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")


class NewestRecommender(Recommender):
    """Most recently submitted recipes first.

    Needs no interactions, so unlike popularity it can rank recipes that were
    never cooked: the baseline that content models must beat on cold items.
    """

    name = "newest"

    def fit(self, data: InteractionData) -> NewestRecommender:
        if data.item_content is None or "submitted" not in data.item_content:
            raise ValueError("newest needs recipe content with a `submitted` date")
        self._scores = _days(data.item_content["submitted"]).astype(np.float32)
        return self

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(user_rows), len(self._scores))).copy()

    def score_items(self, user_rows: np.ndarray, pool: ItemPool) -> np.ndarray:
        scores = _days(pool.content["submitted"]).astype(np.float32)
        return np.broadcast_to(scores, (len(user_rows), len(scores))).copy()


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
