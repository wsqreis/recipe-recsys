from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from recipe_recsys.dataset import InteractionData, ItemPool


class Recommender(ABC):
    """Scores every catalog item for a batch of users.

    Masking (already-seen items, dietary restrictions) and top-k selection
    live outside the model, so every model gets exactly the same treatment.
    """

    name: str
    # True if `fit` needs `data.item_text` (sentence embeddings of the recipes).
    needs_text: bool = False

    @abstractmethod
    def fit(self, data: InteractionData) -> Recommender: ...

    @abstractmethod
    def score(self, user_rows: np.ndarray) -> np.ndarray:
        """Return a dense (len(user_rows), n_items) float32 array; higher is better."""

    def score_items(self, user_rows: np.ndarray, pool: ItemPool) -> np.ndarray:
        """Score recipes that were not in training: a dense (len(user_rows), len(pool)) array.

        Only content-aware models can; the default says so instead of guessing.
        """
        raise NotImplementedError(f"{self.name} cannot score recipes it was not trained on")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{self.name}({params})"
