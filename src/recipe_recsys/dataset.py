"""Sparse user x item interaction matrix with id <-> index mappings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class InteractionData:
    matrix: sp.csr_matrix  # users x items, binary float32
    user_ids: np.ndarray
    item_ids: np.ndarray
    # Long format with `user_idx`, `item_idx`, `date` (used by time-aware models).
    interactions: pd.DataFrame
    # Recipe attributes aligned with `item_ids` (used by content-aware models).
    item_content: pd.DataFrame | None = None

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> InteractionData:
        user_codes, user_ids = pd.factorize(df["user_id"], sort=True)
        item_codes, item_ids = pd.factorize(df["recipe_id"], sort=True)
        matrix = sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (user_codes, item_codes)),
            shape=(len(user_ids), len(item_ids)),
        )
        # Collapse repeated (user, item) pairs into a single binary interaction.
        matrix.data[:] = 1.0
        interactions = pd.DataFrame(
            {"user_idx": user_codes, "item_idx": item_codes, "date": df["date"].to_numpy()}
        )
        return cls(matrix, np.asarray(user_ids), np.asarray(item_ids), interactions)

    def with_content(self, recipes: pd.DataFrame) -> InteractionData:
        self.item_content = recipes.set_index("recipe_id").loc[self.item_ids].reset_index()
        return self

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @cached_property
    def user_index(self) -> dict[int, int]:
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @cached_property
    def item_index(self) -> dict[int, int]:
        return {int(r): i for i, r in enumerate(self.item_ids)}
