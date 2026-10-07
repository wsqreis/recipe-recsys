"""Ranking metrics with binary relevance.

All functions take one user's ranked recommendations and the set of items
that user actually interacted with in the evaluation window.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hits_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> np.ndarray:
    return np.fromiter((item in relevant for item in ranked[:k]), dtype=bool)


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the user's relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    return float(hits_at_k(ranked, relevant, k).sum() / len(relevant))


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """1 if at least one relevant item is in the top-k."""
    return float(hits_at_k(ranked, relevant, k).any())


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Like recall, but a hit at position 1 is worth more than a hit at position k."""
    if not relevant:
        return 0.0
    hits = hits_at_k(ranked, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts[: len(hits)]).sum())
    idcg = float(discounts[: min(len(relevant), k)].sum())
    return dcg / idcg


def catalog_coverage(recommendations: Sequence[Sequence[int]], n_items: int) -> float:
    """Share of the catalog that is recommended to at least one user."""
    if n_items == 0:
        return 0.0
    recommended = {item for recs in recommendations for item in recs}
    return len(recommended) / n_items
