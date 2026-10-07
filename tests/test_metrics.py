import math

import pytest

from recipe_recsys import metrics


def test_recall_counts_share_of_relevant_items_found():
    assert metrics.recall_at_k([1, 2, 3], {1, 3, 9, 10}, k=3) == 0.5
    assert metrics.recall_at_k([1, 2, 3], {1, 3, 9, 10}, k=1) == 0.25


def test_hit_rate_is_binary():
    assert metrics.hit_rate_at_k([5, 6, 7], {7}, k=3) == 1.0
    assert metrics.hit_rate_at_k([5, 6, 7], {7}, k=2) == 0.0


def test_ndcg_rewards_hits_at_the_top():
    assert metrics.ndcg_at_k([1, 2, 3], {1}, k=3) == 1.0
    assert metrics.ndcg_at_k([2, 3, 1], {1}, k=3) == pytest.approx(1 / math.log2(4))
    assert metrics.ndcg_at_k([2, 3, 4], {1}, k=3) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # 5 relevant items but only 2 slots: a perfect top-2 must score 1.
    assert metrics.ndcg_at_k([1, 2], {1, 2, 3, 4, 5}, k=2) == pytest.approx(1.0)


def test_empty_relevant_set_scores_zero():
    assert metrics.recall_at_k([1, 2], set(), k=2) == 0.0
    assert metrics.ndcg_at_k([1, 2], set(), k=2) == 0.0


def test_catalog_coverage():
    assert metrics.catalog_coverage([[0, 1], [1, 2]], n_items=10) == 0.3
