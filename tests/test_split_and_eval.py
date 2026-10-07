import numpy as np
import pandas as pd

from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import build_ground_truth, evaluate, top_k
from recipe_recsys.models import ItemKNNRecommender, PopularityRecommender
from recipe_recsys.split import k_core, temporal_split


def _interactions(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 50, n),
            "recipe_id": rng.integers(0, 80, n),
            "date": pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 3650, n), "D"),
        }
    ).sort_values("date")


def test_temporal_split_never_trains_on_the_future():
    split = temporal_split(_interactions())
    assert split.train["date"].max() < split.val["date"].min()
    assert split.val["date"].max() < split.test["date"].min()
    assert len(split.train) + len(split.val) + len(split.test) == 1000


def test_k_core_enforces_minimum_counts():
    df = k_core(_interactions(), min_user=10, min_item=8)
    assert not df.empty
    assert df["user_id"].value_counts().min() >= 10
    assert df["recipe_id"].value_counts().min() >= 8


def test_ground_truth_excludes_cold_start_and_seen_items():
    train = InteractionData.from_frame(
        pd.DataFrame({"user_id": [1, 1, 2], "recipe_id": [10, 20, 10], "date": pd.Timestamp(0)})
    )
    eval_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "recipe_id": [20, 99, 20, 10],  # seen item, cold item, valid, cold user
            "date": pd.Timestamp(1),
        }
    )
    truth, stats = build_ground_truth(train, eval_df)

    assert truth == {train.user_index[2]: {train.item_index[20]}}
    assert stats.cold_user_interactions == 1
    assert stats.cold_item_interactions == 1


def test_top_k_is_sorted_and_skips_masked_items():
    scores = np.array([[0.1, 0.9, -np.inf, 0.5]], dtype=np.float32)
    [ranked] = top_k(scores, k=4)
    assert ranked.tolist() == [1, 3, 0]


def test_itemknn_learns_co_occurrence():
    # Users who cook A also cook B; user 9 cooked only A and should get B first.
    rows = [(u, "A") for u in range(5)] + [(u, "B") for u in range(5)]
    rows += [(u, "C") for u in range(5, 9)] + [(9, "A")]
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id"]).assign(date=pd.Timestamp(0))
    data = InteractionData.from_frame(df)

    model = ItemKNNRecommender(neighbors=10, shrink=0).fit(data)
    [ranked] = top_k(model.score(np.array([data.user_index[9]])), k=1)
    assert data.item_ids[ranked[0]] == "B"


def test_evaluate_end_to_end_with_popularity():
    split = temporal_split(_interactions(5000))
    train = InteractionData.from_frame(split.train)
    truth, _ = build_ground_truth(train, split.val)
    result = evaluate(PopularityRecommender().fit(train), train, truth, ks=(5, 10))

    assert 0 <= result.metrics["recall@5"] <= result.metrics["recall@10"] <= 1
    assert 0 <= result.metrics["ndcg@10"] <= 1
