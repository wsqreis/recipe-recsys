import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import (
    EvalResult,
    bootstrap_ci,
    build_ground_truth,
    build_new_item_slice,
    build_new_user_slice,
    evaluate,
    evaluate_new_items,
    evaluate_new_users,
    new_item_pool,
    paired_difference,
    top_k,
)
from recipe_recsys.models import (
    IALSRecommender,
    ItemKNNRecommender,
    NewestRecommender,
    PopularityRecommender,
    TextProfileRecommender,
)
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


def test_ials_learns_co_occurrence():
    rows = [(u, "A") for u in range(5)] + [(u, "B") for u in range(5)]
    rows += [(u, "C") for u in range(5, 9)] + [(u, "D") for u in range(5, 9)] + [(9, "A")]
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id"]).assign(date=pd.Timestamp(0))
    data = InteractionData.from_frame(df)

    model = IALSRecommender(factors=4, reg=0.01, alpha=10.0, iterations=20).fit(data)
    scores = model.score(np.array([data.user_index[9]]))[0]
    item = {recipe: i for i, recipe in enumerate(data.item_ids)}
    assert scores[item["B"]] > scores[item["C"]]
    assert scores[item["B"]] > scores[item["D"]]


def test_evaluate_end_to_end_with_popularity():
    split = temporal_split(_interactions(5000))
    train = InteractionData.from_frame(split.train)
    truth, _ = build_ground_truth(train, split.val)
    result = evaluate(PopularityRecommender().fit(train), train, truth, ks=(5, 10))

    assert 0 <= result.metrics["recall@5"] <= result.metrics["recall@10"] <= 1
    assert 0 <= result.metrics["ndcg@10"] <= 1


def test_two_tower_learns_co_occurrence():
    pytest.importorskip("torch")
    from recipe_recsys.data import NUTRITION_COLUMNS
    from recipe_recsys.models.two_tower import TwoTowerRecommender

    # Users cook A then B, or C then D; user 99 just cooked A and should get B before C/D.
    rows = []
    for u in range(40):
        first, second = ("A", "B") if u % 2 == 0 else ("C", "D")
        rows += [(u, first, 0), (u, second, 1)]
    rows.append((99, "A", 0))
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id", "day"])
    df["date"] = pd.Timestamp(0) + pd.to_timedelta(df.pop("day"), "D")
    data = InteractionData.from_frame(df)

    content = pd.DataFrame({"recipe_id": list("ABCD")})
    content["ingredients"] = [["egg"], ["flour"], ["rice"], ["beans"]]
    content["tags"] = [["breakfast"], ["baking"], ["dinner"], ["dinner"]]
    for col in [*NUTRITION_COLUMNS, "minutes", "n_steps", "n_ingredients"]:
        content[col] = 1.0
    data.with_content(content)

    model = TwoTowerRecommender(dim=8, epochs=60, batch_size=16, lr=0.01, min_token_count=1)
    model.fit(data, verbose=False)
    scores = model.score(np.array([data.user_index[99]]))[0]
    item = {recipe: i for i, recipe in enumerate(data.item_ids)}
    assert scores[item["B"]] > max(scores[item["C"]], scores[item["D"]])


def _new_item_setup():
    # Training catalog: A, B (k-core survivors). X was cooked by user 1 in the raw training
    # window but dropped by the k-core; N1/N2 are new; F is submitted after the window ends.
    train_df = pd.DataFrame(
        {"user_id": [1, 1, 2, 1], "recipe_id": ["A", "B", "A", "X"], "date": pd.Timestamp(0)}
    )
    train = InteractionData.from_frame(train_df.iloc[:3])
    recipes = pd.DataFrame(
        {
            "recipe_id": ["A", "B", "X", "N1", "N2", "F"],
            "submitted": pd.to_datetime(["2000", "2000", "2000", "2010", "2011", "2030"]),
        }
    )
    return train, train_df, recipes


def test_new_item_pool_keeps_only_recipes_submitted_in_the_window():
    train, _, recipes = _new_item_setup()
    # X is outside the catalog but older than the window (long tail); F is from the future.
    pool = new_item_pool(train, recipes, pd.Timestamp("2005"), pd.Timestamp("2020"))
    assert pool.item_ids.tolist() == ["N1", "N2"]
    # A wide window still never returns recipes from the training catalog.
    pool = new_item_pool(train, recipes, pd.Timestamp("2000"), pd.Timestamp("2020"))
    assert pool.item_ids.tolist() == ["X", "N1", "N2"]


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_users():
    rng = np.random.default_rng(0)
    small, large = rng.random(50), rng.random(5000)
    low, high = bootstrap_ci(small)
    assert low < small.mean() < high
    low_l, high_l = bootstrap_ci(large)
    assert high_l - low_l < high - low


def test_new_item_slice_excludes_seen_and_unknown_users():
    train, train_df, recipes = _new_item_setup()
    pool = new_item_pool(train, recipes, pd.Timestamp("2000"), pd.Timestamp("2020"))
    eval_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 9, 2],
            # seen before the k-core, valid, valid, cold user, warm recipe (other slice)
            "recipe_id": ["X", "N1", "N2", "N1", "B"],
            "date": pd.Timestamp(1),
        }
    )
    cold = build_new_item_slice(train, train_df, eval_df, pool)
    pos = {r: i for i, r in enumerate(pool.item_ids)}
    assert cold.truth == {
        train.user_index[1]: {pos["N1"]},
        train.user_index[2]: {pos["N2"]},
    }


def test_text_profile_ranks_new_recipes_by_content():
    train, train_df, recipes = _new_item_setup()
    # 2-d "text embeddings": A and B point one way, N1 the same way, N2 the other way.
    train.with_text(np.array([[1, 0], [1, 0]], dtype=np.float32))
    pool = new_item_pool(train, recipes, pd.Timestamp("2000"), pd.Timestamp("2020"))
    pool.text = np.array([[0.6, 0.8], [1, 0], [0, 1]], dtype=np.float32)
    eval_df = pd.DataFrame({"user_id": [1, 2], "recipe_id": ["N1", "N1"], "date": pd.Timestamp(1)})
    cold = build_new_item_slice(train, train_df, eval_df, pool)

    result = evaluate_new_items(TextProfileRecommender().fit(train), cold, ks=(1,))
    assert result.metrics["hit_rate@1"] == 1.0

    with pytest.raises(NotImplementedError):
        evaluate_new_items(PopularityRecommender().fit(train), cold, ks=(1,))


def test_newest_scores_by_submission_date():
    train, _, recipes = _new_item_setup()
    train.with_content(recipes)
    pool = new_item_pool(train, recipes, pd.Timestamp("2000"), pd.Timestamp("2020"))
    model = NewestRecommender().fit(train)
    [ranked] = top_k(model.score_items(np.array([0]), pool), k=3)
    assert pool.item_ids[ranked].tolist() == ["N2", "N1", "X"]


def test_two_tower_scores_new_recipes_from_content():
    pytest.importorskip("torch")
    from recipe_recsys.data import NUTRITION_COLUMNS
    from recipe_recsys.dataset import ItemPool
    from recipe_recsys.models.two_tower import TwoTowerRecommender

    rows = []
    for u in range(40):
        first, second = ("A", "B") if u % 2 == 0 else ("C", "D")
        rows += [(u, first, 0), (u, second, 1)]
    rows.append((99, "A", 0))
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id", "day"])
    df["date"] = pd.Timestamp(0) + pd.to_timedelta(df.pop("day"), "D")
    data = InteractionData.from_frame(df)

    def content(ids, ingredients, tags):
        frame = pd.DataFrame({"recipe_id": ids, "ingredients": ingredients, "tags": tags})
        for col in [*NUTRITION_COLUMNS, "minutes", "n_steps", "n_ingredients"]:
            frame[col] = 1.0
        return frame

    data.with_content(
        content(
            list("ABCD"),
            [["egg"], ["flour"], ["rice"], ["beans"]],
            [["breakfast"], ["baking"], ["dinner"], ["dinner"]],
        )
    )
    data.with_text(np.eye(4, dtype=np.float32))
    # Two recipes never seen in training: one looks like B, the other like D.
    pool_content = content(["newB", "newD"], [["flour"], ["beans"]], [["baking"], ["dinner"]])
    pool = ItemPool(pool_content["recipe_id"].to_numpy(), pool_content, np.eye(4)[[1, 3]])

    model = TwoTowerRecommender(
        dim=8, epochs=60, batch_size=16, lr=0.01, min_token_count=1, text=1, id_dropout=0.5
    )
    model.fit(data, verbose=False)
    [scores] = model.score_items(np.array([data.user_index[99]]), pool)
    assert scores[0] > scores[1]


def test_paired_difference_detects_a_consistent_gain_hidden_by_user_variance():
    rng = np.random.default_rng(0)
    users = rng.random(500)  # users differ a lot from each other...
    base = EvalResult("base", {}, 0, 0.0, per_user={"ndcg@10": users})
    better = EvalResult("better", {}, 0, 0.0, per_user={"ndcg@10": users + 0.01})  # ...+0.01 each

    low_b, high_b = bootstrap_ci(users)
    low_m, _ = bootstrap_ci(users + 0.01)
    assert low_m < high_b  # marginal intervals overlap

    diff = paired_difference(better, base, "ndcg@10")
    assert diff.significant and diff.ci[0] > 0
    assert diff.mean == pytest.approx(0.01)
    assert not paired_difference(base, base, "ndcg@10").significant


def test_new_user_slice_uses_first_recipes_as_history():
    A, B, C, X = 10, 20, 30, 99
    train = InteractionData.from_frame(
        pd.DataFrame({"user_id": [1, 1, 2], "recipe_id": [A, B, C], "date": pd.Timestamp(0)})
    )
    eval_df = pd.DataFrame(
        {
            # user 9: new, cooks B, then A, then X (not in the catalog), then B again
            # user 8: new, only one catalog recipe, so nothing is left to predict
            # user 1: known, belongs to the warm slice
            "user_id": [9, 9, 9, 9, 8, 1],
            "recipe_id": [B, A, X, B, C, C],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01", "2020-01-01"]
            ),
        }
    )
    new = build_new_user_slice(train, eval_df, seed=1)
    assert new.truth == {0: {train.item_index[A]}}
    assert new.histories.toarray()[0].nonzero()[0].tolist() == [train.item_index[B]]


def test_scoring_a_training_history_matches_scoring_the_user():
    A, B, C, D = 10, 20, 30, 40
    rows = [(u, A) for u in range(5)] + [(u, B) for u in range(5)]
    rows += [(u, C) for u in range(5, 9)] + [(u, D) for u in range(5, 9)] + [(9, A)]
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id"]).assign(date=pd.Timestamp(0))
    data = InteractionData.from_frame(df)
    user = np.array([data.user_index[9]])
    item = data.item_index

    knn = ItemKNNRecommender(neighbors=10, shrink=0).fit(data)
    np.testing.assert_allclose(knn.score_histories(data.matrix[user]), knn.score(user))

    # iALS fold-in: a new user who cooked A gets B ranked above C and D.
    ials = IALSRecommender(factors=4, reg=0.01, alpha=10.0, iterations=20).fit(data)
    history = sp.csr_matrix(([1.0], ([0], [item[A]])), shape=(1, data.n_items))
    scores = ials.score_histories(history)[0]
    assert scores[item[B]] > max(scores[item[C]], scores[item[D]])


def test_evaluate_new_users_never_recommends_the_seed_recipe():
    A, B = 10, 20
    train = InteractionData.from_frame(
        pd.DataFrame({"user_id": [1, 1, 2, 2], "recipe_id": [A, B, A, B], "date": pd.Timestamp(0)})
    )
    eval_df = pd.DataFrame(
        {"user_id": [9, 9], "recipe_id": [A, B], "date": pd.to_datetime(["2020", "2021"])}
    )
    new = build_new_user_slice(train, eval_df, seed=1)
    # Popularity ties A and B; A is the seed, so B must be the only recommendation.
    result = evaluate_new_users(PopularityRecommender().fit(train), new, ks=(1,))
    assert result.metrics["hit_rate@1"] == 1.0
