"""Offline evaluation protocol.

For each user that exists in training and has at least one evaluation
interaction with a known recipe:
  1. the model scores every catalog item;
  2. items the user already cooked in training are removed (we recommend new recipes);
  3. items forbidden by the user's dietary restrictions are removed (hard filter);
  4. the top-k remaining items are compared with what the user actually cooked next.

Interactions with users or recipes that never appear in training (cold start)
cannot be served by collaborative models and are reported separately instead
of being silently dropped.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recipe_recsys import metrics
from recipe_recsys.dataset import InteractionData, ItemPool
from recipe_recsys.models import Recommender


@dataclass
class ColdStartStats:
    eval_interactions: int
    cold_user_interactions: int
    cold_item_interactions: int
    evaluable_interactions: int
    evaluable_users: int

    def as_shares(self) -> dict[str, float]:
        total = max(self.eval_interactions, 1)
        return {
            "cold_user_share": self.cold_user_interactions / total,
            "cold_item_share": self.cold_item_interactions / total,
            "evaluable_share": self.evaluable_interactions / total,
        }


@dataclass
class EvalResult:
    model: str
    metrics: dict[str, float]
    restriction_violations: int
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)


def top_k(scores: np.ndarray, k: int) -> list[np.ndarray]:
    """Top-k item indices per row, best first, never returning masked (-inf) items."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    ranked = np.take_along_axis(part, order, axis=1)
    ranked_scores = np.take_along_axis(part_scores, order, axis=1)
    return [row[np.isfinite(s)] for row, s in zip(ranked, ranked_scores, strict=True)]


def mask_scores(
    scores: np.ndarray,
    user_rows: np.ndarray,
    train: InteractionData,
    allowed: np.ndarray | None,
) -> np.ndarray:
    seen = train.matrix[user_rows]
    scores[seen.nonzero()] = -np.inf
    if allowed is not None:
        scores[:, ~allowed] = -np.inf
    return scores


def recommend(
    model: Recommender,
    train: InteractionData,
    user_rows: np.ndarray,
    k: int,
    allowed: np.ndarray | None = None,
) -> list[np.ndarray]:
    scores = model.score(user_rows)
    return top_k(mask_scores(scores, user_rows, train, allowed), k)


def build_ground_truth(
    train: InteractionData, eval_df: pd.DataFrame, allowed: np.ndarray | None = None
) -> tuple[dict[int, set[int]], ColdStartStats]:
    user_rows = eval_df["user_id"].map(train.user_index)
    item_cols = eval_df["recipe_id"].map(train.item_index)
    known = user_rows.notna() & item_cols.notna()

    pairs = pd.DataFrame(
        {"u": user_rows[known].astype(int), "i": item_cols[known].astype(int)}
    ).drop_duplicates()
    # Re-cooking a recipe from training is not a "new" recommendation.
    already_seen = np.asarray(train.matrix[pairs["u"].to_numpy(), pairs["i"].to_numpy()]).ravel()
    pairs = pairs[already_seen == 0]
    if allowed is not None:
        # A user with a restriction can only be expected to cook allowed recipes.
        pairs = pairs[allowed[pairs["i"].to_numpy()]]

    truth = {int(u): set(g.tolist()) for u, g in pairs.groupby("u")["i"]}
    stats = ColdStartStats(
        eval_interactions=len(eval_df),
        cold_user_interactions=int(user_rows.isna().sum()),
        cold_item_interactions=int((user_rows.notna() & item_cols.isna()).sum()),
        evaluable_interactions=len(pairs),
        evaluable_users=len(truth),
    )
    return truth, stats


def evaluate(
    model: Recommender,
    train: InteractionData,
    truth: dict[int, set[int]],
    ks: tuple[int, ...] = (10, 20),
    allowed: np.ndarray | None = None,
    batch_size: int = 512,
) -> EvalResult:
    return _evaluate_rankings(
        repr(model),
        lambda batch: recommend(model, train, batch, max(ks), allowed),
        truth,
        train.n_items,
        ks,
        allowed,
        batch_size,
    )


def _evaluate_rankings(
    model_name: str,
    rank: Callable[[np.ndarray], list[np.ndarray]],
    truth: dict[int, set[int]],
    n_items: int,
    ks: tuple[int, ...],
    allowed: np.ndarray | None,
    batch_size: int,
) -> EvalResult:
    start = time.perf_counter()
    users = np.array(sorted(truth), dtype=np.int64)

    sums: dict[str, float] = {}
    all_recs: list[np.ndarray] = []
    violations = 0
    for batch_start in range(0, len(users), batch_size):
        batch = users[batch_start : batch_start + batch_size]
        recs = rank(batch)
        for user, ranked in zip(batch, recs, strict=True):
            relevant = truth[int(user)]
            ranked_list = ranked.tolist()
            for k in ks:
                for metric_name, fn in (
                    ("recall", metrics.recall_at_k),
                    ("ndcg", metrics.ndcg_at_k),
                    ("hit_rate", metrics.hit_rate_at_k),
                ):
                    key = f"{metric_name}@{k}"
                    sums[key] = sums.get(key, 0.0) + fn(ranked_list, relevant, k)
            if allowed is not None:
                violations += int((~allowed[ranked]).sum())
        all_recs.extend(recs)

    n = max(len(users), 1)
    results = {key: value / n for key, value in sums.items()}
    for k in ks:
        results[f"coverage@{k}"] = metrics.catalog_coverage([r[:k] for r in all_recs], n_items)
    return EvalResult(
        model=model_name,
        metrics=results,
        restriction_violations=violations,
        seconds=time.perf_counter() - start,
    )


# ---- cold items --------------------------------------------------------------------------
#
# The "cold recipes" share printed by build_ground_truth is not dropped forever:
# this slice asks known users to rank only recipes outside the training catalog.
# They are either new (submitted after training ended) or long-tail recipes that
# the k-core removed for having fewer than 5 interactions; for a collaborative
# model both are equally unknown.


@dataclass
class ColdItemSlice:
    pool: ItemPool
    truth: dict[int, set[int]]  # train user row -> pool indices
    seen: sp.csr_matrix  # train users x pool: cooked before the k-core removed the recipe


def cold_item_pool(
    train: InteractionData, recipes: pd.DataFrame, available_before: pd.Timestamp | None = None
) -> ItemPool:
    """Recipes outside the training catalog that existed before the end of the evaluated window.

    Restricting candidates to recipes that appear in the evaluation window would
    leak the future (it tells the model which recipes will be cooked).
    """
    keep = ~recipes["recipe_id"].isin(train.item_ids)
    if available_before is not None:
        keep &= recipes["submitted"] < available_before
    content = recipes[keep].reset_index(drop=True)
    return ItemPool(content["recipe_id"].to_numpy(), content)


def build_cold_item_slice(
    train: InteractionData,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pool: ItemPool,
    allowed: np.ndarray | None = None,
) -> ColdItemSlice:
    """`train_df` is the raw training window (before the k-core): what users already cooked."""
    pool_index = pd.Series(np.arange(len(pool)), index=pool.item_ids)

    def pairs(df: pd.DataFrame) -> pd.DataFrame:
        u = df["user_id"].map(train.user_index)
        i = df["recipe_id"].map(pool_index)
        known = u.notna() & i.notna()
        return pd.DataFrame(
            {"u": u[known].astype(int), "i": i[known].astype(int)}
        ).drop_duplicates()

    seen_pairs = pairs(train_df)
    seen = sp.csr_matrix(
        (
            np.ones(len(seen_pairs), dtype=np.float32),
            (seen_pairs["u"].to_numpy(), seen_pairs["i"].to_numpy()),
        ),
        shape=(train.n_users, len(pool)),
    )
    eval_pairs = pairs(eval_df)
    already_seen = seen[eval_pairs["u"].to_numpy(), eval_pairs["i"].to_numpy()]
    eval_pairs = eval_pairs[np.asarray(already_seen).ravel() == 0]
    if allowed is not None:
        eval_pairs = eval_pairs[allowed[eval_pairs["i"].to_numpy()]]
    truth = {int(u): set(g.tolist()) for u, g in eval_pairs.groupby("u")["i"]}
    return ColdItemSlice(pool, truth, seen)


def evaluate_cold_items(
    model: Recommender,
    cold: ColdItemSlice,
    ks: tuple[int, ...] = (10, 20),
    allowed: np.ndarray | None = None,
    batch_size: int = 256,
) -> EvalResult:
    """Rank only the pool; raises NotImplementedError for models that cannot score it."""

    def rank(batch: np.ndarray) -> list[np.ndarray]:
        scores = np.asarray(model.score_items(batch, cold.pool), dtype=np.float32)
        scores[cold.seen[batch].nonzero()] = -np.inf
        if allowed is not None:
            scores[:, ~allowed] = -np.inf
        return top_k(scores, max(ks))

    return _evaluate_rankings(
        repr(model), rank, cold.truth, len(cold.pool), ks, allowed, batch_size
    )


def results_table(results: list[EvalResult], ks: tuple[int, ...]) -> str:
    cols = [f"{m}@{k}" for k in ks for m in ("recall", "ndcg", "hit_rate", "coverage")]
    header = "| model | " + " | ".join(cols) + " | time (s) |"
    sep = "|" + "---|" * (len(cols) + 2)
    rows = [
        f"| {r.model} | "
        + " | ".join(f"{r.metrics[c]:.4f}" for c in cols)
        + f" | {r.seconds:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def to_dict(result: EvalResult) -> dict:
    return asdict(result)
