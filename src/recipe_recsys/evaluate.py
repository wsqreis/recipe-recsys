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
    # 95% bootstrap interval per metric (over users), see bootstrap_ci.
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Metric value per evaluated user, in sorted user order (for paired comparisons).
    per_user: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


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

    per_user: dict[str, list[float]] = {}
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
                    per_user.setdefault(f"{metric_name}@{k}", []).append(
                        fn(ranked_list, relevant, k)
                    )
            if allowed is not None:
                violations += int((~allowed[ranked]).sum())
        all_recs.extend(recs)

    n = max(len(users), 1)
    values = {key: np.asarray(v) for key, v in per_user.items()}
    results = {key: float(v.sum() / n) for key, v in values.items()}
    for k in ks:
        results[f"coverage@{k}"] = metrics.catalog_coverage([r[:k] for r in all_recs], n_items)
    return EvalResult(
        model=model_name,
        metrics=results,
        restriction_violations=violations,
        seconds=time.perf_counter() - start,
        ci={key: bootstrap_ci(v) for key, v in values.items()},
        per_user=values,
    )


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean over users.

    The fixed seed draws the same user resamples for every model evaluated on
    the same users, so overlapping intervals are a fair "too close to call".
    """
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


# ---- new items ---------------------------------------------------------------------------
#
# The "cold recipes" share printed by build_ground_truth is not dropped forever:
# this slice asks known users to rank only recipes submitted during the evaluated
# window, which no model has seen in training. Long-tail recipes removed by the
# k-core are left out on purpose: with them the pool grows from ~9k to ~194k
# recipes and every model, random included, lands within noise of the others.


@dataclass
class NewItemSlice:
    pool: ItemPool
    truth: dict[int, set[int]]  # train user row -> pool indices
    seen: sp.csr_matrix  # train users x pool: already cooked in the raw training window


def new_item_pool(
    train: InteractionData,
    recipes: pd.DataFrame,
    submitted_from: pd.Timestamp,
    submitted_before: pd.Timestamp | None = None,
) -> ItemPool:
    """Recipes submitted during the evaluated window (and absent from the training catalog).

    Restricting candidates to recipes that are cooked in the window would leak the
    future (it tells the model which recipes will be cooked), so every recipe
    submitted in the window is a candidate.
    """
    keep = ~recipes["recipe_id"].isin(train.item_ids) & (recipes["submitted"] >= submitted_from)
    if submitted_before is not None:
        keep &= recipes["submitted"] < submitted_before
    content = recipes[keep].reset_index(drop=True)
    return ItemPool(content["recipe_id"].to_numpy(), content)


def build_new_item_slice(
    train: InteractionData,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pool: ItemPool,
    allowed: np.ndarray | None = None,
) -> NewItemSlice:
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
    return NewItemSlice(pool, truth, seen)


def evaluate_new_items(
    model: Recommender,
    new: NewItemSlice,
    ks: tuple[int, ...] = (10, 20),
    allowed: np.ndarray | None = None,
    batch_size: int = 256,
) -> EvalResult:
    """Rank only the pool; raises NotImplementedError for models that cannot score it."""

    def rank(batch: np.ndarray) -> list[np.ndarray]:
        scores = np.asarray(model.score_items(batch, new.pool), dtype=np.float32)
        scores[new.seen[batch].nonzero()] = -np.inf
        if allowed is not None:
            scores[:, ~allowed] = -np.inf
        return top_k(scores, max(ks))

    return _evaluate_rankings(repr(model), rank, new.truth, len(new.pool), ks, allowed, batch_size)


def results_table(results: list[EvalResult], ks: tuple[int, ...]) -> str:
    cols = [f"{m}@{k}" for k in ks for m in ("recall", "ndcg", "hit_rate", "coverage")]
    main = f"ndcg@{ks[0]}"
    header = "| model | " + " | ".join(cols) + f" | {main} 95% CI | time (s) |"
    sep = "|" + "---|" * (len(cols) + 3)
    rows = [
        f"| {r.model} | "
        + " | ".join(f"{r.metrics[c]:.4f}" for c in cols)
        + f" | {format_ci(r.ci.get(main))} | {r.seconds:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def format_ci(ci: tuple[float, float] | None) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "-"


@dataclass
class PairedDifference:
    model: str
    baseline: str
    metric: str
    mean: float  # mean over users of (model - baseline)
    ci: tuple[float, float]
    relative: float  # mean / baseline mean

    @property
    def significant(self) -> bool:
        return self.ci[0] > 0 or self.ci[1] < 0


def paired_difference(result: EvalResult, baseline: EvalResult, metric: str) -> PairedDifference:
    """Difference to `baseline` on the same users, with a 95% bootstrap interval.

    Each user is compared with themselves, which removes the user-to-user
    variance that makes the marginal intervals of two models overlap even when
    one model is consistently better.
    """
    a, b = result.per_user[metric], baseline.per_user[metric]
    if a.shape != b.shape:
        raise ValueError("paired comparison needs both models evaluated on the same users")
    diff = a - b
    base = b.mean() if len(b) else 0.0
    return PairedDifference(
        model=result.model,
        baseline=baseline.model,
        metric=metric,
        mean=float(diff.mean()) if len(diff) else 0.0,
        ci=bootstrap_ci(diff),
        relative=float(diff.mean() / base) if base else float("nan"),
    )


def paired_table(diffs: list[PairedDifference]) -> str:
    if not diffs:
        return ""
    lines = [
        f"| model | Δ{diffs[0].metric} vs {diffs[0].baseline} | 95% CI | relative | significant |",
        "|---|---|---|---|---|",
    ]
    for d in diffs:
        lines.append(
            f"| {d.model} | {d.mean:+.4f} | [{d.ci[0]:+.4f}, {d.ci[1]:+.4f}] "
            f"| {d.relative:+.0%} | {'yes' if d.significant else 'no'} |"
        )
    return "\n".join(lines)


def to_dict(result: EvalResult) -> dict:
    out = asdict(result)
    del out["per_user"]
    return out
