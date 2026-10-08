"""Weekly menu planning as set recommendation.

A menu is not the top-n of a ranking. Three goals pull in different directions:

- relevance: what the recommender thinks the person wants;
- variety: seven near-identical chicken dishes make a bad week, so each pick is
  penalized by its text-embedding similarity to the recipes already chosen;
- ingredient reuse: a shorter shopping list, so each pick is rewarded for the
  share of its ingredients already in the basket (pantry staples excluded).

Per-meal limits (calories, minutes) and dietary restrictions are hard filters
on the candidates. Selection is greedy, in the spirit of maximal marginal
relevance (Carbonell & Goldstein, 1998): each step adds the candidate with the
best relevance + reuse bonus - redundancy penalty given the menu so far.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import EvalResult, evaluate_rankings, mask_scores, paired_difference
from recipe_recsys.models import Recommender

# Not worth a line on a shopping list.
PANTRY = frozenset(
    {
        "salt", "pepper", "black pepper", "salt and pepper", "salt & pepper", "water", "oil",
        "olive oil", "vegetable oil", "canola oil", "extra virgin olive oil", "sugar", "ice",
        "cooking spray", "kosher salt", "sea salt", "fresh ground black pepper",
        "ground black pepper",
    }
)  # fmt: skip


@dataclass(frozen=True)
class MenuConfig:
    size: int = 7
    variety: float = 0.5  # weight of the redundancy penalty
    reuse: float = 0.3  # weight of the shared-ingredient bonus
    candidates: int = 300  # only the most relevant candidates are considered


DEFAULT_MENU = MenuConfig()


@dataclass
class Menu:
    picks: list[int]  # positions in the candidate arrays, in pick order
    basket: list[int]  # ingredient columns needed (the shopping list)
    mean_similarity: float  # mean pairwise text similarity of the picks (lower = more varied)


def ingredient_matrix(ingredients: Sequence[Sequence[str]]) -> tuple[sp.csr_matrix, list[str]]:
    """Binary recipes x ingredients matrix without pantry staples, and the column names."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for row, recipe in enumerate(ingredients):
        for name in {i.strip().lower() for i in recipe} - PANTRY:
            rows.append(row)
            cols.append(vocab.setdefault(name, len(vocab)))
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(ingredients), len(vocab))
    )
    return matrix, list(vocab)


def plan_menu(
    relevance: np.ndarray,
    ingredients: sp.csr_matrix,
    text: np.ndarray,
    config: MenuConfig = DEFAULT_MENU,
    allowed: np.ndarray | None = None,
) -> Menu:
    """Pick `config.size` recipes.

    `relevance` (higher is better), `ingredients` (from `ingredient_matrix`) and
    `text` (unit-norm rows) are aligned; `allowed` is the hard-filter mask
    (restrictions, calorie and time limits).
    """
    scores = np.asarray(relevance, dtype=np.float64).copy()
    if allowed is not None:
        scores[~allowed] = -np.inf
    k = min(config.candidates, int(np.isfinite(scores).sum()))
    if k == 0:
        return Menu([], [], 0.0)
    pool = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    pool = pool[np.argsort(-scores[pool], kind="stable")]

    # Relevance on a 0..1 scale within the pool, so the weights mean the same for any model.
    rel = scores[pool]
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.ones_like(rel)
    items = ingredients[pool]
    sizes = np.maximum(np.asarray(items.sum(axis=1)).ravel(), 1)
    sims = text[pool] @ text[pool].T

    picks: list[int] = []
    basket = np.zeros(ingredients.shape[1], dtype=np.float32)
    max_sim = np.zeros(len(pool))
    available = np.ones(len(pool), dtype=bool)
    for _ in range(min(config.size, len(pool))):
        reuse = (items @ basket) / sizes  # share of each candidate's ingredients already bought
        gain = rel + config.reuse * reuse - config.variety * max_sim
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        picks.append(best)
        available[best] = False
        basket[items[best].indices] = 1.0
        max_sim = np.maximum(max_sim, sims[best])

    chosen = pool[picks]
    n = len(chosen)
    pair = text[chosen] @ text[chosen].T
    mean_similarity = float((pair.sum() - np.trace(pair)) / (n * (n - 1))) if n > 1 else 0.0
    return Menu([int(c) for c in chosen], np.flatnonzero(basket).tolist(), mean_similarity)


# ---- offline evaluation --------------------------------------------------------------------


@dataclass
class MenuEvaluation:
    config: MenuConfig
    result: EvalResult  # ranking metrics of the menu as an ordered list (NDCG@size, ...)
    shopping_items: float  # mean shopping list length
    mean_similarity: float  # mean pairwise similarity within a menu


def evaluate_menus(
    model: Recommender,
    train: InteractionData,
    truth: dict[int, set[int]],
    configs: Sequence[MenuConfig],
    ingredients: sp.csr_matrix,
    text: np.ndarray,
) -> list[MenuEvaluation]:
    """What planning costs: each config's menus scored as rankings on the warm slice.

    `ingredients` and `text` are aligned with the training catalog. The config with
    variety = reuse = 0 is the plain top-n of the model.
    """
    return [_evaluate_config(c, model, train, truth, ingredients, text) for c in configs]


def _evaluate_config(
    config: MenuConfig,
    model: Recommender,
    train: InteractionData,
    truth: dict[int, set[int]],
    ingredients: sp.csr_matrix,
    text: np.ndarray,
) -> MenuEvaluation:
    shopping, similarity = [], []

    def rank(batch: np.ndarray) -> list[np.ndarray]:
        scores = mask_scores(model.score(batch), batch, train, None)
        menus = [plan_menu(row, ingredients, text, config) for row in scores]
        shopping.extend(len(m.basket) for m in menus)
        similarity.extend(m.mean_similarity for m in menus)
        return [np.array(m.picks, dtype=np.int64) for m in menus]

    label = f"menu(variety={config.variety}, reuse={config.reuse})"
    result = evaluate_rankings(label, rank, truth, train.n_items, (config.size,), None, 256)
    return MenuEvaluation(config, result, float(np.mean(shopping)), float(np.mean(similarity)))


def menu_table(evaluations: list[MenuEvaluation]) -> str:
    base = evaluations[0].result
    size = evaluations[0].config.size
    metric = f"ndcg@{size}"
    lines = [
        f"| variety | reuse | {metric} | Δ vs top-{size} [95% CI] | hit rate@{size} "
        "| shopping list (items) | similarity within menu | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in evaluations:
        m = e.result.metrics
        if e.result is base:
            delta = "—"
        else:
            d = paired_difference(e.result, base, metric)
            delta = f"{d.relative:+.0%} [{d.ci[0]:+.4f}, {d.ci[1]:+.4f}]"
        lines.append(
            f"| {e.config.variety} | {e.config.reuse} | {m[metric]:.4f} | {delta} "
            f"| {m[f'hit_rate@{size}']:.4f} | {e.shopping_items:.1f} | {e.mean_similarity:.2f} "
            f"| {m[f'coverage@{size}']:.1%} |"
        )
    return "\n".join(lines)
