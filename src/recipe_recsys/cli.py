"""Command line entry point: `uv run recsys {prepare,evaluate,recommend}`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_recsys import data
from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import (
    EvalResult,
    PairedDifference,
    build_ground_truth,
    build_new_item_slice,
    evaluate,
    evaluate_new_items,
    format_ci,
    new_item_pool,
    paired_difference,
    paired_table,
    recommend,
    results_table,
    to_dict,
)
from recipe_recsys.models import Recommender, build_model
from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions
from recipe_recsys.split import k_core, temporal_split
from recipe_recsys.text import DEFAULT_MODEL as DEFAULT_TEXT_MODEL
from recipe_recsys.text import build_text_embeddings, load_text_embeddings

DEFAULT_MODELS = [
    "random",
    "popularity",
    "recent_popularity",
    "itemknn",
]
REPORTS_DIR = Path("reports")
CONTENT_COLUMNS = [
    "recipe_id",
    "ingredients",
    "tags",
    "minutes",
    "n_steps",
    "n_ingredients",
    "submitted",
    *data.NUTRITION_COLUMNS,
]


def _allowed_for(
    item_ids: np.ndarray, recipes: pd.DataFrame, names: list[str]
) -> np.ndarray | None:
    if not names:
        return None
    ingredients = recipes.set_index("recipe_id").loc[item_ids, "ingredients"]
    return allowed_mask(ingredients.tolist(), get_restrictions(names))


def _text_lookup(
    recipes: pd.DataFrame, models: list[Recommender]
) -> Callable[[np.ndarray], np.ndarray] | None:
    """recipe ids -> text embeddings, loaded only if one of the models needs them."""
    if not any(model.needs_text for model in models):
        return None
    ids = recipes["recipe_id"].to_numpy()
    vectors = load_text_embeddings(ids)
    rows = pd.Index(ids)
    return lambda item_ids: vectors[rows.get_indexer(item_ids)]


def cmd_prepare(_: argparse.Namespace) -> None:
    data.prepare()


def cmd_embed(args: argparse.Namespace) -> None:
    path = build_text_embeddings(args.model, batch_size=args.batch_size)
    print(f"saved {path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    interactions = data.load_interactions()
    split = temporal_split(interactions)
    if args.stage == "val":
        train_df, eval_df = split.train, split.val
    else:
        train_df, eval_df = pd.concat([split.train, split.val]), split.test

    recipes = data.load_recipes(columns=CONTENT_COLUMNS)
    train = InteractionData.from_frame(k_core(train_df, args.min_user, args.min_item))
    train.with_content(recipes)
    models = [build_model(spec) for spec in args.models]
    text = _text_lookup(recipes, models)
    if text is not None:
        train.with_text(text(train.item_ids))
    allowed = _allowed_for(train.item_ids, recipes, args.restrict)
    truth, stats = build_ground_truth(train, eval_df, allowed)

    print(
        f"stage={args.stage} | val_start={split.val_start:%Y-%m-%d} "
        f"test_start={split.test_start:%Y-%m-%d}"
    )
    density = train.matrix.nnz / np.prod(train.matrix.shape)
    print(
        f"train: {train.n_users:,} users x {train.n_items:,} recipes, "
        f"{train.matrix.nnz:,} interactions (density {density:.5%})"
    )
    if allowed is not None:
        print(f"restrictions {args.restrict}: {allowed.mean():.1%} of the catalog is allowed")
    shares = stats.as_shares()
    print(
        f"eval interactions: {stats.eval_interactions:,} "
        f"| cold users {shares['cold_user_share']:.1%} "
        f"| cold recipes {shares['cold_item_share']:.1%} "
        f"| evaluable {shares['evaluable_share']:.1%} ({stats.evaluable_users:,} users)\n"
    )

    new, new_allowed, new_info = None, None, {}
    if "new_item" in args.slice:
        window_start = split.val_start if args.stage == "val" else split.test_start
        window_end = split.test_start if args.stage == "val" else None
        pool = new_item_pool(train, recipes, window_start, window_end)
        if text is not None:
            pool.text = text(pool.item_ids)
        new_allowed = _allowed_for(pool.item_ids, recipes, args.restrict)
        new = build_new_item_slice(train, train_df, eval_df, pool, new_allowed)
        new_info = {
            "candidates": len(pool),
            "interactions": sum(len(items) for items in new.truth.values()),
            "users": len(new.truth),
        }
        print(
            f"new_item slice: {new_info['candidates']:,} recipes submitted during the window "
            f"| {new_info['interactions']:,} interactions from {new_info['users']:,} known users\n"
        )

    ks = tuple(args.k)
    results, new_results = [], []
    for model in models:
        model.fit(train)
        if "warm" in args.slice:
            results.append(evaluate(model, train, truth, ks=ks, allowed=allowed))
            _print_result(results[-1], ks, allowed)
        if new is not None:
            try:
                new_results.append(evaluate_new_items(model, new, ks=ks, allowed=new_allowed))
            except NotImplementedError as e:
                print(f"  {model} [new_item]: n/a ({e})")
                continue
            _print_result(new_results[-1], ks, new_allowed, "new_item")

    baselines = [repr(build_model(spec)) for spec in args.baseline]
    paired = {
        "warm": _paired(results, baselines, f"ndcg@{ks[0]}"),
        "new_item": _paired(new_results, baselines, f"ndcg@{ks[0]}"),
    }
    sections = []
    if results:
        sections.append(results_table(results, ks))
        if paired["warm"]:
            sections.append(paired_table(paired["warm"]))
    if new_results:
        sections.append("## new_item\n\n" + results_table(new_results, ks))
        if paired["new_item"]:
            sections.append(paired_table(paired["new_item"]))
    print("\n" + "\n\n".join(sections))

    if not args.save:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "stage": args.stage,
        "restrictions": args.restrict,
        "train": {"users": train.n_users, "items": train.n_items, "interactions": train.matrix.nnz},
        "cold_start": {**vars(stats), **shares},
        "results": [to_dict(r) for r in results],
        "paired": [asdict(d) for d in paired["warm"]],
    }
    if new is not None:
        report["new_item"] = {
            **new_info,
            "results": [to_dict(r) for r in new_results],
            "paired": [asdict(d) for d in paired["new_item"]],
        }
    (REPORTS_DIR / f"{args.save}.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS_DIR / f"{args.save}.md").write_text("\n\n".join(sections) + "\n")
    print(f"\nsaved reports/{args.save}.md")


def _paired(results: list[EvalResult], baselines: list[str], metric: str) -> list[PairedDifference]:
    """Compare every model with the first of `baselines` that was evaluated on this slice."""
    by_name = {r.model: r for r in results}
    base = next((by_name[name] for name in baselines if name in by_name), None)
    if base is None:
        return []
    return [paired_difference(r, base, metric) for r in results if r is not base]


def _print_result(
    result: EvalResult, ks: tuple[int, ...], allowed: np.ndarray | None, label: str = ""
) -> None:
    violations = f" | violations={result.restriction_violations}" if allowed is not None else ""
    tag = f" [{label}]" if label else ""
    main = f"ndcg@{ks[0]}"
    print(
        f"  {result.model}{tag}: {main}={result.metrics[main]:.4f} "
        f"{format_ci(result.ci.get(main))} ({result.seconds:.1f}s){violations}"
    )


def cmd_recommend(args: argparse.Namespace) -> None:
    interactions = data.load_interactions()
    recipes = data.load_recipes().set_index("recipe_id")
    train = InteractionData.from_frame(k_core(interactions, args.min_user, args.min_item))
    train.with_content(recipes.reset_index()[CONTENT_COLUMNS])
    model = build_model(args.model)
    text = _text_lookup(recipes.reset_index(), [model])
    if text is not None:
        train.with_text(text(train.item_ids))

    if args.user not in train.user_index:
        raise SystemExit(f"user {args.user} not found (needs >= {args.min_user} interactions)")
    row = train.user_index[args.user]

    history = train.interactions[train.interactions["user_idx"] == row].sort_values("date")
    print(f"user {args.user} cooked {len(history)} recipes; latest:")
    for item in history["item_idx"].tail(5):
        print(f"  - {recipes.at[train.item_ids[item], 'name']}")

    restrict_df = recipes.reset_index()
    allowed = _allowed_for(train.item_ids, restrict_df, args.restrict)
    model.fit(train)
    [recs] = recommend(model, train, np.array([row]), args.n, allowed)

    label = f" (restrictions: {', '.join(args.restrict)})" if args.restrict else ""
    print(f"\n{model} recommends{label}:")
    for rank, item in enumerate(recs, 1):
        recipe = recipes.loc[train.item_ids[item]]
        print(
            f"  {rank:2}. {recipe['name']}  [{recipe['minutes']} min, "
            f"{recipe['calories']:.0f} kcal]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="recsys", description=__doc__)
    sub = parser.add_subparsers(required=True)

    sub.add_parser("prepare", help="download Food.com and cache it as parquet").set_defaults(
        func=cmd_prepare
    )

    emb = sub.add_parser("embed", help="encode recipe text with a sentence encoder (cached)")
    emb.add_argument("--model", default=DEFAULT_TEXT_MODEL)
    emb.add_argument("--batch-size", type=int, default=256)
    emb.set_defaults(func=cmd_embed)

    restrict_kwargs = {
        "nargs": "*",
        "default": [],
        "choices": sorted(RESTRICTIONS),
        "help": "hard dietary restrictions",
    }

    ev = sub.add_parser("evaluate", help="offline evaluation on a temporal split")
    ev.add_argument(
        "--stage",
        choices=["val", "test"],
        default="val",
        help="val: train -> val (for tuning). test: train+val -> test (final numbers)",
    )
    ev.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="model specs, e.g. itemknn:neighbors=50,shrink=5.0",
    )
    ev.add_argument(
        "--slice",
        nargs="+",
        choices=["warm", "new_item"],
        default=["warm"],
        help="warm: known users x training catalog. new_item: known users x recipes "
        "submitted during the evaluated window (only content-aware models can score them)",
    )
    ev.add_argument(
        "--baseline",
        nargs="+",
        default=[],
        metavar="SPEC",
        help="report each model's paired difference to the first of these model specs "
        "evaluated on each slice, e.g. `--baseline popularity random`",
    )
    ev.add_argument("--k", nargs="+", type=int, default=[10, 20])
    ev.add_argument("--restrict", **restrict_kwargs)
    ev.add_argument("--save", metavar="NAME", help="write reports/NAME.md and reports/NAME.json")
    ev.add_argument("--min-user", type=int, default=5)
    ev.add_argument("--min-item", type=int, default=5)
    ev.set_defaults(func=cmd_evaluate)

    rec = sub.add_parser("recommend", help="recommend recipes for one user")
    rec.add_argument("--user", type=int, required=True)
    rec.add_argument("--model", default="itemknn")
    rec.add_argument("--n", type=int, default=10)
    rec.add_argument("--restrict", **restrict_kwargs)
    rec.add_argument("--min-user", type=int, default=5)
    rec.add_argument("--min-item", type=int, default=5)
    rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
