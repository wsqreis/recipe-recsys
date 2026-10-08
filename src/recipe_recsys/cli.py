"""Command line entry point: `uv run recsys {prepare,evaluate,recommend}`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_recsys import data
from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import (
    build_ground_truth,
    evaluate,
    recommend,
    results_table,
    to_dict,
)
from recipe_recsys.models import build_model
from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions
from recipe_recsys.split import k_core, temporal_split
from recipe_recsys.text import DEFAULT_MODEL as DEFAULT_TEXT_MODEL
from recipe_recsys.text import build_text_embeddings

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
    *data.NUTRITION_COLUMNS,
]


def _allowed_for(
    train: InteractionData, recipes: pd.DataFrame, names: list[str]
) -> np.ndarray | None:
    if not names:
        return None
    ingredients = recipes.set_index("recipe_id").loc[train.item_ids, "ingredients"]
    return allowed_mask(ingredients.tolist(), get_restrictions(names))


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
    allowed = _allowed_for(train, recipes, args.restrict)
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

    ks = tuple(args.k)
    results = []
    for spec in args.models:
        model = build_model(spec).fit(train)
        result = evaluate(model, train, truth, ks=ks, allowed=allowed)
        results.append(result)
        violations = f" | violations={result.restriction_violations}" if allowed is not None else ""
        print(
            f"  {result.model}: ndcg@{ks[0]}={result.metrics[f'ndcg@{ks[0]}']:.4f} "
            f"({result.seconds:.1f}s){violations}"
        )

    table = results_table(results, ks)
    print("\n" + table)

    if not args.save:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "stage": args.stage,
        "restrictions": args.restrict,
        "train": {"users": train.n_users, "items": train.n_items, "interactions": train.matrix.nnz},
        "cold_start": {**vars(stats), **shares},
        "results": [to_dict(r) for r in results],
    }
    (REPORTS_DIR / f"{args.save}.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS_DIR / f"{args.save}.md").write_text(table + "\n")
    print(f"\nsaved reports/{args.save}.md")


def cmd_recommend(args: argparse.Namespace) -> None:
    interactions = data.load_interactions()
    recipes = data.load_recipes().set_index("recipe_id")
    train = InteractionData.from_frame(k_core(interactions, args.min_user, args.min_item))
    train.with_content(recipes.reset_index()[CONTENT_COLUMNS])

    if args.user not in train.user_index:
        raise SystemExit(f"user {args.user} not found (needs >= {args.min_user} interactions)")
    row = train.user_index[args.user]

    history = train.interactions[train.interactions["user_idx"] == row].sort_values("date")
    print(f"user {args.user} cooked {len(history)} recipes; latest:")
    for item in history["item_idx"].tail(5):
        print(f"  - {recipes.at[train.item_ids[item], 'name']}")

    restrict_df = recipes.reset_index()
    allowed = _allowed_for(train, restrict_df, args.restrict)
    model = build_model(args.model).fit(train)
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
