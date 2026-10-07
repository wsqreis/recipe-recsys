"""Download, parse and cache the Food.com dataset.

Raw source: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
DATA_DIR = Path("data/processed")

# Order of the values in the raw `nutrition` column. Except for calories,
# values are percentage of daily value (PDV).
NUTRITION_COLUMNS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbs_pdv",
]


def download_raw() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def _parse_recipes(raw: pd.DataFrame) -> pd.DataFrame:
    recipes = raw.rename(columns={"id": "recipe_id"})
    for col in ["tags", "ingredients", "steps"]:
        recipes[col] = recipes[col].map(ast.literal_eval)

    nutrition = pd.DataFrame(
        recipes["nutrition"].map(ast.literal_eval).tolist(),
        columns=NUTRITION_COLUMNS,
        index=recipes.index,
    )
    recipes = pd.concat([recipes.drop(columns="nutrition"), nutrition], axis=1)
    recipes["submitted"] = pd.to_datetime(recipes["submitted"])
    recipes["name"] = recipes["name"].fillna("").str.strip()
    return recipes


def _parse_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    # Every review is an implicit "cooked it" signal. rating == 0 means the
    # user left a review without a rating, not a bad rating.
    interactions = raw[["user_id", "recipe_id", "date", "rating"]].copy()
    interactions["date"] = pd.to_datetime(interactions["date"])
    return interactions.sort_values("date", kind="stable").reset_index(drop=True)


def prepare(data_dir: Path = DATA_DIR) -> None:
    raw_dir = download_raw()
    data_dir.mkdir(parents=True, exist_ok=True)

    recipes = _parse_recipes(pd.read_csv(raw_dir / "RAW_recipes.csv"))
    recipes.to_parquet(data_dir / "recipes.parquet", index=False)

    interactions = _parse_interactions(pd.read_csv(raw_dir / "RAW_interactions.csv"))
    interactions.to_parquet(data_dir / "interactions.parquet", index=False)

    print(f"recipes: {len(recipes):,} | interactions: {len(interactions):,} -> {data_dir}")


def load_recipes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "recipes.parquet")


def load_interactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "interactions.parquet")
