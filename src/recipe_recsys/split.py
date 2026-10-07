"""Temporal train/validation/test split.

Random splits leak the future into training: the model learns from reviews
written after the ones it is asked to predict. Here every interaction in
validation/test happens strictly after every interaction in train.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split(
    interactions: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1
) -> TemporalSplit:
    dates = interactions["date"]
    val_start = dates.quantile(1 - val_frac - test_frac)
    test_start = dates.quantile(1 - test_frac)

    return TemporalSplit(
        train=interactions[dates < val_start],
        val=interactions[(dates >= val_start) & (dates < test_start)],
        test=interactions[dates >= test_start],
        val_start=val_start,
        test_start=test_start,
    )


def k_core(interactions: pd.DataFrame, min_user: int = 5, min_item: int = 5) -> pd.DataFrame:
    """Iteratively drop users and items with too few interactions.

    Apply only to training data: filtering with counts from the future would
    leak information about which users/items stay active.
    """
    df = interactions
    while True:
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        item_counts = df["recipe_id"].map(df["recipe_id"].value_counts())
        keep = (user_counts >= min_user) & (item_counts >= min_item)
        if keep.all():
            return df
        df = df[keep]
