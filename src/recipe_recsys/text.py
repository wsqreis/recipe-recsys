"""Sentence embeddings of recipe text (name, ingredients, description).

These vectors depend only on the recipe itself, never on who cooked it, so
they exist for recipes with no interactions at all: the input that cold-start
models need. They are computed once with a pretrained sentence encoder and
cached next to the processed data.

The text puts the name and the ingredients first and the free-form description
last: the encoder truncates long inputs, and the description is the noisiest
and least structured part.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_recsys.data import DATA_DIR, load_recipes

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def recipe_text(recipes: pd.DataFrame) -> list[str]:
    texts = []
    for name, ingredients, description in zip(
        recipes["name"], recipes["ingredients"], recipes["description"], strict=True
    ):
        parts = [str(name).strip(), "Ingredients: " + ", ".join(ingredients) + "."]
        if isinstance(description, str) and description.strip():
            parts.append(" ".join(description.split()))
        texts.append(" ".join(parts))
    return texts


def cache_path(model: str = DEFAULT_MODEL, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"text_{model.rsplit('/', 1)[-1]}.npz"


def build_text_embeddings(
    model: str = DEFAULT_MODEL,
    data_dir: Path = DATA_DIR,
    batch_size: int = 256,
    device: str | None = None,
) -> Path:
    from sentence_transformers import SentenceTransformer

    recipes = load_recipes(data_dir, columns=["recipe_id", "name", "ingredients", "description"])
    encoder = SentenceTransformer(model, device=device)
    start = time.perf_counter()
    vectors = encoder.encode(
        recipe_text(recipes),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(
        f"encoded {len(vectors):,} recipes -> {vectors.shape[1]} dims "
        f"in {time.perf_counter() - start:.0f}s on {encoder.device}"
    )
    path = cache_path(model, data_dir)
    # float16 halves the file (231k x 384) at no measurable cost for cosine similarity.
    np.savez(path, recipe_id=recipes["recipe_id"].to_numpy(), vectors=vectors.astype(np.float16))
    return path


def query_encoder(model: str = DEFAULT_MODEL, device: str | None = None):
    """str -> unit-norm float32 vector, in the same space as the cached recipe embeddings."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model, device=device)
    return lambda text: encoder.encode(text, normalize_embeddings=True).astype(np.float32)


def load_text_embeddings(
    recipe_ids: np.ndarray, model: str = DEFAULT_MODEL, data_dir: Path = DATA_DIR
) -> np.ndarray:
    """Unit-norm float32 vectors aligned with `recipe_ids`."""
    path = cache_path(model, data_dir)
    if not path.exists():
        raise SystemExit(f"{path} not found: run `recsys embed` first")
    cached = np.load(path)
    index = pd.Index(cached["recipe_id"])
    rows = index.get_indexer(recipe_ids)
    if (rows < 0).any():
        raise ValueError(f"{(rows < 0).sum()} recipes have no text embedding")
    return cached["vectors"][rows].astype(np.float32)
