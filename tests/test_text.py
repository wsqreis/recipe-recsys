import numpy as np
import pandas as pd
import pytest

from recipe_recsys.text import cache_path, load_text_embeddings, recipe_text


def test_recipe_text_puts_name_and_ingredients_before_description():
    recipes = pd.DataFrame(
        {
            "name": ["pancakes ", "plain rice"],
            "ingredients": [["flour", "egg"], ["rice"]],
            "description": ["fluffy  and\nquick", None],
        }
    )
    assert recipe_text(recipes) == [
        "pancakes Ingredients: flour, egg. fluffy and quick",
        "plain rice Ingredients: rice.",
    ]


def test_load_text_embeddings_aligns_with_requested_ids(tmp_path):
    vectors = np.eye(3, dtype=np.float16)
    np.savez(cache_path("m", tmp_path), recipe_id=np.array([10, 20, 30]), vectors=vectors)

    out = load_text_embeddings(np.array([30, 10]), model="m", data_dir=tmp_path)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [[0, 0, 1], [1, 0, 0]])

    with pytest.raises(ValueError, match="no text embedding"):
        load_text_embeddings(np.array([99]), model="m", data_dir=tmp_path)
