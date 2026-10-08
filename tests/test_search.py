import numpy as np
import pandas as pd
import pytest

from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions
from recipe_recsys.search import (
    ParsedQuery,
    RecipeSearch,
    _to_query,
    detect_restrictions,
    parse_request,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("algo sem lactose, com frango", {"lactose"}),
        ("bolo vegano sem glúten", {"vegan", "gluten"}),
        ("I'm celiac, quick dinner?", {"gluten"}),
        ("cookies, my son has a peanut allergy", {"nuts"}),
        ("sem amendoim e sem ovo", {"nuts", "egg"}),
        # Asking FOR an ingredient is not a restriction.
        ("peanut butter cookies", set()),
        ("eggplant parmesan", set()),
        ("lasagna de carne", set()),
    ],
)
def test_detect_restrictions(text, expected):
    assert detect_restrictions(text) == expected


def test_to_query_drops_unknown_values_and_bad_minutes():
    parsed = _to_query(
        {
            "restrictions": ["lactose", "keto"],
            "include": [" Chicken ", ""],
            "exclude": [],
            "max_minutes": 0,
            "query": " quick chicken ",
        }
    )
    assert parsed == ParsedQuery(["lactose"], ["chicken"], [], None, "quick chicken")


class _FakeParser:
    def parse(self, text):
        return ParsedQuery(query="chicken")


def test_keyword_restrictions_are_added_when_the_llm_misses_them():
    parsed, keywords = parse_request("frango sem lactose", _FakeParser())
    assert parsed.restrictions == ["lactose"] and keywords == {"lactose"}


def _searcher():
    recipes = pd.DataFrame(
        {
            "recipe_id": [1, 2, 3, 4],
            "name": ["creamy chicken", "lemon chicken", "eggplant stew", "slow chicken"],
            "ingredients": [
                ["chicken", "heavy cream"],
                ["chicken breasts", "lemon"],
                ["eggplant", "tomato"],
                ["chicken", "onion"],
            ],
            "minutes": [15, 20, 25, 300],
        }
    )
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [1, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    masks = {
        name: allowed_mask(recipes["ingredients"].tolist(), get_restrictions([name]))
        for name in RESTRICTIONS
    }
    return RecipeSearch(recipes, vectors, lambda text: np.array([1, 0], np.float32), masks)


def test_search_applies_every_hard_filter_after_ranking():
    search = _searcher()
    parsed = ParsedQuery(
        restrictions=["lactose"], include=["chicken"], max_minutes=60, query="chicken"
    )
    # 1 has cream, 3 has no chicken, 4 takes 300 minutes: only 2 is left.
    assert search.search(parsed)["recipe_id"].tolist() == [2]


def test_include_and_exclude_match_whole_words():
    search = _searcher()
    # "egg" must not match "eggplant".
    assert search.allowed(ParsedQuery(include=["egg"])).sum() == 0
    assert search.allowed(ParsedQuery(exclude=["egg"])).all()
    # Plural: "chicken" matches "chicken breasts".
    assert search.allowed(ParsedQuery(include=["chicken"])).tolist() == [True, True, False, True]
