import urllib.error

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from recipe_recsys.api import ServingState, create_app  # noqa: E402
from recipe_recsys.dataset import InteractionData  # noqa: E402
from recipe_recsys.models import IALSRecommender  # noqa: E402
from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions  # noqa: E402
from recipe_recsys.search import ParsedQuery, RecipeSearch  # noqa: E402

RECIPES = pd.DataFrame(
    {
        "recipe_id": [10, 20, 30, 40],
        "name": ["creamy chicken", "lemon chicken", "bean stew", "chicken pie"],
        "ingredients": [
            ["chicken", "heavy cream"],
            ["chicken breasts", "lemon"],
            ["beans", "tomato"],
            ["chicken", "flour", "butter"],
        ],
        "minutes": [15, 20, 25, 60],
        "calories": [500.0, 300.0, 250.0, 700.0],
        "tags": [["main-dish"], ["main-dish"], ["soups-stews"], ["main-dish"]],
    }
)


class _Parser:
    model = "fake-llm"

    def __init__(self, fail: bool = False):
        self.fail = fail

    def parse(self, text: str) -> ParsedQuery:
        if self.fail:
            raise urllib.error.URLError("connection refused")
        return ParsedQuery(include=["chicken"], query="chicken")


def _state(parser=None) -> ServingState:
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.8, 0.2]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    masks = {
        name: allowed_mask(RECIPES["ingredients"].tolist(), get_restrictions([name]))
        for name in RESTRICTIONS
    }
    search = RecipeSearch(RECIPES, vectors, lambda text: np.array([1, 0], np.float32), masks)
    # Users who cook 10 also cook 20; users who cook 30 also cook 40.
    rows = [(u, 10) for u in range(5)] + [(u, 20) for u in range(5)]
    rows += [(u, 30) for u in range(5, 12)] + [(u, 40) for u in range(5, 12)]
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id"]).assign(date=pd.Timestamp(0))
    catalog = InteractionData.from_frame(df)
    model = IALSRecommender(factors=4, reg=0.01, alpha=10.0, iterations=20).fit(catalog)
    return ServingState(
        recipes=RECIPES.set_index("recipe_id"),
        search=search,
        parser=parser,
        catalog=catalog,
        model=model,
        popularity=np.asarray(catalog.matrix.sum(axis=0)).ravel(),
    )


def test_search_applies_parsed_constraints_and_keyword_restrictions():
    client = TestClient(create_app(_state(_Parser())))
    body = client.post("/api/search", json={"query": "frango sem lactose"}).json()
    assert body["llm_used"] and body["keyword_restrictions"] == ["lactose"]
    # 10 has cream, 40 has butter, 30 has no chicken.
    assert [r["recipe_id"] for r in body["results"]] == [20]


def test_search_falls_back_to_keywords_when_the_llm_is_down():
    client = TestClient(create_app(_state(_Parser(fail=True))))
    body = client.post("/api/search", json={"query": "sem lactose"}).json()
    assert not body["llm_used"] and "unavailable" in body["llm_error"]
    assert {r["recipe_id"] for r in body["results"]} == {20, 30}


def test_recommend_folds_in_a_new_user_and_respects_restrictions():
    client = TestClient(create_app(_state()))
    body = client.post("/api/recommend", json={"cooked": [10, 999]}).json()
    assert body["strategy"] == "personalized"
    assert body["used_history"] == [10] and body["ignored_history"] == [999]
    assert body["results"][0]["recipe_id"] == 20  # co-cooked with 10
    assert 10 not in [r["recipe_id"] for r in body["results"]]  # already cooked

    body = client.post("/api/recommend", json={"cooked": [30], "restrictions": ["gluten"]}).json()
    assert 40 not in [r["recipe_id"] for r in body["results"]]  # chicken pie has flour


def test_recommend_without_history_is_popularity():
    client = TestClient(create_app(_state()))
    body = client.post("/api/recommend", json={}).json()
    assert body["strategy"] == "popularity"
    assert body["results"][0]["recipe_id"] in {30, 40}  # cooked by 7 users vs 5

    response = client.post("/api/recommend", json={"restrictions": ["keto"]})
    assert response.status_code == 422


def test_recipe_lookup_and_detail():
    client = TestClient(create_app(_state()))
    names = [r["name"] for r in client.get("/api/recipes", params={"q": "chicken"}).json()]
    assert set(names) == {"creamy chicken", "lemon chicken", "chicken pie"}
    assert client.get("/api/recipes/30").json()["name"] == "bean stew"
    assert client.get("/api/recipes/999").status_code == 404


def test_menu_respects_limits_and_returns_a_shopping_list():
    client = TestClient(create_app(_state()))
    request = {"size": 3, "max_calories": 400, "main_dishes_only": False}
    body = client.post("/api/menu", json=request).json()
    # Only recipes 20 (300 kcal) and 30 (250 kcal) fit the calorie limit.
    assert {d["recipe_id"] for d in body["days"]} == {20, 30}
    # Shopping items are normalized: "chicken breasts" -> "chicken breast", "beans" -> "bean".
    assert {"chicken breast", "lemon", "bean", "tomato"} == set(body["shopping_list"])
    assert body["mean_calories"] == 275.0

    request = {"size": 4, "restrictions": ["vegetarian"], "main_dishes_only": False}
    body = client.post("/api/menu", json=request).json()
    assert [d["recipe_id"] for d in body["days"]] == [30]

    # By default only main dishes: the bean stew (30) is tagged as a soup.
    body = client.post("/api/menu", json={"size": 4}).json()
    assert 30 not in [d["recipe_id"] for d in body["days"]]
