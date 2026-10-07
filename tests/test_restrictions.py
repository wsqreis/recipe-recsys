import numpy as np
import pandas as pd
import pytest

from recipe_recsys.dataset import InteractionData
from recipe_recsys.evaluate import recommend
from recipe_recsys.models import PopularityRecommender
from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions, is_allowed


@pytest.mark.parametrize(
    ("restriction", "ingredient"),
    [
        ("vegetarian", "boneless skinless chicken breasts"),
        ("vegetarian", "worcestershire sauce"),  # contains anchovies
        ("vegetarian", "chicken stock"),
        ("vegan", "honey"),
        ("vegan", "eggs"),
        ("lactose", "sharp cheddar cheese"),
        ("lactose", "unsalted butter"),
        ("egg", "large eggs"),
        ("egg", "mayonnaise"),
        ("gluten", "all-purpose flour"),
        ("gluten", "worcestershire sauce"),  # usually contains malt vinegar
        ("gluten", "soy sauce"),
        ("nuts", "chopped walnuts"),
        ("nuts", "peanuts"),
        # Regressions found by auditing the most frequent "allowed" ingredients.
        ("gluten", "baguette"),
        ("gluten", "hamburger buns"),
        ("gluten", "cream of mushroom soup"),
        ("gluten", "hoisin sauce"),
        ("gluten", "pancake mix"),
        ("vegetarian", "crabmeat"),
        ("vegetarian", "catfish fillets"),
        ("lactose", "cool whip"),
        ("lactose", "creme fraiche"),
        ("lactose", "non-dairy creamer"),  # may contain caseinate
        # A marker only covers its own restriction.
        ("nuts", "gluten-free almond flour"),
        ("gluten", "dairy-free cookies"),
    ],
)
def test_forbidden_ingredients_are_caught(restriction, ingredient):
    assert RESTRICTIONS[restriction].violates(ingredient)


@pytest.mark.parametrize(
    ("restriction", "ingredient"),
    [
        ("lactose", "coconut milk"),
        ("lactose", "peanut butter"),
        ("lactose", "cream of tartar"),
        ("egg", "eggplant"),
        ("nuts", "nutmeg"),
        ("nuts", "butternut squash"),
        ("nuts", "coconut"),
        ("gluten", "rice flour"),
        ("gluten", "tamari"),
        ("vegetarian", "graham crackers"),  # "ham" inside another word
        ("vegetarian", "tofu"),
        ("vegetarian", "poultry seasoning"),
        ("gluten", "gluten-free all-purpose flour"),
        ("gluten", "rice vermicelli"),
        ("lactose", "dairy-free cheese"),
        ("vegetarian", "vegetarian bacon"),
        ("egg", "eggless mayonnaise"),
    ],
)
def test_safe_lookalikes_are_allowed(restriction, ingredient):
    assert not RESTRICTIONS[restriction].violates(ingredient)


def test_one_bad_ingredient_excludes_the_recipe():
    restrictions = get_restrictions(["vegetarian"])
    assert is_allowed(["tomato", "basil", "olive oil"], restrictions)
    assert not is_allowed(["tomato", "basil", "bacon"], restrictions)


def test_unknown_restriction_fails_loudly():
    with pytest.raises(ValueError, match="unknown restrictions"):
        get_restrictions(["keto-ish"])


def test_recommendations_never_include_forbidden_recipes():
    # The most popular recipe is forbidden; it must not leak into the top-k.
    ingredients = [["bacon", "eggs"], ["tomato", "basil"], ["rice", "beans"], ["chicken"]]
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 1, 2, 3, 5],
            "recipe_id": [0, 0, 0, 0, 3, 3, 1, 2],
            "date": pd.date_range("2020-01-01", periods=8),
        }
    )
    data = InteractionData.from_frame(df)
    allowed = allowed_mask(
        [ingredients[r] for r in data.item_ids], get_restrictions(["vegetarian"])
    )
    model = PopularityRecommender().fit(data)

    recs = recommend(model, data, np.arange(data.n_users), k=4, allowed=allowed)

    for ranked in recs:
        assert allowed[ranked].all()
    # Fewer safe items than k: return fewer items rather than unsafe ones.
    assert all(len(r) <= allowed.sum() for r in recs)
