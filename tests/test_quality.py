import pytest

from recipe_recsys.quality import normalize_ingredient, plausible_calories, plausible_minutes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("garlic cloves", "garlic"),
        ("Minced Garlic Cloves", "garlic"),
        ("fresh garlic", "garlic"),
        ("onions", "onion"),
        ("tomatoes", "tomato"),
        ("fresh raspberries", "raspberry"),
        ("lemon, juice of", "lemon"),
        ("fresh lemon juice", "lemon"),
        ("lime zest", "lime"),
        ("egg whites", "egg"),
        ("eggs", "egg"),
        ("boneless skinless chicken breast halves", "chicken breast"),
        ("fresh parsley leaves", "parsley"),
        # Different products stay different.
        ("green onions", "green onion"),
        ("red onion", "red onion"),
        ("dried parsley", "dried parsley"),
        ("egg noodles", "egg noodle"),
        ("orange juice", "orange juice"),
        ("garlic powder", "garlic powder"),
        ("crushed red pepper", "crushed red pepper"),
        ("diced tomatoes", "diced tomato"),
        # Not plurals.
        ("molasses", "molasses"),
        ("asparagus", "asparagus"),
        ("swiss cheese", "swiss cheese"),
    ],
)
def test_normalize_ingredient(raw, expected):
    assert normalize_ingredient(raw) == expected


def test_plausible_numbers():
    assert plausible_minutes(30) and plausible_minutes(3 * 24 * 60)
    assert not plausible_minutes(0) and not plausible_minutes(2_147_483_647)
    assert plausible_calories(450) and not plausible_calories(0)
    assert not plausible_calories(434_360)
