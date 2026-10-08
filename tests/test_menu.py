import numpy as np

from recipe_recsys.menu import MenuConfig, ingredient_matrix, plan_menu

# Four chicken dishes that are near-duplicates, and two different dishes.
INGREDIENTS = [
    ["chicken", "lemon", "salt"],
    ["chicken", "lemon", "garlic"],
    ["chicken", "lemon", "thyme"],
    ["chicken", "lemon", "rosemary"],
    ["rice", "beans", "onion"],
    ["pasta", "tomato", "basil"],
]
TEXT = np.array(
    [[1, 0, 0], [0.99, 0.14, 0], [0.98, 0.2, 0], [0.97, 0.24, 0], [0, 1, 0], [0, 0, 1]],
    dtype=np.float32,
)
TEXT /= np.linalg.norm(TEXT, axis=1, keepdims=True)
RELEVANCE = np.array([1.0, 0.95, 0.9, 0.85, 0.5, 0.4])


def test_ingredient_matrix_drops_pantry_staples():
    matrix, vocab = ingredient_matrix([["Salt", "chicken"], ["chicken", "water"]])
    assert vocab == ["chicken"]
    assert matrix.toarray().tolist() == [[1.0], [1.0]]


def test_without_weights_the_menu_is_the_top_n():
    matrix, _ = ingredient_matrix(INGREDIENTS)
    menu = plan_menu(RELEVANCE, matrix, TEXT, MenuConfig(size=3, variety=0, reuse=0))
    assert menu.picks == [0, 1, 2]


def test_variety_replaces_near_duplicates():
    matrix, _ = ingredient_matrix(INGREDIENTS)
    plain = plan_menu(RELEVANCE, matrix, TEXT, MenuConfig(size=3, variety=0, reuse=0))
    varied = plan_menu(RELEVANCE, matrix, TEXT, MenuConfig(size=3, variety=1.0, reuse=0))
    assert set(varied.picks) == {0, 4, 5}
    assert varied.mean_similarity < plain.mean_similarity


def test_reuse_shortens_the_shopping_list():
    matrix, _ = ingredient_matrix(INGREDIENTS)
    relevance = np.array([1.0, 0.7, 0.7, 0.7, 0.75, 0.75])
    plain = plan_menu(relevance, matrix, TEXT, MenuConfig(size=2, variety=0, reuse=0))
    reused = plan_menu(relevance, matrix, TEXT, MenuConfig(size=2, variety=0, reuse=1.0))
    assert len(reused.basket) < len(plain.basket)


def test_hard_filter_is_never_violated():
    matrix, _ = ingredient_matrix(INGREDIENTS)
    allowed = np.array([False, True, True, True, True, True])
    menu = plan_menu(RELEVANCE, matrix, TEXT, MenuConfig(size=6), allowed)
    assert 0 not in menu.picks and len(menu.picks) == 5
