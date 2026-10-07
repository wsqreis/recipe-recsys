"""Hard dietary restrictions.

A recipe is excluded if ANY ingredient matches a forbidden term. This is a
safety filter, not a ranking signal: a restricted recipe must never be
recommended, no matter how high it scores.

Design choices:
- Ingredients, not tags. Food.com tags are user-provided and unreliable
  (e.g. recipes tagged `gluten-free` that use worcestershire sauce, which
  usually contains malt).
- Conservative by default. False positives (hiding a safe recipe) are cheap;
  false negatives (showing an unsafe one) are not. Known safe phrases such as
  "coconut milk" are whitelisted explicitly instead of loosening the rules.
- Keyword matching is a heuristic. It cannot see hidden ingredients inside
  processed products. A real product needs curated ingredient data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Restriction:
    name: str
    forbidden: tuple[str, ...]
    # Phrases removed before matching, e.g. "coconut milk" for lactose.
    allowed: tuple[str, ...] = ()
    # Labels that make the rest of the ingredient safe, e.g. "gluten-free" in
    # "gluten-free all-purpose flour".
    free_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_forbidden_re", _word_regex(self.forbidden))
        object.__setattr__(self, "_allowed_re", _optional_regex(self.allowed))
        object.__setattr__(self, "_marker_re", _optional_regex(self.free_markers))

    def violates(self, ingredient: str) -> bool:
        text = ingredient.lower()
        if self._marker_re is not None and (marker := self._marker_re.search(text)):
            text = text[: marker.start()]
        if self._allowed_re is not None:
            text = self._allowed_re.sub(" ", text)
        return self._forbidden_re.search(text) is not None


def _word_regex(terms: Iterable[str]) -> re.Pattern[str]:
    # Longest first so multi-word terms win; optional plural "s"/"es".
    alternatives = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")(?:e?s)?\b")


def _optional_regex(terms: tuple[str, ...]) -> re.Pattern[str] | None:
    return _word_regex(terms) if terms else None


# fmt: off
_MEAT = (
    "beef", "steak", "veal", "pork", "ham", "bacon", "prosciutto", "pancetta", "sausage",
    "chorizo", "pepperoni", "salami", "hot dog", "lamb", "mutton", "goat", "venison", "bison",
    "chicken", "turkey", "duck", "goose", "hen", "meat", "ground round", "chuck", "brisket",
    "ribs", "gelatin", "lard", "suet", "bone broth", "beef broth", "chicken broth",
    "chicken stock", "beef stock", "bouillon", "meatball", "meatloaf", "kielbasa", "andouille",
    "bratwurst", "pastrami", "jerky", "liver", "giblets", "sirloin", "ribeye", "spam",
    "drumstick",
)
_SEAFOOD = (
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "anchovy", "anchovies",
    "sardine", "mackerel", "shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
    "scallop", "squid", "calamari", "octopus", "fish sauce", "worcestershire sauce",
    # Compound words are missed by word boundaries, so they are listed explicitly.
    "crabmeat", "catfish", "swordfish", "monkfish", "whitefish", "shellfish", "crayfish",
    "crawfish", "bluefish", "haddock", "pollock", "snapper", "grouper", "flounder", "sole",
    "herring", "sea bass", "mahi mahi", "lox", "caviar", "roe", "surimi",
)
_DAIRY = (
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "buttermilk", "whey", "casein",
    "caseinate", "creamer", "ghee", "half-and-half", "half and half", "sour cream", "cream cheese",
    "mozzarella", "parmesan", "cheddar", "ricotta", "feta", "brie", "custard", "ice cream",
    "creme fraiche", "crème fraîche", "mascarpone", "gruyere", "gouda", "romano", "asiago",
    "provolone", "paneer", "queso", "kefir", "velveeta", "dulce de leche", "ganache",
    "cool whip", "whipped topping", "chocolate chip", "white chocolate", "margarine",
)
_DAIRY_LOOKALIKES = (
    "coconut milk", "coconut cream", "almond milk", "soy milk", "soymilk", "rice milk",
    "oat milk", "cashew milk", "peanut butter", "almond butter", "cashew butter", "apple butter",
    "cocoa butter", "nut butter", "sunflower butter", "cream of tartar",
)
_EGG = (
    "egg", "eggs", "egg white", "egg yolk", "mayonnaise", "mayo", "meringue", "eggnog",
    "aioli", "hollandaise", "custard", "brioche", "challah", "ladyfinger",
)
_GLUTEN = (
    "wheat", "flour", "bread", "breadcrumbs", "bread crumbs", "panko", "crouton", "pasta",
    "spaghetti", "macaroni", "noodle", "lasagna", "couscous", "barley", "rye", "malt",
    "semolina", "farina", "bulgur", "seitan", "spelt", "cracker", "biscuit", "tortilla",
    "pie crust", "pita", "bagel", "beer", "soy sauce", "teriyaki sauce",
    "worcestershire sauce", "cake mix", "brownie mix", "graham", "pretzel", "oats", "oatmeal",
    "bun", "roll", "baguette", "croissant", "brioche", "ciabatta", "focaccia", "sourdough",
    "muffin", "naan", "pastry", "phyllo", "filo", "dough", "crust", "cookie", "wafer", "cake",
    "gnocchi", "orzo", "ravioli", "tortellini", "fettuccine", "linguine", "penne", "rigatoni",
    "ziti", "vermicelli", "udon", "ramen", "wonton", "dumpling", "matzo", "stuffing", "bisquick",
    "pancake", "cupcake", "cheesecake", "shortcake", "cornbread", "flatbread", "shortbread",
    "gingerbread", "breadstick", "piecrust", "hoisin sauce", "gravy", "roux", "bran", "durum",
    "farro", "kamut", "triticale",
    # Condensed "cream of ..." soups are thickened with wheat flour.
    "cream of mushroom soup", "cream of chicken soup", "cream of celery soup",
)
_GLUTEN_LOOKALIKES = (
    "rice flour", "almond flour", "coconut flour", "corn flour", "tapioca flour",
    "potato flour", "chickpea flour", "buckwheat flour", "corn tortilla", "rice noodle",
    "rice vermicelli", "rice paper", "tamari", "root beer", "ginger beer", "rice cake",
)
_TREE_NUTS_AND_PEANUTS = (
    "nut", "peanut", "almond", "walnut", "pecan", "cashew", "hazelnut", "filbert",
    "pistachio", "macadamia", "brazil nut", "pine nut", "praline", "marzipan", "nutella",
    "nougat", "pesto", "amaretto", "frangelico", "gianduja", "baklava",
)
_NUT_LOOKALIKES = ("nutmeg", "butternut", "coconut", "water chestnut", "doughnut", "donut")
# fmt: on

RESTRICTIONS: dict[str, Restriction] = {
    r.name: r
    for r in [
        Restriction(
            "vegetarian",
            _MEAT + _SEAFOOD,
            free_markers=("vegetarian", "vegan", "meatless", "meat-free", "plant-based"),
        ),
        Restriction(
            "vegan",
            _MEAT + _SEAFOOD + _DAIRY + _EGG + ("honey",),
            allowed=_DAIRY_LOOKALIKES,
            free_markers=("vegan", "plant-based"),
        ),
        Restriction(
            "lactose",
            _DAIRY,
            allowed=_DAIRY_LOOKALIKES,
            # Not "non-dairy": in the US it may legally contain caseinate (milk protein).
            free_markers=("dairy-free", "dairy free", "lactose-free", "lactose free", "vegan"),
        ),
        Restriction("egg", _EGG, free_markers=("egg-free", "eggless", "vegan")),
        Restriction(
            "gluten",
            _GLUTEN,
            allowed=_GLUTEN_LOOKALIKES,
            free_markers=("gluten-free", "gluten free"),
        ),
        Restriction(
            "nuts",
            _TREE_NUTS_AND_PEANUTS,
            allowed=_NUT_LOOKALIKES,
            free_markers=("nut-free", "nut free"),
        ),
    ]
}


def get_restrictions(names: Iterable[str]) -> list[Restriction]:
    unknown = set(names) - RESTRICTIONS.keys()
    if unknown:
        raise ValueError(f"unknown restrictions {sorted(unknown)}; options: {sorted(RESTRICTIONS)}")
    return [RESTRICTIONS[n] for n in names]


def is_allowed(ingredients: Sequence[str], restrictions: Sequence[Restriction]) -> bool:
    return not any(r.violates(ing) for r in restrictions for ing in ingredients)


def allowed_mask(
    ingredients_per_item: Sequence[Sequence[str]], restrictions: Sequence[Restriction]
) -> np.ndarray:
    """Boolean mask aligned with `ingredients_per_item`: True = safe to recommend."""
    return np.fromiter(
        (is_allowed(ings, restrictions) for ings in ingredients_per_item),
        dtype=bool,
        count=len(ingredients_per_item),
    )
