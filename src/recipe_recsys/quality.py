"""Data quality: ingredient names for shopping lists, and implausible numbers.

Food.com ingredient strings are free text: "garlic cloves", "garlic clove",
"fresh garlic" and "minced garlic cloves" are one thing to buy. For shopping
lists (and the menu's reuse bonus) they are normalized with small, explicit
rules. The rules are conservative: different products stay different ("red
onion" vs "green onion", "dried parsley" vs "parsley"), because merging them
would put the wrong thing in the basket.

The normalization is NOT used by the restriction filter (which needs the
original text, e.g. "gluten-free" markers) nor by the models (their results
were measured on the original vocabulary).

Numbers: some recipes take 0 minutes or 2,147,483,647 minutes (the largest
32-bit integer, an error value), and some list the whole recipe's calories as
one serving. Values outside a plausible range are treated as unknown.
"""

from __future__ import annotations

import re

# Words that describe preparation or size, not a different product.
_DROP = {
    "fresh", "freshly", "minced", "chopped", "sliced", "large", "small", "medium",
    "boneless", "skinless", "finely", "coarsely", "thinly", "peeled",
}  # fmt: skip
# Not dropped on purpose: "crushed red pepper" is chili flakes, not a bell pepper, and
# "diced tomatoes" / "crushed tomatoes" are usually canned, a different product.
# Trailing words that name a part of the product you buy whole.
_PARTS = {"clove", "cloves", "halves", "half", "sprig", "sprigs", "leaves", "leaf"}
# Citrus juice and zest come from the fruit on the list.
_CITRUS = {"lemon", "lime"}
_CITRUS_PARTS = {"juice", "zest", "peel", "rind"}
_EGG_PARTS = {"white", "whites", "yolk", "yolks"}
# Plural-looking words that are not plurals.
_NOT_PLURAL = {"molasses", "grits", "swiss", "brussels", "hummus", "asparagus", "couscous"}

MAX_MINUTES = 7 * 24 * 60  # longer than a week is a data error, not a recipe
MAX_CALORIES = 3000  # per serving; above this the value is most likely the whole recipe


def _singular(word: str) -> str:
    if word in _NOT_PLURAL or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def normalize_ingredient(name: str) -> str:
    """Shopping-list name: 'Minced Garlic Cloves' -> 'garlic', 'lemon, juice of' -> 'lemon'."""
    text = name.strip().lower()
    # "lemon, juice of" / "orange, zest of" -> "lemon juice" / "orange zest"
    if match := re.fullmatch(r"(.+?),\s*(\w+) of", text):
        text = f"{match.group(1)} {match.group(2)}"
    words = [w for w in re.split(r"\s+", text) if w and w not in _DROP]
    while len(words) > 1 and words[-1] in _PARTS:
        words.pop()
    if len(words) == 2 and words[0] in _CITRUS and words[1] in _CITRUS_PARTS:
        words = words[:1]
    if len(words) == 2 and words[0] == "egg" and words[1] in _EGG_PARTS:
        words = ["egg"]  # whites and yolks: you buy eggs (but egg noodles are noodles)
    words = [*words[:-1], _singular(words[-1])] if words else words
    return " ".join(words) or text


def plausible_minutes(minutes: float) -> bool:
    return 0 < minutes <= MAX_MINUTES


def plausible_calories(calories: float) -> bool:
    return 0 < calories <= MAX_CALORIES


def quality_report(recipes) -> str:
    """Markdown summary of the issues above, on the full recipe table."""
    from collections import Counter

    raw = Counter(i.strip().lower() for ing in recipes["ingredients"] for i in ing)
    normalized: dict[str, set[str]] = {}
    for name in raw:
        normalized.setdefault(normalize_ingredient(name), set()).add(name)
    merged = sorted(
        ((k, v) for k, v in normalized.items() if len(v) > 1),
        key=lambda kv: -sum(raw[n] for n in kv[1]),
    )
    minutes, calories = recipes["minutes"], recipes["calories"]
    lines = [
        f"{len(recipes):,} recipes.",
        "",
        "| check | count |",
        "|---|---|",
        f"| distinct ingredient strings | {len(raw):,} |",
        f"| distinct shopping items after normalization | {len(normalized):,} |",
        f"| recipes with 0 minutes | {(minutes == 0).sum():,} |",
        f"| recipes over {MAX_MINUTES:,} minutes (one week) | {(minutes > MAX_MINUTES).sum():,} |",
        f"| largest minutes value | {minutes.max():,} |",
        f"| recipes with 0 calories | {(calories == 0).sum():,} |",
        f"| recipes over {MAX_CALORIES:,} kcal per serving | {(calories > MAX_CALORIES).sum():,} |",
        f"| largest calories value | {calories.max():,.0f} |",
        "",
        "Most frequent merges (shopping item <- ingredient strings):",
        "",
    ]
    for item, names in merged[:15]:
        variants = ", ".join(f"{n} ({raw[n]:,})" for n in sorted(names, key=lambda n: -raw[n]))
        lines.append(f"- **{item}** <- {variants}")
    return "\n".join(lines)
