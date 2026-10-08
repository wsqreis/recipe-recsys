"""Natural-language recipe search: an LLM parses the request, rules enforce it.

    "algo sem lactose, com frango, em 20 min"
        -> LLM (local, via Ollama) -> {"restrictions": ["lactose"], "include": ["chicken"],
                                       "max_minutes": 20, "query": "quick chicken dish"}
        -> semantic retrieval over recipe text embeddings, using `query`
        -> hard filters (restrictions, time, ingredients) applied after ranking

The LLM only translates the request into a structured query; it never decides
whether a recipe is safe. Restrictions are enforced by the same rule-based
filter as the recommender. Because a restriction the LLM fails to extract would
silently disappear, a keyword detector runs on the raw text as well and the
union of both is used: a spurious restriction hides recipes, a missing one
shows unsafe recipes, so errors are pushed to the safe side.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_recsys.data import DATA_DIR
from recipe_recsys.restrictions import RESTRICTIONS, allowed_mask, get_restrictions

DEFAULT_LLM = "ministral-3:8b"


@dataclass
class ParsedQuery:
    restrictions: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    max_minutes: int | None = None
    query: str = ""


# ---- parsing ------------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "restrictions": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(RESTRICTIONS)},
        },
        "include": {"type": "array", "items": {"type": "string"}},
        "exclude": {"type": "array", "items": {"type": "string"}},
        "max_minutes": {"type": ["integer", "null"]},
        "query": {"type": "string"},
    },
    "required": ["restrictions", "include", "exclude", "max_minutes", "query"],
}

SYSTEM_PROMPT = """\
You turn a recipe request (in any language) into a JSON search query, in English.

Fields:
- restrictions: dietary restrictions the user explicitly asks for, only from this list:
  vegetarian (no meat or fish), vegan (no animal products), lactose (no dairy),
  egg (no eggs), gluten (no gluten), nuts (no tree nuts or peanuts).
  Only add a restriction if the request states it ("without dairy", "I'm vegan",
  "peanut allergy", "I don't eat meat"). Never infer one from the ingredients: a
  request with chicken is not vegetarian. Leaving out one specific ingredient is
  not a restriction: "no pork" is an exclude, not vegetarian; "no butter" is an
  exclude, not lactose; "omelet without onion" has no restriction.
- include: ingredients the user explicitly names as required, as common English
  nouns ("chicken", "peanut butter"). Dish names are not ingredients: "lasagna",
  "curry", "pancakes", "brownies" go in the query, not in include. Never add
  ingredients the user did not name: "vegetable soup" has no required ingredient.
- exclude: specific ingredients the user does not want that are not covered by a
  restriction ("no mushrooms" -> "mushroom"). Empty if none.
- max_minutes: maximum total time in minutes if the user gives one ("quick" alone
  is not a number: use null). "half an hour" = 30, "1 hour" = 60.
- query: a short English description of the dish for semantic search: dish type,
  main ingredients, cuisine, style ("quick chicken dinner", "chocolate birthday cake").

Examples:
"algo sem lactose, com frango, em 20 min" ->
{"restrictions": ["lactose"], "include": ["chicken"], "exclude": [], "max_minutes": 20,
 "query": "quick chicken dish"}
"vegan chocolate cake, no nuts please" ->
{"restrictions": ["vegan", "nuts"], "include": ["chocolate"], "exclude": [], "max_minutes": null,
 "query": "vegan chocolate cake"}
"sopa de legumes sem cogumelo" ->
{"restrictions": [], "include": [], "exclude": ["mushroom"], "max_minutes": null,
 "query": "vegetable soup"}
"""


class OllamaParser:
    """Parses requests with a local LLM served by Ollama (structured JSON output)."""

    def __init__(self, model: str = DEFAULT_LLM, host: str | None = None, timeout: float = 120):
        self.model = model
        # Not OLLAMA_HOST: that one configures the Ollama server itself (often "0.0.0.0").
        url = host or os.environ.get("RECSYS_OLLAMA_URL", "http://localhost:11434")
        self.host = url.rstrip("/")
        self.timeout = timeout

    def parse(self, text: str) -> ParsedQuery:
        payload = {
            "model": self.model,
            "stream": False,
            "format": SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content = json.loads(response.read())["message"]["content"]
        return _to_query(json.loads(content))


# Excluded terms that name a whole restricted category. Excluding "meat" with a
# word filter still lets chicken and fish through, so it becomes the restriction.
_CATEGORY_EXCLUDES: dict[str, str] = {
    "meat": "vegetarian",
    "animal product": "vegan",
    "dairy": "lactose",
    "dairy product": "lactose",
    "milk": "lactose",
    "lactose": "lactose",
    "egg": "egg",
    "gluten": "gluten",
    "nut": "nuts",
    "tree nut": "nuts",
    "peanut": "nuts",
}


def _singular(term: str) -> str:
    return term[:-1] if term.endswith("s") and not term.endswith("ss") else term


def _to_query(raw: dict) -> ParsedQuery:
    minutes = raw.get("max_minutes")
    exclude = [s.strip().lower() for s in raw.get("exclude", []) if s.strip()]
    restrictions = {r for r in raw.get("restrictions", []) if r in RESTRICTIONS}
    restrictions |= {
        _CATEGORY_EXCLUDES[_singular(t)] for t in exclude if _singular(t) in _CATEGORY_EXCLUDES
    }
    return ParsedQuery(
        restrictions=sorted(restrictions),
        include=[s.strip().lower() for s in raw.get("include", []) if s.strip()],
        exclude=exclude,
        max_minutes=int(minutes) if isinstance(minutes, int | float) and minutes > 0 else None,
        query=str(raw.get("query", "")).strip(),
    )


# Phrases that state a restriction, in English and Portuguese. Deliberately broad.
_RESTRICTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vegetarian": ("vegetarian", "vegetariano", "vegetariana", "meatless", "no meat", "sem carne"),
    "vegan": ("vegan", "vegano", "vegana", "plant-based", "plant based"),
    "lactose": (
        "lactose", "dairy-free", "dairy free", "no dairy", "without dairy", "non-dairy",
        "sem leite", "sem laticínios", "sem laticinios",
    ),
    "egg": (
        "egg-free", "egg free", "eggless", "no egg", "no eggs", "without egg", "without eggs",
        "sem ovo", "sem ovos",
    ),
    "gluten": (
        "gluten", "glúten", "celiac", "coeliac", "celíaco", "celiaco", "celíaca", "celiaca",
    ),
    # Not "peanut" alone: "peanut butter cookies" asks for peanuts, not against them.
    "nuts": (
        "nut-free", "nut free", "no nuts", "without nuts", "nut allergy", "peanut allergy",
        "peanut-free", "peanut free", "no peanuts", "sem nozes", "sem castanha", "sem castanhas",
        "sem amendoim", "alergia a nozes", "alergia a castanha", "alergia a amendoim",
    ),
}  # fmt: skip


def detect_restrictions(text: str) -> set[str]:
    """Keyword safety net: restrictions mentioned anywhere in the raw request."""
    lowered = text.lower()
    return {
        name
        for name, phrases in _RESTRICTION_KEYWORDS.items()
        if any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", lowered) for p in phrases)
    }


# ---- retrieval -------------------------------------------------------------------------------


def _term_regex(terms: list[str]) -> str:
    # Word match with an optional plural: "egg" matches "eggs" but not "eggplant".
    return r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")(?:e?s)?\b"


def restriction_masks(recipes: pd.DataFrame, data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    """One allowed-mask per restriction over the whole catalog, cached on disk (slow to build)."""
    path = data_dir / "restriction_masks.npz"
    ids = recipes["recipe_id"].to_numpy()
    if path.exists():
        cached = np.load(path)
        if np.array_equal(cached["recipe_id"], ids):
            return {name: cached[name] for name in RESTRICTIONS}
    ingredients = recipes["ingredients"].tolist()
    masks = {name: allowed_mask(ingredients, get_restrictions([name])) for name in RESTRICTIONS}
    np.savez(path, recipe_id=ids, **masks)
    return masks


class RecipeSearch:
    def __init__(
        self,
        recipes: pd.DataFrame,
        vectors: np.ndarray,
        encode,
        masks: dict[str, np.ndarray],
    ):
        """`recipes` needs recipe_id, name, ingredients and minutes; `vectors` are aligned with it.

        `encode` maps a string to a unit-norm vector in the same space as `vectors`.
        """
        self.recipes = recipes.reset_index(drop=True)
        self.vectors = vectors
        self.encode = encode
        self.masks = masks
        self._ingredient_text = self.recipes["ingredients"].map(" | ".join).str.lower()

    def allowed(self, parsed: ParsedQuery) -> np.ndarray:
        keep = np.ones(len(self.recipes), dtype=bool)
        for name in parsed.restrictions:
            keep &= self.masks[name]
        if parsed.max_minutes is not None:
            # Zero-minute recipes are data errors, not instant meals.
            minutes = self.recipes["minutes"].to_numpy()
            keep &= (minutes > 0) & (minutes <= parsed.max_minutes)
        for term in parsed.include:
            keep &= self._ingredient_text.str.contains(_term_regex([term])).to_numpy()
        if parsed.exclude:
            keep &= ~self._ingredient_text.str.contains(_term_regex(parsed.exclude)).to_numpy()
        return keep

    def search(self, parsed: ParsedQuery, k: int = 10) -> pd.DataFrame:
        text = parsed.query or " ".join(parsed.include) or "recipe"
        scores = self.vectors @ self.encode(text)
        scores[~self.allowed(parsed)] = -np.inf
        top = np.argsort(-scores, kind="stable")[:k]
        top = top[np.isfinite(scores[top])]
        return self.recipes.iloc[top].assign(score=scores[top])


def parse_request(text: str, parser: OllamaParser | None) -> tuple[ParsedQuery, set[str]]:
    """LLM parse (if any) plus the keyword safety net. Returns the query and the keyword hits."""
    parsed = parser.parse(text) if parser is not None else ParsedQuery(query=text)
    keywords = detect_restrictions(text)
    parsed.restrictions = sorted(set(parsed.restrictions) | keywords)
    return parsed, keywords


def as_dict(parsed: ParsedQuery) -> dict:
    return asdict(parsed)
