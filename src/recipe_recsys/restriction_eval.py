"""Measure the restriction filter against labeled recipes.

The recommender reports "0 violations", but that check uses the filter to
audit itself. Here the filter (keywords over ingredients) is compared with
independent labels (evaluation/restriction_labels.jsonl, see
evaluation/restriction_labeling.md for how they were made), together with an
LLM classifier and the union of both.

The positive class is "forbidden": recall is the safety metric (share of
unsafe recipes the filter blocks), precision is the usefulness metric (share
of blocked recipes that were really unsafe). Uncertain labels are excluded.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_recsys.evaluate import bootstrap_ci
from recipe_recsys.restrictions import RESTRICTIONS, get_restrictions, is_allowed
from recipe_recsys.search import OllamaParser

DEFAULT_LABELS = Path("evaluation/restriction_labels.jsonl")

CLASSIFIER_PROMPT = """\
You check whether a recipe is compatible with dietary restrictions, from its name
and ingredient list. Judge each ingredient by its most common US commercial form
(margarine usually contains whey; worcestershire sauce contains anchovies and
malt vinegar; regular soy sauce contains wheat). Trust explicit labels such as
"gluten-free flour" or "vegan butter". Optional ingredients count.

- vegetarian: no meat, poultry, fish, seafood, gelatin, meat or fish broth/stock, lard.
- vegan: vegetarian, and no dairy, eggs, honey or other animal products.
- lactose: no dairy (milk, butter, cream, cheese, yogurt, whey, casein, ghee).
- egg: no eggs or egg products (mayonnaise, meringue, egg noodles).
- gluten: no wheat, barley, rye, malt, regular oats, or products made from them.
- nuts: no tree nuts, peanuts or products made from them (coconut and nutmeg are fine).

Answer true if the recipe is compatible with the restriction, false otherwise.
"""

SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "boolean"} for name in sorted(RESTRICTIONS)},
    "required": sorted(RESTRICTIONS),
}


class LLMClassifier:
    """Asks a local LLM (Ollama) whether a recipe is compatible with each restriction."""

    def __init__(self, parser: OllamaParser):
        self.parser = parser  # reused for the model name, host and timeout

    def forbidden(self, name: str, ingredients: list[str]) -> set[str]:
        payload = {
            "model": self.parser.model,
            "stream": False,
            "format": SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"{name}\nIngredients: {'; '.join(ingredients)}"},
            ],
        }
        request = urllib.request.Request(
            f"{self.parser.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.parser.timeout) as response:
            answer = json.loads(json.loads(response.read())["message"]["content"])
        return {r for r in RESTRICTIONS if answer.get(r) is False}


def keyword_forbidden(ingredients: list[str]) -> set[str]:
    return {r for r in RESTRICTIONS if not is_allowed(ingredients, get_restrictions([r]))}


def load_labels(path: Path = DEFAULT_LABELS) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


@dataclass
class ClassifierScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    # Per labeled (recipe, restriction) pair: 1 if a forbidden pair was caught (for recall CIs).
    caught: list[int] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")


def evaluate_restrictions(
    labels: pd.DataFrame,
    recipes: pd.DataFrame,
    classifiers: dict[str, Callable[[str, list[str]], set[str]]],
) -> dict[str, dict[str, ClassifierScores]]:
    """Scores per classifier and per restriction (plus "all")."""
    recipes = recipes.set_index("recipe_id")
    scores = {
        c: {r: ClassifierScores() for r in [*sorted(RESTRICTIONS), "all"]} for c in classifiers
    }
    for row in labels.itertuples():
        name = recipes.at[row.recipe_id, "name"]
        ingredients = list(recipes.at[row.recipe_id, "ingredients"])
        predictions = {c: fn(name, ingredients) for c, fn in classifiers.items()}
        for restriction in sorted(RESTRICTIONS):
            if restriction in row.uncertain:
                continue
            truth = restriction in row.forbidden
            for c, predicted in predictions.items():
                hit = restriction in predicted
                for s in (scores[c][restriction], scores[c]["all"]):
                    s.tp += truth and hit
                    s.fp += (not truth) and hit
                    s.fn += truth and not hit
                    s.tn += (not truth) and not hit
                    if truth:
                        s.caught.append(int(hit))
                if truth and not hit:
                    scores[c][restriction].misses.append(f"{name} ({'; '.join(ingredients)})")
    return scores


def scores_markdown(scores: dict[str, dict[str, ClassifierScores]], n_recipes: int) -> str:
    lines = [
        f"{n_recipes} random recipes, labeled forbidden / allowed / uncertain per restriction "
        "(uncertain pairs excluded). Positive class: forbidden.",
        "",
        "| classifier | precision | recall (safety) | recall 95% CI | unsafe recipes missed |",
        "|---|---|---|---|---|",
    ]
    for c, per in scores.items():
        s = per["all"]
        low, high = bootstrap_ci(np.asarray(s.caught, dtype=float))
        lines.append(
            f"| {c} | {s.precision():.1%} | {s.recall():.1%} | [{low:.1%}, {high:.1%}] "
            f"| {s.fn} of {s.tp + s.fn} |"
        )
    restrictions = sorted(RESTRICTIONS)
    lines += [
        "",
        "Recall per restriction (forbidden pairs caught / forbidden pairs):",
        "",
        "| classifier | " + " | ".join(restrictions) + " |",
        "|---|" + "---|" * len(restrictions),
    ]
    for c, per in scores.items():
        cells = [f"{per[r].tp}/{per[r].tp + per[r].fn}" for r in restrictions]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Precision per restriction (correct blocks / blocks):", ""]
    lines += ["| classifier | " + " | ".join(restrictions) + " |"]
    lines += ["|---|" + "---|" * len(restrictions)]
    for c, per in scores.items():
        cells = [f"{per[r].tp}/{per[r].tp + per[r].fp}" for r in restrictions]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    for c, per in scores.items():
        misses = [(r, m) for r in restrictions for m in per[r].misses]
        if misses:
            lines += ["", f"Unsafe recipes missed by {c}:", ""]
            lines += [f"- {r}: {m}" for r, m in misses]
    return "\n".join(lines)
