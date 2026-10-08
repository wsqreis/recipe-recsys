"""Evaluation of natural-language search on a hand-labeled set of requests.

Each request is labeled with the restrictions, required/excluded ingredients
and time limit it states (evaluation/search_queries.jsonl). The labels were
written before looking at the parser's output.

Safety comes first: what matters most is the recall of restrictions (a missed
restriction shows unsafe recipes) and, end to end, whether any returned recipe
violates a restriction the user actually stated. Precision matters too: a
spurious restriction silently hides recipes the user wanted.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recipe_recsys.search import (
    OllamaParser,
    ParsedQuery,
    RecipeSearch,
    detect_restrictions,
)

DEFAULT_QUERIES = Path("evaluation/search_queries.jsonl")


def load_queries(path: Path = DEFAULT_QUERIES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _norm(term: str) -> str:
    term = term.strip().lower()
    return term[:-1] if term.endswith("s") and not term.endswith("ss") and len(term) > 3 else term


def terms_match(predicted: list[str], expected: list[str]) -> bool:
    """Lenient set equality: "chicken breast" counts as "chicken", "potatoes" as "potato"."""
    pred, gold = [_norm(t) for t in predicted], [_norm(t) for t in expected]

    def covered(term: str, others: list[str]) -> bool:
        return any(term in other or other in term for other in others)

    return all(covered(g, pred) for g in gold) and all(covered(p, gold) for p in pred)


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    missed_queries: int = 0

    def add(self, predicted: set[str], expected: set[str]) -> None:
        self.tp += len(predicted & expected)
        self.fp += len(predicted - expected)
        self.fn += len(expected - predicted)
        self.missed_queries += bool(expected - predicted)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0


@dataclass
class SearchEvalReport:
    n_queries: int
    restrictions: dict[str, _Counts]
    max_minutes_accuracy: float
    include_accuracy: float
    exclude_accuracy: float
    with_results: float
    results_checked: int
    violating_results: int
    median_latency: float
    failures: list[dict] = field(default_factory=list)

    def markdown(self) -> str:
        lines = [
            f"{self.n_queries} labeled requests (Portuguese and English).",
            "",
            "| restrictions from | precision | recall | requests with a missed restriction |",
            "|---|---|---|---|",
        ]
        for source, c in self.restrictions.items():
            lines.append(
                f"| {source} | {c.precision():.0%} | {c.recall():.0%} "
                f"| {c.missed_queries} of {self.n_queries} |"
            )
        lines += [
            "",
            "| end to end (union) | value |",
            "|---|---|",
            f"| max minutes parsed correctly | {self.max_minutes_accuracy:.0%} |",
            f"| required ingredients parsed correctly | {self.include_accuracy:.0%} |",
            f"| excluded ingredients parsed correctly | {self.exclude_accuracy:.0%} |",
            f"| requests with at least one result | {self.with_results:.0%} |",
            f"| results violating a stated restriction | {self.violating_results} of "
            f"{self.results_checked} |",
            f"| median LLM latency | {self.median_latency:.1f} s |",
        ]
        if self.failures:
            lines += ["", "Parsing errors:", ""]
            lines += [f"- `{f['text']}`: {', '.join(f['errors'])}" for f in self.failures]
        return "\n".join(lines)


def evaluate_search(
    queries: list[dict], parser: OllamaParser, searcher: RecipeSearch, k: int = 10
) -> SearchEvalReport:
    counts = {"LLM": _Counts(), "keywords": _Counts(), "LLM + keywords": _Counts()}
    minutes_ok = include_ok = exclude_ok = with_results = checked = violating = 0
    latencies, failures = [], []

    for q in queries:
        expected = set(q["restrictions"])
        start = time.perf_counter()
        llm = parser.parse(q["text"])
        latencies.append(time.perf_counter() - start)
        keywords = detect_restrictions(q["text"])
        union = set(llm.restrictions) | keywords
        counts["LLM"].add(set(llm.restrictions), expected)
        counts["keywords"].add(keywords, expected)
        counts["LLM + keywords"].add(union, expected)

        errors = []
        if expected - union:
            errors.append(f"missed {sorted(expected - union)}")
        if union - expected:
            errors.append(f"spurious {sorted(union - expected)}")
        if llm.max_minutes == q["max_minutes"]:
            minutes_ok += 1
        else:
            errors.append(f"minutes {llm.max_minutes} (expected {q['max_minutes']})")
        if terms_match(llm.include, q["include"]):
            include_ok += 1
        else:
            errors.append(f"include {llm.include} (expected {q['include']})")
        if terms_match(llm.exclude, q["exclude"]):
            exclude_ok += 1
        else:
            errors.append(f"exclude {llm.exclude} (expected {q['exclude']})")
        if errors:
            failures.append({"text": q["text"], "errors": errors})

        final = ParsedQuery(sorted(union), llm.include, llm.exclude, llm.max_minutes, llm.query)
        results = searcher.search(final, k)
        with_results += not results.empty
        rows = results.index.to_numpy()
        checked += len(rows)
        # Judged against what the user stated, not what the parser understood.
        unsafe = np.zeros(len(rows), dtype=bool)
        for name in expected:
            unsafe |= ~searcher.masks[name][rows]
        violating += int(unsafe.sum())

    n = len(queries)
    return SearchEvalReport(
        n_queries=n,
        restrictions=counts,
        max_minutes_accuracy=minutes_ok / n,
        include_accuracy=include_ok / n,
        exclude_accuracy=exclude_ok / n,
        with_results=with_results / n,
        results_checked=checked,
        violating_results=violating,
        median_latency=statistics.median(latencies),
        failures=failures,
    )
