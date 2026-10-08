"""HTTP API: natural-language search and recommendations for new users.

    uvicorn recipe_recsys.api:app          (or `recsys serve`)

Everything heavy (recipes, text embeddings, restriction masks, the iALS model)
is built once at startup into a `ServingState`. `create_app` takes that state
as an argument, so tests can serve a four-recipe catalog without loading data,
the sentence encoder or the LLM.

Recommendations are for users who are not in the data: the request carries the
recipes the person has cooked, and iALS folds them in (the same closed-form
update as in training, no retraining). With no history, it falls back to
popularity, which phase 3 showed is what works best for a first visit.
Restrictions are always a hard filter after ranking.
"""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recipe_recsys.dataset import InteractionData
from recipe_recsys.models import Recommender
from recipe_recsys.restrictions import RESTRICTIONS
from recipe_recsys.search import OllamaParser, ParsedQuery, RecipeSearch, as_dict, parse_request

WEB_DIST = Path("web/dist")


@dataclass
class ServingState:
    recipes: pd.DataFrame  # indexed by recipe_id
    search: RecipeSearch
    parser: OllamaParser | None
    catalog: InteractionData  # interactions the recommender was trained on
    model: Recommender  # must implement score_histories
    popularity: np.ndarray  # interactions per catalog item, for the empty-history fallback


# ---- schemas ------------------------------------------------------------------------------


class RecipeCard(BaseModel):
    recipe_id: int
    name: str
    minutes: int
    ingredients: list[str]
    calories: float | None = None
    score: float | None = None


class RecipeDetail(RecipeCard):
    description: str | None = None
    steps: list[str] = []
    tags: list[str] = []


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    n: int = Field(default=12, ge=1, le=50)
    use_llm: bool = True


class SearchResponse(BaseModel):
    parsed: dict
    keyword_restrictions: list[str]
    llm_used: bool
    llm_error: str | None = None
    results: list[RecipeCard]


class RecommendRequest(BaseModel):
    cooked: list[int] = Field(default=[], max_length=200)
    restrictions: list[str] = []
    n: int = Field(default=12, ge=1, le=50)


class RecommendResponse(BaseModel):
    strategy: str  # "personalized" or "popularity"
    used_history: list[int]  # cooked recipes the model knows
    ignored_history: list[int]  # cooked recipes outside the model's catalog
    results: list[RecipeCard]


# ---- app ----------------------------------------------------------------------------------


def _card(recipes: pd.DataFrame, recipe_id: int, score: float | None = None) -> RecipeCard:
    row = recipes.loc[recipe_id]
    calories = row.get("calories")
    return RecipeCard(
        recipe_id=int(recipe_id),
        name=str(row["name"]),
        minutes=int(row["minutes"]),
        ingredients=list(row["ingredients"]),
        calories=float(calories) if calories is not None and pd.notna(calories) else None,
        score=score,
    )


def create_app(state: ServingState) -> FastAPI:
    app = FastAPI(title="recipe-recsys", version="0.1.0")
    recipes = state.recipes

    @app.get("/api/health")
    def health() -> dict:
        return {"recipes": len(recipes), "llm": state.parser.model if state.parser else None}

    @app.get("/api/restrictions")
    def restrictions() -> list[str]:
        return sorted(RESTRICTIONS)

    @app.post("/api/search")
    def search(request: SearchRequest) -> SearchResponse:
        parser = state.parser if request.use_llm else None
        llm_error = None
        try:
            parsed, keywords = parse_request(request.query, parser)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            # The LLM is optional: without it, keywords + semantic search still work.
            llm_error = f"LLM unavailable ({e}); used keywords only"
            parsed, keywords = parse_request(request.query, None)
            parser = None
        results = state.search.search(parsed, k=request.n)
        return SearchResponse(
            parsed=as_dict(parsed),
            keyword_restrictions=sorted(keywords),
            llm_used=parser is not None,
            llm_error=llm_error,
            results=[
                _card(recipes, rid, float(s))
                for rid, s in zip(results["recipe_id"], results["score"], strict=True)
            ],
        )

    @app.post("/api/recommend")
    def recommend(request: RecommendRequest) -> RecommendResponse:
        unknown = set(request.restrictions) - RESTRICTIONS.keys()
        if unknown:
            raise HTTPException(422, f"unknown restrictions {sorted(unknown)}")
        index = state.catalog.item_index
        used = [r for r in dict.fromkeys(request.cooked) if r in index]
        ignored = [r for r in dict.fromkeys(request.cooked) if r not in index]

        n_items = state.catalog.n_items
        if used:
            cols = np.array([index[r] for r in used])
            history = sp.csr_matrix(
                (np.ones(len(cols), dtype=np.float32), (np.zeros(len(cols), dtype=int), cols)),
                shape=(1, n_items),
            )
            scores = np.asarray(state.model.score_histories(history), dtype=np.float32)[0]
            scores[cols] = -np.inf  # already cooked
            strategy = "personalized"
        else:
            scores = state.popularity.astype(np.float32).copy()
            strategy = "popularity"

        # Same masks as search, aligned to the catalog through recipe ids.
        positions = state.search.positions(state.catalog.item_ids)
        allowed = state.search.allowed(ParsedQuery(restrictions=sorted(request.restrictions)))
        scores[~allowed[positions]] = -np.inf

        top = np.argsort(-scores, kind="stable")[: request.n]
        top = top[np.isfinite(scores[top])]
        return RecommendResponse(
            strategy=strategy,
            used_history=used,
            ignored_history=ignored,
            results=[_card(recipes, state.catalog.item_ids[i], float(scores[i])) for i in top],
        )

    @app.get("/api/recipes")
    def find_recipes(q: str = "", n: int = 20) -> list[RecipeCard]:
        """Name lookup among recipes the recommender knows, most cooked first."""
        ids = state.catalog.item_ids
        names = recipes.loc[ids, "name"].str.lower().to_numpy()
        terms = q.lower().split()
        match = np.array([all(t in name for t in terms) for name in names], dtype=bool)
        order = np.argsort(-state.popularity, kind="stable")
        hits = [i for i in order if match[i]][: max(1, min(n, 50))]
        return [_card(recipes, ids[i]) for i in hits]

    @app.get("/api/recipes/{recipe_id}")
    def recipe(recipe_id: int) -> RecipeDetail:
        if recipe_id not in recipes.index:
            raise HTTPException(404, "recipe not found")
        row = recipes.loc[recipe_id]
        description = row.get("description")
        return RecipeDetail(
            **_card(recipes, recipe_id).model_dump(),
            description=description if isinstance(description, str) else None,
            steps=list(row.get("steps", [])),
            tags=list(row.get("tags", [])),
        )

    if WEB_DIST.exists():
        # The built frontend (web/), served from the same origin as the API.
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def load_state(llm: str | None = None) -> ServingState:
    """Build the serving state from data/processed (slow: ~1 minute)."""
    from recipe_recsys import data
    from recipe_recsys.models.ials import IALSRecommender
    from recipe_recsys.search import DEFAULT_LLM, restriction_masks
    from recipe_recsys.split import k_core
    from recipe_recsys.text import load_text_embeddings, query_encoder

    recipes = data.load_recipes()
    vectors = load_text_embeddings(recipes["recipe_id"].to_numpy())
    search_columns = recipes[["recipe_id", "name", "ingredients", "minutes"]]
    search = RecipeSearch(search_columns, vectors, query_encoder(), restriction_masks(recipes))

    # Serving uses every interaction: there is nothing left to evaluate here.
    catalog = InteractionData.from_frame(k_core(data.load_interactions()))
    model = IALSRecommender().fit(catalog)
    popularity = np.asarray(catalog.matrix.sum(axis=0)).ravel()
    return ServingState(
        recipes=recipes.set_index("recipe_id"),
        search=search,
        parser=OllamaParser(llm or DEFAULT_LLM),
        catalog=catalog,
        model=model,
        popularity=popularity,
    )


def __getattr__(name: str):
    # `uvicorn recipe_recsys.api:app` builds the real state lazily on first access.
    if name == "app":
        return create_app(load_state())
    raise AttributeError(name)
