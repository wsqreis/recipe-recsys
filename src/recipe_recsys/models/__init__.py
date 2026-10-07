from importlib import import_module

from recipe_recsys.models.base import Recommender
from recipe_recsys.models.baselines import (
    PopularityRecommender,
    RandomRecommender,
    RecentPopularityRecommender,
)
from recipe_recsys.models.ials import IALSRecommender
from recipe_recsys.models.itemknn import ItemKNNRecommender

# Model name -> "module:Class". Imported on demand so that models with heavy
# optional dependencies (PyTorch) do not break the others when not installed.
MODELS: dict[str, str] = {
    "random": "baselines:RandomRecommender",
    "popularity": "baselines:PopularityRecommender",
    "recent_popularity": "baselines:RecentPopularityRecommender",
    "itemknn": "itemknn:ItemKNNRecommender",
    "ials": "ials:IALSRecommender",
    "two_tower": "two_tower:TwoTowerRecommender",
}


def get_model_class(name: str) -> type[Recommender]:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; options: {sorted(MODELS)}")
    module, _, cls = MODELS[name].partition(":")
    try:
        return getattr(import_module(f"recipe_recsys.models.{module}"), cls)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"model {name!r} needs {e.name!r}: run `uv sync --extra cpu` (or `--extra cuda`)"
        ) from e


def _parse_value(value: str) -> int | float | str:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def build_model(spec: str) -> Recommender:
    """Build from `name` or `name:param=value,param=value`, e.g. `itemknn:neighbors=50`."""
    name, _, raw_params = spec.partition(":")
    params = {}
    for pair in filter(None, raw_params.split(",")):
        key, _, value = pair.partition("=")
        params[key] = _parse_value(value)
    return get_model_class(name)(**params)


__all__ = [
    "MODELS",
    "IALSRecommender",
    "ItemKNNRecommender",
    "PopularityRecommender",
    "RandomRecommender",
    "RecentPopularityRecommender",
    "Recommender",
    "build_model",
    "get_model_class",
]
