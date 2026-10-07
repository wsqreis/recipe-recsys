from recipe_recsys.models.base import Recommender
from recipe_recsys.models.baselines import (
    PopularityRecommender,
    RandomRecommender,
    RecentPopularityRecommender,
)
from recipe_recsys.models.itemknn import ItemKNNRecommender

MODELS: dict[str, type[Recommender]] = {
    cls.name: cls
    for cls in [
        RandomRecommender,
        PopularityRecommender,
        RecentPopularityRecommender,
        ItemKNNRecommender,
    ]
}


def build_model(spec: str) -> Recommender:
    """Build from `name` or `name:param=value,param=value`, e.g. `itemknn:neighbors=50`."""
    name, _, raw_params = spec.partition(":")
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; options: {sorted(MODELS)}")
    params = {}
    for pair in filter(None, raw_params.split(",")):
        key, _, value = pair.partition("=")
        params[key] = float(value) if "." in value else int(value)
    return MODELS[name](**params)


__all__ = ["MODELS", "Recommender", "build_model"]
