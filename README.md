# recipe-recsys

A recipe recommender with **hard dietary constraints** (allergies and diets are never violated), **honest offline evaluation** and, in the next phases, learned embeddings, an LLM layer and a deployed demo.

Data: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions): 231k recipes and 1.1M reviews from 2000 to 2018.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, temporal split, baselines, metrics, dietary restriction filter | ✅ |
| 2 | Collaborative filtering with embeddings (iALS, two-tower) and ablations | ✅ |
| 3 | Content embeddings for cold start, natural-language search with an LLM, weekly meal planning | ⏳ |
| 4 | API (FastAPI), vector index, deployment, demo (Docker image already available) | ⏳ |

## Running

### With Docker (any OS)

```bash
docker compose build recsys
docker compose run --rm recsys prepare                  # download Food.com into ./data (~1 min)
docker compose run --rm recsys evaluate --stage test --save test
docker compose run --rm recsys recommend --user 29196 --restrict vegetarian gluten
docker compose run --rm --entrypoint pytest recsys -q
```

With an NVIDIA GPU (Linux with the NVIDIA Container Toolkit, or Windows with Docker Desktop on WSL2), use the `recsys-gpu` service. It installs the CUDA build of PyTorch, and the two-tower model picks up the GPU automatically:

```bash
docker compose --profile gpu build recsys-gpu
docker compose run --rm recsys-gpu evaluate --stage val --models two_tower
```

`data/` and `reports/` are mounted from the host, so downloads and results survive the container.

### Locally with uv

Requires [uv](https://docs.astral.sh/uv/). uv downloads Python 3.12 automatically. PyTorch is an optional extra: pick `cpu` or `cuda` (CUDA 12.6). Without it, every model except `two_tower` still works.

```bash
uv sync --extra cpu                         # or: uv sync --extra cuda
uv run recsys prepare                       # download Food.com and cache it as parquet (~1 min)
uv run recsys evaluate --stage val          # hyperparameter tuning
uv run recsys evaluate --stage test --save test   # final numbers, written to reports/test.md
uv run recsys evaluate --stage test --restrict vegetarian gluten
uv run recsys recommend --user 29196 --restrict vegetarian gluten
uv run pytest
```

To compare hyperparameters: `--models itemknn:neighbors=50,shrink=10.0 itemknn:neighbors=50,shrink=200.0`. Reports are only written with `--save NAME`, so ad hoc runs never overwrite them.

## Evaluation protocol

- **Implicit feedback:** every review counts as "cooked this recipe". On Food.com, a rating of 0 means a review without a rating, not a bad rating.
- **Global temporal split:** train until 2011-12, validation until 2014-02, test afterwards. A random split would let the model learn from reviews written *after* the ones it has to predict.
- **k-core (≥ 5 interactions per user and per recipe) applied to training data only.** Filtering with counts from the test period would leak which users stay active.
- **Tune on validation, run test once.** Hyperparameters were chosen on validation; the test set was run only with the final configurations.
- **Recipes already cooked in training are excluded** from both recommendations and ground truth: the goal is to recommend something new.
- **Cold start is reported, not hidden.** Interactions from users or recipes that do not exist in training cannot be served by collaborative filtering, and their share is printed in every evaluation.
- **Metrics:** Recall@K, NDCG@K, HitRate@K (binary relevance) and **catalog coverage**, the share of recipes recommended to at least one user.

## Models

| model | idea |
|---|---|
| `random`, `popularity`, `recent_popularity` | Non-personalized baselines. Any real model has to beat them. |
| `itemknn` | "People who cooked X also cooked Y": cosine similarity between recipes, pruned to the top-k neighbors, with shrinkage for pairs seen together only a few times. |
| `ials` | Implicit ALS matrix factorization (Hu, Koren & Volinsky, 2008, "Collaborative Filtering for Implicit Feedback Datasets") in plain numpy: a 32-dimensional embedding per user and recipe, with unobserved pairs as weak negatives and closed-form alternating updates. |
| `two_tower` | PyTorch retrieval model. The user tower pools the embeddings of the user's last 30 recipes, so a user with a few interactions gets a vector without retraining. The recipe tower combines an id embedding with ingredients, tags and nutrition. Trained with in-batch softmax and logQ correction (Yi et al., 2019, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"). |

## Results

Hyperparameters were tuned on validation (train → 2011-12, evaluated on 2011-12 → 2014-02). The final configurations were then retrained on train + validation and evaluated **once** on test (2014-02 → 2018-12): 16,873 users × 39,476 recipes in training (0.08% density), 2,388 users evaluated.

| model | NDCG@10 val | NDCG@10 test | vs. popularity (test) | coverage@10 (test) |
|---|---|---|---|---|
| random | 0.0003 | 0.0007 | −95% | 45.5% |
| popularity | 0.0185 | 0.0140 | — | 0.1% |
| recent_popularity (180 days) | 0.0191 | 0.0121 | −14% | 0.1% |
| itemknn (k=50, shrink=10) | 0.0103 | 0.0129 | −8% | **16.3%** |
| itemknn (k=50, shrink=50) | 0.0148 | 0.0145 | +4% | 4.0% |
| itemknn (k=50, shrink=200) | 0.0181 | 0.0154 | +10% | 1.2% |
| **ials** (32 factors) | **0.0196** | **0.0162** | **+16%** | 1.0% |
| two_tower | 0.0192 | 0.0142 | +1% | 1.7% |
| two_tower, ablation: no content (ids only) | 0.0151 | 0.0124 | −11% | 1.0% |
| two_tower, ablation: no logQ correction | 0.0010 | 0.0010 | −93% | 27.0% |

Full tables (recall, hit rate, @20) in [reports/val.md](reports/val.md) and [reports/test.md](reports/test.md).

### What the numbers show

1. **The popularity baseline is hard to beat.** The best model, iALS, beats it by 16% on NDCG@10. This is common on sparse data, and it is why no model is reported without baselines.
2. **The simplest learned model won.** iALS, a 2008 algorithm in about 40 lines of numpy, beat the two-tower network on test. On validation the two were tied (0.0196 vs 0.0192); on test the two-tower dropped to popularity level. It did not hold up over a 5-year horizon, while iALS did. Picking the model on test would have hidden this.
3. **Less capacity, more regularization.** Every iALS grid step towards fewer factors (128 → 32) and stronger regularization improved validation NDCG. For the two-tower, validation NDCG peaked after 3–5 epochs and then fell (0.0181 → 0.0124 at 10 epochs) while training loss kept dropping: with 0.08% density, extra capacity memorizes noise.
4. **Accuracy vs. diversity.** Popularity recommends the same ~40 recipes to everyone (0.1% of the catalog). Across models, the configurations with the best NDCG are the ones closest to popularity: in ItemKNN, a higher `shrink` improves NDCG but drops coverage from 16% to 1%, because shrinkage favors items with many co-occurrences. Choosing the operating point is a product decision, not just a metric.
5. **logQ correction is not optional.** Without it, the two-tower collapses to near-random (−93%). With in-batch negatives, popular recipes show up as negatives far more often than rare ones, so the model learns to push them down and recommends niche recipes (27% coverage). On a dataset where popularity carries this much signal, that destroys the ranking.
6. **Recipe content helps even when the id is available.** Removing ingredients, tags and nutrition from the two-tower costs 21% on validation and 13% on test. Content will matter even more in phase 3, where it is the only signal for new recipes.
7. **Recent popularity did worse than all-time popularity on test**, even though it was slightly better on validation. The 6-month trend signal did not hold over a 5-year test horizon.
8. **Cold start dominates.** On test, **79% of interactions come from users who do not exist in training**, and 10% from new recipes. Only 11% of interactions can be evaluated with collaborative filtering. This is the main motivation for phase 3: recommending from recipe *content* and stated preferences.

## Dietary restrictions

Restrictions are a **hard filter applied after ranking**, not a score penalty: a forbidden recipe never shows up, however high it scores.

Available restrictions: `vegetarian`, `vegan`, `lactose`, `egg`, `gluten`, `nuts`.

| restriction (test) | allowed catalog | popularity | itemknn | ials | two_tower |
|---|---|---|---|---|---|
| vegetarian | 57.5% | 0.0152 | 0.0141 | 0.0150 | **0.0158** |
| gluten | 45.2% | 0.0205 | 0.0176 | **0.0242** | 0.0215 |
| lactose + nuts | 33.2% | 0.0248 | 0.0207 | **0.0281** | 0.0211 |

NDCG@10 on test; every model had **0 restriction violations**. The ranking changes with the restriction: the two-tower is best for vegetarians and iALS for the other two.

Design decisions:

- **Ingredients, not tags.** Food.com tags are user-provided and unreliable: some recipes tagged `gluten-free` use worcestershire sauce, which usually contains barley malt.
- **Conservative by default.** Hiding a safe recipe (false positive) is cheap; showing an unsafe one (false negative) is not. Known safe phrases such as "coconut milk" and "nutmeg" are whitelisted explicitly instead of loosening the rules.
- **Markers only apply to their own restriction.** "Gluten-free almond flour" is allowed for gluten and still blocked for nuts. "Non-dairy" is **not** a marker: in the US, a "non-dairy" product may contain caseinate (milk protein).
- **Audited against the dataset itself.** Listing the most frequent ingredients each filter allows revealed real failures in the first version: `baguette`, `hamburger buns` and `cream of mushroom soup` passed the gluten filter, and `crabmeat` and `catfish` passed the vegetarian one (compound words escape word boundaries). Each failure became a regression test in [tests/test_restrictions.py](tests/test_restrictions.py).

**Important limitation:** the "violations=0" check in the evaluation uses the same filter to audit itself, so it is a consistency check, not a proof of safety. Keywords cannot see ingredients hidden inside processed products. A real product would need curated ingredient data and a labeled set to measure the filter's precision and recall. This is planned for phase 3, compared against an LLM-based classifier.

## Project layout

```
src/recipe_recsys/
  data.py           Food.com download and parsing → parquet
  split.py          temporal split and k-core
  dataset.py        sparse user × recipe matrix and id mappings
  restrictions.py   dietary restrictions (hard filter)
  metrics.py        recall, ndcg, hit rate, coverage
  evaluate.py       evaluation protocol, masking and top-k
  models/           random, popularity, recent_popularity, itemknn, ials, two_tower
  cli.py            recsys prepare | evaluate | recommend
tests/              metrics, leak-free split, restrictions and regressions
reports/            evaluation results (.md versioned, .json ignored)
```

Every model implements the same interface (`fit` and `score`). Masking (already-seen items and restrictions) and top-k selection live outside the models, so every model gets exactly the same treatment during evaluation.

An engineering note: ItemKNN's item × item co-occurrence matrix has about 10⁸ non-zeros. It is built one block of rows at a time and pruned to the top-k neighbors before the next block, which keeps peak memory at about 1.5 GB.
