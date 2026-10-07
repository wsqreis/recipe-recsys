# recipe-recsys

A recipe recommender with **hard dietary constraints** (allergies and diets are never violated), **honest offline evaluation** and, in the next phases, learned embeddings, an LLM layer and a deployed demo.

Data: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions): 231k recipes and 1.1M reviews from 2000 to 2018.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, temporal split, baselines, metrics, dietary restriction filter | ✅ |
| 2 | Collaborative filtering with embeddings (matrix factorization, two-tower) | ⏳ |
| 3 | Content embeddings for cold start, natural-language search with an LLM, weekly meal planning | ⏳ |
| 4 | API (FastAPI), vector index, Docker, deployment, demo | ⏳ |

## Running

Requires [uv](https://docs.astral.sh/uv/). uv downloads Python 3.12 automatically.

```bash
uv sync
uv run recsys prepare                       # download Food.com and cache it as parquet (~1 min)
uv run recsys evaluate --stage val          # hyperparameter tuning
uv run recsys evaluate --stage test         # final numbers
uv run recsys evaluate --stage test --restrict vegetarian gluten
uv run recsys recommend --user 29196 --restrict vegetarian gluten
uv run pytest
```

To compare hyperparameters: `--models itemknn:neighbors=50,shrink=10.0 itemknn:neighbors=50,shrink=200.0`.

## Evaluation protocol

- **Implicit feedback:** every review counts as "cooked this recipe". On Food.com, a rating of 0 means a review without a rating, not a bad rating.
- **Global temporal split:** train until 2011-12, validation until 2014-02, test afterwards. A random split would let the model learn from reviews written *after* the ones it has to predict.
- **k-core (≥ 5 interactions per user and per recipe) applied to training data only.** Filtering with counts from the test period would leak which users stay active.
- **Tune on validation, run test once.** Hyperparameters were chosen on validation; the test set was run only with the final configurations.
- **Recipes already cooked in training are excluded** from both recommendations and ground truth: the goal is to recommend something new.
- **Cold start is reported, not hidden.** Interactions from users or recipes that do not exist in training cannot be served by collaborative filtering, and their share is printed in every evaluation.
- **Metrics:** Recall@K, NDCG@K, HitRate@K (binary relevance) and **catalog coverage**, the share of recipes recommended to at least one user.

## Results (test, 2014-02 → 2018-12)

Training data: 16,873 users × 39,476 recipes (0.08% density). 2,388 users evaluated.

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 |
|---|---|---|---|---|
| random | 0.0007 | 0.0007 | 0.0029 | 45.5% |
| popularity | 0.0226 | 0.0140 | 0.0582 | 0.1% |
| recent_popularity (180 days) | 0.0177 | 0.0121 | 0.0523 | 0.1% |
| itemknn (k=50, shrink=10) | 0.0171 | 0.0129 | 0.0486 | **16.3%** |
| itemknn (k=50, shrink=50) | 0.0218 | 0.0145 | 0.0540 | 4.0% |
| itemknn (k=50, shrink=200) | 0.0220 | **0.0154** | **0.0620** | 1.2% |

Full tables (including @20) in [reports/](reports/).

### What the numbers show

1. **The popularity baseline is hard to beat.** The most accurate ItemKNN beats it by only 10% on NDCG@10. This is common on sparse data, and it is why no model is reported without baselines.
2. **Accuracy vs. diversity.** Popularity recommends the same ~40 recipes to everyone (0.1% of the catalog). In ItemKNN, a higher `shrink` improves NDCG but drops coverage from 16% to 1%: the model gradually *becomes* a popularity recommender, because shrinkage favors items with many co-occurrences. Choosing the operating point is a product decision, not just a metric.
3. **Recent popularity did worse than all-time popularity on test**, even though it was slightly better on validation. The 6-month trend signal did not hold over a 5-year test horizon.
4. **Cold start dominates.** On test, **79% of interactions come from users who do not exist in training**, and 10% from new recipes. Only 11% of interactions can be evaluated with collaborative filtering. This is the main motivation for phase 3: recommending from recipe *content* and stated preferences.

## Dietary restrictions

Restrictions are a **hard filter applied after ranking**, not a score penalty: a forbidden recipe never shows up, however high it scores.

Available restrictions: `vegetarian`, `vegan`, `lactose`, `egg`, `gluten`, `nuts`.

| restriction (test) | allowed catalog | NDCG@10 popularity | NDCG@10 itemknn |
|---|---|---|---|
| vegetarian | 57.5% | 0.0152 | 0.0141 |
| gluten | 45.2% | 0.0205 | 0.0176 |
| lactose + nuts | 33.2% | 0.0248 | 0.0207 |

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
  models/           random, popularity, recent_popularity, itemknn
  cli.py            recsys prepare | evaluate | recommend
tests/              metrics, leak-free split, restrictions and regressions
reports/            evaluation results (.md versioned, .json ignored)
```

Every model implements the same interface (`fit` and `score`). Masking (already-seen items and restrictions) and top-k selection live outside the models, so every model gets exactly the same treatment during evaluation.

An engineering note: ItemKNN's item × item co-occurrence matrix has about 10⁸ non-zeros. It is built one block of rows at a time and pruned to the top-k neighbors before the next block, which keeps peak memory at about 1.5 GB.
