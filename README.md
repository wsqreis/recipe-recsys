# recipe-recsys

A recipe recommender with **hard dietary constraints** (allergies and diets are never violated), **honest offline evaluation** and, in the next phases, learned embeddings, an LLM layer and a deployed demo.

Data: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions): 231k recipes and 1.1M reviews from 2000 to 2018.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, temporal split, baselines, metrics, dietary restriction filter | ✅ |
| 2 | Collaborative filtering with embeddings (iALS, two-tower) and ablations | ✅ |
| 3 | Cold start: new recipes ✅ and new users ✅; natural-language search with a local LLM ✅; measuring the restriction filter ✅; weekly meal planning | ⏳ |
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
docker compose run --rm recsys-gpu embed     # sentence embeddings of every recipe (~3 min on an RTX 4060)
docker compose run --rm recsys-gpu evaluate --stage val --slice warm new_item \
    --baseline popularity two_tower --models popularity ials text_profile two_tower two_tower:text=1,id_dropout=0.5
```

`data/` and `reports/` are mounted from the host, so downloads and results survive the container.

### Locally with uv

Requires [uv](https://docs.astral.sh/uv/). uv downloads Python 3.12 automatically. PyTorch is an optional extra: pick `cpu` or `cuda` (CUDA 12.6). Without it, every model except `two_tower` still works. The `text` extra (sentence-transformers) is needed for `recsys embed`; combine it with `cpu` or `cuda` so PyTorch comes from the right index.

```bash
uv sync --extra cpu --extra text            # or: uv sync --extra cuda --extra text
uv run recsys prepare                       # download Food.com and cache it as parquet (~1 min)
uv run recsys evaluate --stage val          # hyperparameter tuning
uv run recsys evaluate --stage test --save test   # final numbers, written to reports/test.md
uv run recsys evaluate --stage test --restrict vegetarian gluten
uv run recsys recommend --user 29196 --restrict vegetarian gluten
uv run pytest
```

To compare hyperparameters: `--models itemknn:neighbors=50,shrink=10.0 itemknn:neighbors=50,shrink=200.0`. Add `--baseline popularity` to get each model's paired difference to popularity with a 95% interval. Reports are only written with `--save NAME`, so ad hoc runs never overwrite them.

## Evaluation protocol

- **Implicit feedback:** every review counts as "cooked this recipe". On Food.com, a rating of 0 means a review without a rating, not a bad rating.
- **Global temporal split:** train until 2011-12, validation until 2014-02, test afterwards. A random split would let the model learn from reviews written *after* the ones it has to predict.
- **k-core (≥ 5 interactions per user and per recipe) applied to training data only.** Filtering with counts from the test period would leak which users stay active.
- **Tune on validation, run test once.** Hyperparameters were chosen on validation; the test set was run only with the final configurations.
- **Recipes already cooked in training are excluded** from both recommendations and ground truth: the goal is to recommend something new.
- **Cold start is reported, not hidden.** Interactions from users or recipes that do not exist in training cannot be served by collaborative filtering, and their share is printed in every evaluation.
- **Metrics:** Recall@K, NDCG@K, HitRate@K (binary relevance) and **catalog coverage**, the share of recipes recommended to at least one user.
- **Uncertainty is reported.** Every metric has a 95% bootstrap interval over users. Model comparisons use a **paired** bootstrap of the per-user difference (`--baseline`): each user is compared with themselves, which removes the user-to-user variance that makes the marginal intervals of two models overlap even when one is consistently better. A difference counts as real only if its interval excludes zero.
- **New recipes are a separate slice** (`--slice new_item`): known users rank only the recipes submitted during the evaluated window. Every recipe submitted in the window is a candidate, not only the ones that were cooked, which would leak the future.

## Models

| model | idea |
|---|---|
| `random`, `popularity`, `recent_popularity`, `newest` | Non-personalized baselines. Any real model has to beat them. `newest` (most recently submitted first) needs no interactions, so it can also rank new recipes. |
| `itemknn` | "People who cooked X also cooked Y": cosine similarity between recipes, pruned to the top-k neighbors, with shrinkage for pairs seen together only a few times. |
| `ials` | Implicit ALS matrix factorization (Hu, Koren & Volinsky, 2008, "Collaborative Filtering for Implicit Feedback Datasets") in plain numpy: a 32-dimensional embedding per user and recipe, with unobserved pairs as weak negatives and closed-form alternating updates. |
| `two_tower` | PyTorch retrieval model. The user tower pools the embeddings of the user's last 30 recipes, so a user with a few interactions gets a vector without retraining. The recipe tower combines an id embedding with ingredients, tags and nutrition. Trained with in-batch softmax and logQ correction (Yi et al., 2019, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"). Phase 3 options: `text=1` adds the recipe's sentence embedding, and `id_dropout=0.5` hides the recipe id in half of the training examples so the tower also works for recipes without a trained id. |
| `text_profile` | Content-based: the user is the mean sentence embedding (all-MiniLM-L6-v2 over name, ingredients and description) of their last 30 recipes; recipes are ranked by cosine similarity. |

## Results

Hyperparameters were tuned on validation (train → 2011-12, evaluated on 2011-12 → 2014-02). The final configurations were then retrained on train + validation and evaluated **once** on test (2014-02 → 2018-12): 16,873 users × 39,476 recipes in training (0.08% density), 2,388 users evaluated.

| model | NDCG@10 val | NDCG@10 test | vs. popularity (test) | paired Δ vs. popularity, 95% CI (test) | coverage@10 (test) |
|---|---|---|---|---|---|
| random | 0.0003 | 0.0007 | −95% | −0.0133 [−0.0163, −0.0105] | 45.5% |
| popularity | 0.0185 | 0.0140 | — | — | 0.1% |
| recent_popularity (180 days) | 0.0191 | 0.0121 | −14% | −0.0018 [−0.0042, +0.0005] | 0.1% |
| itemknn (k=50, shrink=10) | 0.0103 | 0.0129 | −8% | | **16.3%** |
| itemknn (k=50, shrink=50) | 0.0148 | 0.0145 | +4% | | 4.0% |
| itemknn (k=50, shrink=200) | 0.0181 | 0.0154 | +10% | +0.0014 [−0.0009, +0.0036] | 1.2% |
| **ials** (32 factors) | **0.0196** | **0.0162** | **+16%** | **+0.0023 [+0.0003, +0.0043]** | 1.0% |
| two_tower | 0.0192 | 0.0142 | +1% | +0.0001 [−0.0019, +0.0025] | 1.7% |
| two_tower, ablation: no content (ids only) | 0.0151 | 0.0124 | −11% | | 1.0% |
| two_tower, ablation: no logQ correction | 0.0010 | 0.0010 | −93% | | 27.0% |

Full tables (recall, hit rate, @20) in [reports/val.md](reports/val.md) and [reports/test.md](reports/test.md).

The paired intervals were added in phase 3 ([reports/test_paired.md](reports/test_paired.md)). That required a second run on test, made with the **same final configurations** and used only to measure uncertainty: no model or hyperparameter was chosen from it. It reproduced the original numbers (the two-tower differs in the 4th decimal: 0.0141 on GPU vs 0.0142 on CPU).

### What the numbers show

1. **The popularity baseline is hard to beat.** The best model, iALS, beats it by 16% on NDCG@10, and it is the **only** model whose paired interval on test excludes zero, by a small margin. On validation, not even iALS separates from popularity (+0.0011, 95% CI [−0.0008, +0.0028]). This is common on sparse data, and it is why no model is reported without baselines and intervals.
2. **The simplest learned model won.** iALS, a 2008 algorithm in about 40 lines of numpy, beat the two-tower network on test. On validation the two were tied (0.0196 vs 0.0192); on test the two-tower dropped to popularity level. It did not hold up over a 5-year horizon, while iALS did. Picking the model on test would have hidden this.
3. **Less capacity, more regularization.** Every iALS grid step towards fewer factors (128 → 32) and stronger regularization improved validation NDCG. For the two-tower, validation NDCG peaked after 3–5 epochs and then fell (0.0181 → 0.0124 at 10 epochs) while training loss kept dropping: with 0.08% density, extra capacity memorizes noise.
4. **Accuracy vs. diversity.** Popularity recommends the same ~40 recipes to everyone (0.1% of the catalog). Across models, the configurations with the best NDCG are the ones closest to popularity: in ItemKNN, a higher `shrink` improves NDCG but drops coverage from 16% to 1%, because shrinkage favors items with many co-occurrences. Choosing the operating point is a product decision, not just a metric.
5. **logQ correction is not optional.** Without it, the two-tower collapses to near-random (−93%). With in-batch negatives, popular recipes show up as negatives far more often than rare ones, so the model learns to push them down and recommends niche recipes (27% coverage). On a dataset where popularity carries this much signal, that destroys the ranking.
6. **Recipe content helps even when the id is available.** Removing ingredients, tags and nutrition from the two-tower costs 21% on validation and 13% on test. Content will matter even more in phase 3, where it is the only signal for new recipes.
7. **Recent popularity did worse than all-time popularity on test**, even though it was slightly better on validation. The 6-month trend signal did not hold over a 5-year test horizon.
8. **Cold start dominates.** On test, **79% of interactions come from users who do not exist in training**, and 10% from new recipes. Only 11% of interactions can be evaluated with collaborative filtering. This is the main motivation for phase 3: recommending from recipe *content* and stated preferences.

## New recipes (phase 3)

Collaborative models cannot score a recipe nobody has cooked yet. Phase 3 adds sentence embeddings of each recipe (`recsys embed`: all-MiniLM-L6-v2 over name, ingredients and description, 231k recipes in 157 s on an RTX 4060) and evaluates them on the `new_item` slice.

Validation, 719 known users ranking the 9,425 recipes submitted during the window ([reports/val_new_item.md](reports/val_new_item.md)):

| model | NDCG@10 [95% CI] | paired Δ vs. two_tower without text [95% CI] |
|---|---|---|
| random (3 seeds) | 0.0012 – 0.0022 | not significant for any seed |
| newest | 0.0000 | −0.0023 [−0.0035, −0.0012] |
| two_tower (ingredients, tags, nutrition; no text) | 0.0023 [0.0012, 0.0035] | — |
| two_tower + text | 0.0039 [0.0019, 0.0067] | +0.0015 [−0.0006, +0.0044] |
| two_tower + text + id dropout 0.5 | **0.0051** [0.0024, 0.0089] | +0.0028 [−0.0000, +0.0065] |
| text_profile | **0.0051** [0.0026, 0.0082] | **+0.0028 [+0.0001, +0.0059]** |

What this shows:

1. **Structured content alone is no better than random on new recipes.** The two-tower without text sits inside the range of three random seeds.
2. **Text roughly doubles NDCG on new recipes, but the evidence is weak.** With 719 users, only `text_profile` clears zero, by 0.0001. This is a direction worth pursuing, not a solved problem.
3. **The simple model ties the neural one again.** A 10-line mean of sentence embeddings matches the two-tower with text and id dropout. Id dropout is what makes the two-tower usable here: without it, the tower never learns to work without the id.
4. **Text costs accuracy on known recipes.** On the warm slice the two-tower drops from 0.0193 to 0.0177 with text (same GPU run), and `text_profile` alone is near random (0.0008): pure similarity recommends near-duplicates of what the user already cooked (mean cosine 0.69 to their history, vs 0.57 for what they actually cooked next). A production system would be hybrid: collaborative scores for known recipes, content for new ones.
5. **Embeddings capture topic, not safety.** Among the nearest neighbors of "gluten free banana bread" is "gluten free banana muffins", whose ingredient list includes all-purpose flour: the text says gluten-free, the ingredients do not, and the embedding follows the text. Dietary restrictions stay a rule-based hard filter over ingredients.
6. **The first slice design measured nothing.** It used every recipe outside the training catalog as a candidate (194k, mostly long-tail recipes removed by the k-core), and every model, random included, got a hit for 2 to 6 of 2,842 users. It was replaced by the new-recipe slice; the confidence intervals are what made the problem visible.

These models have not been run on test yet: their configuration is not final.

## New users (phase 3)

Most evaluation interactions come from users who do not exist in training. A user with no history can only get non-personalized recommendations, so the `new_user` slice reproduces the first moment personalization is possible: the user has cooked their first N recipes (`--seed-recipes N`), and the models rank what they cook next. Every model can score a history it was not trained on (`score_histories`): iALS folds the user in with the same closed-form update it uses in training, ItemKNN sums item similarities, the two-tower runs its user tower, and no model is retrained.

Validation, paired difference in NDCG@10 against popularity ([seed 1](reports/val_new_user_seed1.md), [seed 3](reports/val_new_user_seed3.md), [seed 5](reports/val_new_user_seed5.md)):

| history | new users | popularity NDCG@10 | itemknn | ials | two_tower |
|---|---|---|---|---|---|
| 1 recipe | 2,826 | 0.0254 | +14% [−0.0012, +0.0085] | +1% [−0.0027, +0.0035] | −4% [−0.0035, +0.0012] |
| 3 recipes | 678 | 0.0252 | −2% [−0.0076, +0.0062] | +9% [−0.0036, +0.0081] | −1% [−0.0061, +0.0053] |
| 5 recipes | 282 | 0.0287 | +8% [−0.0050, +0.0104] | +16% [−0.0032, +0.0122] | −6% [−0.0087, +0.0061] |

What this shows:

1. **After one recipe, nothing beats popularity.** One data point says little about taste. `text_profile` is significantly worse (−61%): similarity to a single recipe recommends near-duplicates of it.
2. **iALS drifts away from popularity as the history grows** (+1% → +9% → +16%), but no difference is significant: the sample shrinks as fast as the signal grows.
3. **Most new users never build a history.** Of the 2,826 new users with at least 2 catalog recipes in the window, only 678 reach 4 and 282 reach 6. For most users, popularity filtered by their restrictions is the right default; the lever is product (ask preferences at sign-up), not a better model.

## Natural-language search (phase 3)

```bash
# Ollama must be running on the host with the model pulled: `ollama pull ministral-3:8b`
docker compose run --rm recsys search "something dairy-free with chicken, ready in 20 minutes"
```

```
parsed: {"restrictions": ["lactose"], "include": ["chicken"], "exclude": [], "max_minutes": 20, "query": "quick dairy-free chicken dish"}
   1. dairy free cream of chicken  [20 min]  (chicken broth, almond milk, flour, salt, pepper, parsley, ...)
   2. quick   easy chicken in wine sauce  [17 min]  (boneless skinless chicken breasts, olive oil, green onions, garlic, ...)
```

Requests can also be written in other languages: the LLM translates them, and the evaluation sets below mix English and Portuguese requests on purpose.

How it works:

1. **A local LLM parses the request, it does not decide.** `ministral-3:8b` (6 GB, fits an 8 GB GPU) runs in [Ollama](https://ollama.com) and turns the request, in any language, into a JSON query constrained by a schema: restrictions, required and excluded ingredients, time limit, and an English description of the dish. ~1.2 s per request.
2. **Retrieval** ranks all 231k recipes by cosine similarity between the description and the recipe text embeddings.
3. **Every constraint is a hard filter applied after ranking**, with the same rule-based restriction filter as the recommender. The LLM never judges whether a recipe is safe.
4. **Two safety nets around the LLM.** A keyword detector runs on the raw request and its restrictions are added to the LLM's. And an excluded term that names a whole category ("meat", "dairy", "eggs") becomes the restriction, because a word filter on "meat" still lets chicken and fish through.

Evaluated on hand-labeled requests in Portuguese and English (`recsys eval-search`):

| | development set, v1 (42 requests) | held-out set, v1 → v2 (30 requests) |
|---|---|---|
| restriction recall, LLM + keywords | 97% (keywords alone: 84%) | 100% → 100% (keywords alone: 67%) |
| restriction precision | 100% | 88% → **95%** |
| results violating a restriction the user stated | **8 of 399** | 0 of 299 → 0 of 300 |
| time limit parsed correctly | 100% | 100% → 100% |
| required ingredients parsed correctly | 86% | 87% → 87% |

Reports: [search_v1](reports/search_v1.md), [search_v1_holdout](reports/search_v1_holdout.md), [search_v2_holdout](reports/search_v2_holdout.md).

What this shows:

1. **The LLM understands what keywords cannot.** "I don't eat animal products", "nada de laticínios" (Portuguese for "no dairy at all") and "my son can't have milk" have no fixed phrase to match. Keywords alone missed a third of the held-out restrictions.
2. **The dangerous failure looks reasonable.** All 8 unsafe results in v1 came from one request, "algo leve, não como carne" (I don't eat meat), parsed as "exclude the ingredient *meat*". Chicken passed the filter. v2's category rule closes this without relying on the LLM getting it right.
3. **The LLM also errs the other way.** On the held-out set, v1 turned "no butter" into lactose intolerance, "no pork" into vegetarian and "omelet without onion" into egg-free. Those are safe but hide what the user asked for. A prompt rule ("leaving out one ingredient is not a restriction") cut them from 3 to 1.
4. **Measured honestly.** The v2 changes came from reading v1's errors on the development set, so v2 numbers on that set (100% recall, 0 violations, [search_v2](reports/search_v2.md)) are optimistic. The held-out set was written and committed before changing the parser, and is the number to trust. It is small (30 requests), so it shows the direction, not a guarantee.
5. **Required ingredients are ambiguous, not unsafe.** Is "carrot" required in "bolo de cenoura" (carrot cake), or just the dish name? Both readings are defensible, and the label itself is the problem. This affects how narrow the search is, not safety.

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

**Important limitation:** the "violations=0" check in the evaluation uses the same filter to audit itself, so it is a consistency check, not a proof of safety. Phase 3 measured the filter against independent labels (next section): it lets through about 3% of unsafe (recipe, restriction) pairs.

### Measuring the filter (phase 3)

Two random samples of 150 recipes were labeled forbidden / allowed / uncertain for each restriction, following a [written policy](evaluation/restriction_labeling.md) (judge each ingredient by its typical US commercial form). **The labels were made by an LLM (Claude), from the ingredient lists, with no human review**, before running the filter or the classifier on them. The second sample was labeled and committed before any change to the filter. Uncertain pairs (e.g. "tortillas" for gluten: corn or wheat?) are excluded. The positive class is "forbidden", so recall is the safety metric.

| classifier | held-out sample: recall (safety) | precision | unsafe pairs missed |
|---|---|---|---|
| keywords, v1 | 97.4% [95.8%, 98.9%] | 100% | 12 of 457 |
| keywords, v2 | 97.4% [95.8%, 98.9%] | 100% | 12 of 457 |
| LLM classifier (ministral-3:8b) | 97.4% [95.8%, 98.9%] | 83.3% | 12 of 457 |
| **keywords + LLM** (forbidden if either says so) | **100%** [100%, 100%] | 83.7% | **0 of 457** |

`recsys eval-restrictions`; reports: [v1 on the first sample](reports/restrictions_v1.md), [v1 held-out](reports/restrictions_v1_holdout.md), [v2 held-out](reports/restrictions_v2_holdout.md).

What this shows:

1. **Fixing what you have seen does not generalize.** v2 adds every word the filter missed on the first sample (`hamburger`, singular `breadcrumb`, `sherbet`, `ravioli`, `ranch dressing`, ...). On that sample, recall jumps from 95.6% to 99.8% ([report](reports/restrictions_v2.md)). On the held-out sample it does not move: its misses are other processed products (caesar dressing, marshmallows, Jell-O, pumpernickel, corn flakes with malt, corn muffin mix with lard). The vocabulary of processed foods has a long tail. Measured only on the first sample, the filter would have been reported at 99.8%.
2. **The 8B LLM is not a better filter.** It has the same recall as the keywords but much worse precision: on the held-out sample it blocks 54 of the 78 vegetarian recipes, apparently treating dairy and eggs as meat. On the first sample it also missed trivial cases (a beef steak as vegan, pancakes with flour as gluten-free).
3. **They fail in different places, so the union works.** Keywords know the obvious ingredients and miss processed products; the LLM knows processed products and misses the obvious. Together: 0 unsafe pairs missed on the held-out sample, at the cost of hiding about one in five safe pairs. For a severe allergy, that trade is defensible; the union is not wired into the recommender yet.
4. **Limits.** 457 forbidden pairs is a small sample (the 100% has a real failure rate above zero), and the labels come from a model following a policy, not from a dietitian or from product labels.

## Project layout

```
src/recipe_recsys/
  data.py           Food.com download and parsing → parquet
  split.py          temporal split and k-core
  dataset.py        sparse user × recipe matrix and id mappings
  restrictions.py   dietary restrictions (hard filter)
  text.py           sentence embeddings of recipe text (cached)
  search.py         natural-language search: LLM parse (Ollama) + retrieval + hard filters
  search_eval.py    parsing and safety metrics on labeled requests (evaluation/)
  restriction_eval.py  restriction filter vs labeled recipes, with an LLM classifier
  metrics.py        recall, ndcg, hit rate, coverage
  evaluate.py       evaluation protocol, slices, masking, top-k, bootstrap intervals
  models/           random, popularity, recent_popularity, newest, itemknn, ials,
                    text_profile, two_tower
  cli.py            recsys prepare | embed | evaluate | recommend | search | eval-search
                    | eval-restrictions
tests/              metrics, leak-free split, restrictions and regressions
reports/            evaluation results (.md versioned, .json ignored)
```

Every model implements the same interface (`fit` and `score`, plus `score_histories` for users outside training and `score_items` for content-aware models that can rank recipes outside the training catalog). Masking (already-seen items and restrictions) and top-k selection live outside the models, so every model gets exactly the same treatment during evaluation.

An engineering note: ItemKNN's item × item co-occurrence matrix has about 10⁸ non-zeros. It is built one block of rows at a time and pruned to the top-k neighbors before the next block, which keeps peak memory at about 1.5 GB.
