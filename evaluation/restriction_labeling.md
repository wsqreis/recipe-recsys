# Labeling policy for the restriction filter

Written before drawing or reading the sample, so that labels follow fixed
rules instead of whatever the filter happens to do.

**Who labeled:** Claude (an LLM, via Claude Code), from the ingredient list
and recipe name only, without running the filter or the LLM classifier first.
No person reviewed the labels. Treat the results as "agreement with a strong
model following this policy", not as ground truth from a dietitian.

## Unit and classes

One label per (recipe, restriction), for the six restrictions: `vegetarian`,
`vegan`, `lactose`, `egg`, `gluten`, `nuts`.

- **forbidden**: at least one ingredient, in its typical form, is not
  compatible with the restriction. This is the positive class: the filter's job
  is to catch it.
- **allowed**: no ingredient is incompatible.
- **uncertain**: compatibility depends on a product choice the recipe does not
  specify, and both versions are common (e.g. "chocolate chips" for lactose:
  semisweet chips are often dairy-free, milk chocolate chips are not). Uncertain
  labels are reported separately and excluded from precision and recall.

## Rules

1. **Typical commercial form.** Judge an ingredient by its most common
   commercial version in the US (the dataset is US-centric). Margarine usually
   contains whey or buttermilk → forbidden for lactose and vegan. Worcestershire
   sauce contains anchovies → forbidden for vegetarian and vegan; it usually
   contains malt vinegar → forbidden for gluten. Regular soy sauce contains wheat.
2. **Explicit free-from labels are trusted.** "Gluten-free flour", "vegan
   butter", "dairy-free chocolate chips" are taken at face value.
3. **Restriction meanings.**
   - vegetarian: no meat, poultry, fish, seafood, or ingredients made from them
     (gelatin, meat/fish broths and stocks, lard, anchovy-based sauces). Dairy,
     eggs and honey are fine.
   - vegan: vegetarian, and no dairy, eggs, honey or other animal products.
   - lactose: no dairy (milk, butter, cream, cheese, yogurt, whey, casein,
     ghee). Strictly this is "dairy-free", which is how the filter defines it.
   - egg: no eggs or egg-based products (mayonnaise, meringue, egg pasta when
     stated).
   - gluten: no wheat, barley, rye, malt, or products made from them (bread,
     regular pasta, flour, beer, regular oats are treated as forbidden because
     of cross-contamination, regular soy sauce).
   - nuts: no tree nuts or peanuts, or products made from them (peanut butter,
     almond milk, nut oils, pesto with pine nuts). Coconut and nutmeg are
     allowed.
4. **Unnamed products are judged by name.** "Cake mix", "cream of mushroom
   soup", "crescent rolls": use the typical product (wheat flour, dairy).
   "Bouillon cube" without a flavor: uncertain for vegetarian.
5. **Ignore quantities and "optional".** An optional ingredient is still in the
   list, so it counts.
