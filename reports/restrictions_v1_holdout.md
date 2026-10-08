150 random recipes, labeled forbidden / allowed / uncertain per restriction (uncertain pairs excluded). Positive class: forbidden.

| classifier | precision | recall (safety) | recall 95% CI | unsafe recipes missed |
|---|---|---|---|---|
| keywords (current filter) | 100.0% | 97.4% | [95.8%, 98.9%] | 12 of 457 |
| LLM (ministral-3:8b) | 83.3% | 97.4% | [95.8%, 98.9%] | 12 of 457 |
| keywords + LLM | 83.7% | 100.0% | [100.0%, 100.0%] | 0 of 457 |

Recall per restriction (forbidden pairs caught / forbidden pairs):

| classifier | egg | gluten | lactose | nuts | vegan | vegetarian |
|---|---|---|---|---|---|---|
| keywords (current filter) | 48/51 | 85/88 | 102/103 | 15/15 | 129/130 | 66/70 |
| LLM (ministral-3:8b) | 48/51 | 85/88 | 103/103 | 15/15 | 127/130 | 67/70 |
| keywords + LLM | 51/51 | 88/88 | 103/103 | 15/15 | 130/130 | 70/70 |

Precision per restriction (correct blocks / blocks):

| classifier | egg | gluten | lactose | nuts | vegan | vegetarian |
|---|---|---|---|---|---|---|
| keywords (current filter) | 48/48 | 85/85 | 102/102 | 15/15 | 129/129 | 66/66 |
| LLM (ministral-3:8b) | 48/70 | 85/91 | 103/107 | 15/18 | 127/127 | 67/121 |
| keywords + LLM | 51/73 | 88/94 | 103/107 | 15/18 | 130/130 | 70/124 |

Unsafe recipes missed by keywords (current filter):

- egg: caesar green beans (water; green beans; caesar salad dressing; dried cranberries)
- egg: rag alla bolognese   the authentic recipe (beef; pancetta; carrots; celery; onions; tomato concentrate; red wine; milk; salt; pepper; tagliatelle pasta noodles)
- egg: the brittany (marshmallows; creamy peanut butter; bread; butter; marshmallow cream)
- gluten: flounder fillet in herb sauce (celery ribs; onions; butter; dill; tarragon; sour cream; condensed mushroom soup; parsley; dijon mustard; salt and pepper; flounder fillets)
- gluten: eggstatic egg salad (eggs; celery; fresh chives; fresh italian parsley; coarse black pepper; balsamic vinegar; mayonnaise; pumpernickel rounds)
- gluten: magic cookie bars a grandma brown special (corn flakes; white sugar; butter; chocolate chips; white chocolate chips; coconut; pecans; condensed milk)
- lactose: caesar green beans (water; green beans; caesar salad dressing; dried cranberries)
- vegan: caesar green beans (water; green beans; caesar salad dressing; dried cranberries)
- vegetarian: caesar green beans (water; green beans; caesar salad dressing; dried cranberries)
- vegetarian: the brittany (marshmallows; creamy peanut butter; bread; butter; marshmallow cream)
- vegetarian: baked corn  kid friendly (whole kernel corn; instant minced onion; garlic powder; butter; corn muffin mix; milk)
- vegetarian: creamsicle jello (boiling water; orange jell-o; orange juice; vanilla ice cream)

Unsafe recipes missed by LLM (ministral-3:8b):

- egg: vegetable cheese souffle (butter; flour; salt; cayenne pepper; milk; sharp cheddar cheese; egg yolks; sliced mushrooms; red bell peppers; broccoli; olive oil; egg whites; confectioners' sugar)
- egg: no pudge brownie clone (sugar; flour; cocoa; egg whites; cornstarch; baking soda; salt; non-fat vanilla yogurt; nonstick cooking spray)
- egg: 7 up pound cake (butter; sugar; eggs; cake flour; lemon flavoring; carbonated lemon-lime beverage)
- gluten: greek penne and chicken (penne pasta; butter; red onion; garlic cloves; skinless chicken breast half; water-packed artichoke hearts; tomatoes; feta cheese; fresh parsley; lemon juice; dried oregano; salt; ground black pepper)
- gluten: cheesy garlic pull apart bread (pillsbury refrigerated biscuits; garlic cloves; butter; mozzarella cheese)
- gluten: no pudge brownie clone (sugar; flour; cocoa; egg whites; cornstarch; baking soda; salt; non-fat vanilla yogurt; nonstick cooking spray)
- vegan: lemon pepper grilled salmon (lemon-pepper seasoning; extra virgin olive oil; fresh garlic cloves; salmon fillets; lemon; fresh parsley)
- vegan: clams with tomato and basil (extra virgin olive oil; garlic cloves; crushed red pepper flakes; littleneck clams; dry white wine; tomato sauce; scallions; fresh basil leaves; salt)
- vegan: cuban ropa vieja   crock pot  oamc (beef flank steak; vegetable oil; beef broth; tomato sauce; tomato paste; onion; green bell pepper; garlic cloves; fresh cilantro; olive oil; vinegar)
- vegetarian: lemon pepper grilled salmon (lemon-pepper seasoning; extra virgin olive oil; fresh garlic cloves; salmon fillets; lemon; fresh parsley)
- vegetarian: clams with tomato and basil (extra virgin olive oil; garlic cloves; crushed red pepper flakes; littleneck clams; dry white wine; tomato sauce; scallions; fresh basil leaves; salt)
- vegetarian: cuban ropa vieja   crock pot  oamc (beef flank steak; vegetable oil; beef broth; tomato sauce; tomato paste; onion; green bell pepper; garlic cloves; fresh cilantro; olive oil; vinegar)
