150 random recipes, labeled forbidden / allowed / uncertain per restriction (uncertain pairs excluded). Positive class: forbidden.

| classifier | precision | recall (safety) | recall 95% CI | unsafe recipes missed |
|---|---|---|---|---|
| keywords (current filter) | 99.5% | 95.6% | [93.6%, 97.4%] | 20 of 455 |
| LLM (ministral-3:8b) | 84.4% | 94.9% | [93.0%, 96.9%] | 23 of 455 |
| keywords + LLM | 84.8% | 98.5% | [97.4%, 99.6%] | 7 of 455 |

Recall per restriction (forbidden pairs caught / forbidden pairs):

| classifier | egg | gluten | lactose | nuts | vegan | vegetarian |
|---|---|---|---|---|---|---|
| keywords (current filter) | 44/50 | 89/94 | 90/94 | 15/15 | 129/133 | 68/69 |
| LLM (ministral-3:8b) | 44/50 | 85/94 | 92/94 | 15/15 | 129/133 | 67/69 |
| keywords + LLM | 47/50 | 93/94 | 92/94 | 15/15 | 132/133 | 69/69 |

Precision per restriction (correct blocks / blocks):

| classifier | egg | gluten | lactose | nuts | vegan | vegetarian |
|---|---|---|---|---|---|---|
| keywords (current filter) | 44/44 | 89/89 | 90/91 | 15/15 | 129/129 | 68/69 |
| LLM (ministral-3:8b) | 44/62 | 85/89 | 92/99 | 15/15 | 129/131 | 67/116 |
| keywords + LLM | 47/65 | 93/97 | 92/99 | 15/15 | 132/134 | 69/118 |

Unsafe recipes missed by keywords (current filter):

- egg: ravioli with sage cream sauce (ravioli; butter; pecans; shallot; white wine; whipping cream; fresh sage; parmesan cheese)
- egg: almost fat free banana pudding (reduced-fat vanilla wafers; bananas; fat free cream cheese; fat-free sweetened condensed milk; sugar free fat free french vanilla pudding and pie filling mix; 1% low-fat milk; fat-free whipped topping)
- egg: strawberry vanilla pudding shortcake (milk; instant vanilla pudding; cool whip; poundcake; strawberry)
- egg: crispy fried onion strings (sweet onion; ranch dressing; milk; flour; salt; black pepper; cayenne pepper; vegetable oil)
- egg: linguine with leeks and mushrooms (leek; button mushrooms; bay leaf; butter; plain flour; low-fat milk; fresh chives; chives; salt; black pepper; fresh linguine)
- egg: blueberry tortellini salad (tortellini; fresh blueberries; fresh strawberries; mandarin oranges; green grape; pecans; sugar; vinegar; fresh lemon juice; salt; dry mustard; oil; poppy seeds)
- gluten: ricotta swiss wedges  crazy meatloaf (eggs; cooked bacon; fine dry breadcrumb; tomato sauce; hot salsa; dried basil; dried oregano; dried thyme; salt; pepper; lean ground beef; light ricotta cheese; swiss cheese; fresh parsley)
- gluten: strawberry vanilla pudding shortcake (milk; instant vanilla pudding; cool whip; poundcake; strawberry)
- gluten: pasta e vognole  pasta   clam soup (bacon; onion; celery ribs; bell pepper; carrot; garlic cloves; dry white wine; fat free chicken broth; clams; bay leaves; oregano leaves; dried oregano; fresh parsley leaves; fresh ground black pepper; kosher salt; diced tomatoes; tomato paste; ditalini)
- gluten: cranberry pecan butter tarts (eggs; corn syrup; granulated sugar; butter; vanilla; pecans; cranberries; 3-inch tart shells)
- gluten: mixed dhal (ghee; onions; garlic cloves; fresh ginger; fresh green chile; black mustard seeds; ground cumin; ground coriander; ground turmeric; asafoetida powder; toor dal; urad dal; mung dal; channa dal; crushed tomatoes; chicken stock; coconut cream; cilantro)
- lactose: raspberry champagne punch (frozen raspberries in light syrup; lemon juice concentrate; sugar; rose wine; champagne; raspberry sherbet)
- lactose: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
- lactose: white chocolate cashew popcorn (popcorn; cashew halves; white almond bark)
- lactose: blueberry tortellini salad (tortellini; fresh blueberries; fresh strawberries; mandarin oranges; green grape; pecans; sugar; vinegar; fresh lemon juice; salt; dry mustard; oil; poppy seeds)
- vegan: raspberry champagne punch (frozen raspberries in light syrup; lemon juice concentrate; sugar; rose wine; champagne; raspberry sherbet)
- vegan: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
- vegan: white chocolate cashew popcorn (popcorn; cashew halves; white almond bark)
- vegan: blueberry tortellini salad (tortellini; fresh blueberries; fresh strawberries; mandarin oranges; green grape; pecans; sugar; vinegar; fresh lemon juice; salt; dry mustard; oil; poppy seeds)
- vegetarian: hamburger and green bean casserole (cream of mushroom soup; milk; seasoning salt; pepper; fried onions; french style green beans; hamburger; cheese; mashed potatoes)

Unsafe recipes missed by LLM (ministral-3:8b):

- egg: almost fat free banana pudding (reduced-fat vanilla wafers; bananas; fat free cream cheese; fat-free sweetened condensed milk; sugar free fat free french vanilla pudding and pie filling mix; 1% low-fat milk; fat-free whipped topping)
- egg: hamentashen (flour; salt; sugar; butter; ice water; egg yolk; plastic wrap; wax paper; cookie cutter; baking sheet; jam)
- egg: smoked salmon   camembert frittata (smoked salmon; asparagus spears; camembert cheese; eggs; cream; salt and pepper; baby salad leaves; crusty bread)
- egg: crispy fried onion strings (sweet onion; ranch dressing; milk; flour; salt; black pepper; cayenne pepper; vegetable oil)
- egg: linguine with leeks and mushrooms (leek; button mushrooms; bay leaf; butter; plain flour; low-fat milk; fresh chives; chives; salt; black pepper; fresh linguine)
- egg: passover egg noodles (eggs; water; potato starch; salt; oil)
- gluten: karen s blueberry yogurt muffins (flour; sugar; baking powder; nutmeg; salt; eggs; blueberry yogurt; orange rind; orange juice; vegetable oil; blueberries)
- gluten: millionaire s shortbread (unsalted butter; unbleached all-purpose flour; granulated sugar; kosher salt; sweetened condensed milk; milk chocolate; fleur de sel)
- gluten: hamentashen (flour; salt; sugar; butter; ice water; egg yolk; plastic wrap; wax paper; cookie cutter; baking sheet; jam)
- gluten: blue cheese pasta (half-and-half; butter; garlic; onion; blue cheese; parmesan cheese; salt and pepper; pasta)
- gluten: garlic cheese french bread (french bread; swiss cheese; garlic; half-and-half; fresh parsley)
- gluten: choco banana squares (butter; sugar; light brown sugar; banana; flour; baking powder; salt; semi-sweet chocolate chips)
- gluten: daddy s fluffy homemade pancakes (flour; sugar; baking soda; baking powder; milk; oil; egg)
- gluten: rigatoni caprese (rigatoni pasta; plum tomatoes; fresh basil leaves; fresh mozzarella cheese; extra virgin olive oil; capers; salt; fresh ground pepper; garlic clove; fresh parmesan cheese)
- gluten: mixed dhal (ghee; onions; garlic cloves; fresh ginger; fresh green chile; black mustard seeds; ground cumin; ground coriander; ground turmeric; asafoetida powder; toor dal; urad dal; mung dal; channa dal; crushed tomatoes; chicken stock; coconut cream; cilantro)
- lactose: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
- lactose: blueberry tortellini salad (tortellini; fresh blueberries; fresh strawberries; mandarin oranges; green grape; pecans; sugar; vinegar; fresh lemon juice; salt; dry mustard; oil; poppy seeds)
- vegan: amazing garlicy balsamic vinaigrette (sweet onion; garlic cloves; red bell pepper; fresh basil leaf; oregano leaves; salt; black pepper; coleman's dry mustard; worcestershire sauce; fresh lemon juice; balsamic vinegar; extra virgin olive oil)
- vegan: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
- vegan: chef flower s how do i make grill marks on a steak (filet of beef; olive oil; salt; black pepper)
- vegan: fresh greens  avocado with a lime vinaigrette (red onion; avocado; red leaf lettuce; romaine lettuce; fresh cilantro; black beans; kosher salt; ground black pepper; grape tomatoes; olive oil; balsamic vinegar; honey; jalapeno pepper; garlic; fresh ginger; fresh lime juice; red pepper flakes)
- vegetarian: amazing garlicy balsamic vinaigrette (sweet onion; garlic cloves; red bell pepper; fresh basil leaf; oregano leaves; salt; black pepper; coleman's dry mustard; worcestershire sauce; fresh lemon juice; balsamic vinegar; extra virgin olive oil)
- vegetarian: chef flower s how do i make grill marks on a steak (filet of beef; olive oil; salt; black pepper)

Unsafe recipes missed by keywords + LLM:

- egg: almost fat free banana pudding (reduced-fat vanilla wafers; bananas; fat free cream cheese; fat-free sweetened condensed milk; sugar free fat free french vanilla pudding and pie filling mix; 1% low-fat milk; fat-free whipped topping)
- egg: crispy fried onion strings (sweet onion; ranch dressing; milk; flour; salt; black pepper; cayenne pepper; vegetable oil)
- egg: linguine with leeks and mushrooms (leek; button mushrooms; bay leaf; butter; plain flour; low-fat milk; fresh chives; chives; salt; black pepper; fresh linguine)
- gluten: mixed dhal (ghee; onions; garlic cloves; fresh ginger; fresh green chile; black mustard seeds; ground cumin; ground coriander; ground turmeric; asafoetida powder; toor dal; urad dal; mung dal; channa dal; crushed tomatoes; chicken stock; coconut cream; cilantro)
- lactose: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
- lactose: blueberry tortellini salad (tortellini; fresh blueberries; fresh strawberries; mandarin oranges; green grape; pecans; sugar; vinegar; fresh lemon juice; salt; dry mustard; oil; poppy seeds)
- vegan: lime freeze  non alcoholic beverage (club soda; ice; lime sherbet; frozen limeade concentrate)
