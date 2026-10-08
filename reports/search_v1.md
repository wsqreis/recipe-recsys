LLM: ministral-3:8b

42 labeled requests (Portuguese and English).

| restrictions from | precision | recall | requests with a missed restriction |
|---|---|---|---|
| LLM | 100% | 97% | 1 of 42 |
| keywords | 100% | 84% | 4 of 42 |
| LLM + keywords | 100% | 97% | 1 of 42 |

| end to end (union) | value |
|---|---|
| max minutes parsed correctly | 100% |
| required ingredients parsed correctly | 86% |
| excluded ingredients parsed correctly | 98% |
| requests with at least one result | 98% |
| results violating a stated restriction | 8 of 399 |
| median LLM latency | 1.2 s |

Parsing errors:

- `lasanha de carne`: include ['meat', 'lasagna'] (expected ['beef'])
- `algo leve, não como carne`: missed ['vegetarian'], exclude ['meat'] (expected [])
- `macarrão sem queijo pronto em 25 min`: include ['pasta'] (expected [])
- `nut-free brownies for a school bake sale`: include ['brownies'] (expected [])
- `plant-based protein bowl`: include ['protein'] (expected [])
- `shrimp pasta, I'm allergic to gluten`: include ['shrimp', 'pasta'] (expected ['shrimp'])
- `cheesecake that's egg free`: include ['cheese'] (expected [])
