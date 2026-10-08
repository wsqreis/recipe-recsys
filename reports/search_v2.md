LLM: ministral-3:8b

42 labeled requests (Portuguese and English).

| restrictions from | precision | recall | requests with a missed restriction |
|---|---|---|---|
| LLM | 100% | 100% | 0 of 42 |
| keywords | 100% | 84% | 4 of 42 |
| LLM + keywords | 100% | 100% | 0 of 42 |

| end to end (union) | value |
|---|---|
| max minutes parsed correctly | 100% |
| required ingredients parsed correctly | 83% |
| excluded ingredients parsed correctly | 98% |
| requests with at least one result | 98% |
| results violating a stated restriction | 0 of 395 |
| median LLM latency | 1.3 s |

Parsing errors:

- `lasanha de carne`: include [] (expected ['beef'])
- `algo leve, não como carne`: exclude ['meat'] (expected [])
- `bolo de cenoura`: include [] (expected ['carrot'])
- `macarrão sem queijo pronto em 25 min`: include ['pasta'] (expected [])
- `nut-free brownies for a school bake sale`: include ['brownies'] (expected [])
- `eggplant parmesan in under an hour`: include ['eggplant', 'parmesan'] (expected ['eggplant'])
- `shrimp pasta, I'm allergic to gluten`: include ['shrimp', 'pasta'] (expected ['shrimp'])
- `cheesecake that's egg free`: include ['cheesecake', 'cheese'] (expected [])
