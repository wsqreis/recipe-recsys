LLM: ministral-3:8b

30 labeled requests (Portuguese and English).

| restrictions from | precision | recall | requests with a missed restriction |
|---|---|---|---|
| LLM | 88% | 100% | 0 of 30 |
| keywords | 100% | 67% | 6 of 30 |
| LLM + keywords | 88% | 100% | 0 of 30 |

| end to end (union) | value |
|---|---|
| max minutes parsed correctly | 100% |
| required ingredients parsed correctly | 87% |
| excluded ingredients parsed correctly | 93% |
| requests with at least one result | 100% |
| results violating a stated restriction | 0 of 299 |
| median LLM latency | 1.2 s |

Parsing errors:

- `frango grelhado com legumes`: include ['chicken', 'vegetables'] (expected ['chicken'])
- `salmon with no butter`: spurious ['lactose'], exclude [] (expected ['butter'])
- `pão de queijo`: include [] (expected ['cheese'])
- `omelete sem cebola`: spurious ['egg']
- `vegan curry with chickpeas`: include ['chickpeas', 'curry'] (expected ['chickpea'])
- `chili without pork, under an hour`: spurious ['vegetarian'], exclude [] (expected ['pork'])
- `pancakes, but my son can't have milk`: include ['pancakes'] (expected [])
