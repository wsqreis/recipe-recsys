| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | time (s) |
|---|---|---|---|---|---|
| popularity() | 0.0296 | 0.0205 | 0.0623 | 0.0008 | 0.6 |
| itemknn(neighbors=50, shrink=50.0) | 0.0221 | 0.0176 | 0.0545 | 0.0290 | 0.6 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0337 | 0.0242 | 0.0758 | 0.0075 | 0.8 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1) | 0.0311 | 0.0215 | 0.0687 | 0.0120 | 0.8 |
