| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | time (s) |
|---|---|---|---|---|---|
| popularity() | 0.0239 | 0.0152 | 0.0515 | 0.0008 | 0.8 |
| itemknn(neighbors=50, shrink=50.0) | 0.0212 | 0.0141 | 0.0498 | 0.0319 | 0.8 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0230 | 0.0150 | 0.0544 | 0.0080 | 1.1 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1) | 0.0244 | 0.0158 | 0.0590 | 0.0134 | 1.1 |
