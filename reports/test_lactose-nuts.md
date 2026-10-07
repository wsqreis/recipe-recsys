| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | time (s) |
|---|---|---|---|---|---|
| popularity() | 0.0427 | 0.0248 | 0.0828 | 0.0008 | 0.5 |
| itemknn(neighbors=50, shrink=50.0) | 0.0317 | 0.0207 | 0.0631 | 0.0252 | 0.6 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0438 | 0.0281 | 0.0866 | 0.0065 | 0.7 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1) | 0.0335 | 0.0211 | 0.0684 | 0.0105 | 0.7 |
