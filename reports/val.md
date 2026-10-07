| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0002 | 0.0003 | 0.0022 | 0.6732 | 0.0005 | 0.0004 | 0.0050 | 0.8954 | 2.4 |
| popularity() | 0.0249 | 0.0185 | 0.0807 | 0.0012 | 0.0425 | 0.0239 | 0.1350 | 0.0020 | 1.7 |
| recent_popularity(days=180) | 0.0261 | 0.0191 | 0.0827 | 0.0011 | 0.0450 | 0.0243 | 0.1360 | 0.0016 | 1.7 |
| itemknn(neighbors=50, shrink=10.0) | 0.0137 | 0.0103 | 0.0499 | 0.2564 | 0.0229 | 0.0130 | 0.0829 | 0.4069 | 1.5 |
| itemknn(neighbors=50, shrink=50.0) | 0.0193 | 0.0148 | 0.0692 | 0.0645 | 0.0309 | 0.0184 | 0.1114 | 0.1187 | 1.5 |
| itemknn(neighbors=50, shrink=200.0) | 0.0248 | 0.0181 | 0.0804 | 0.0199 | 0.0385 | 0.0222 | 0.1273 | 0.0413 | 1.5 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0258 | 0.0196 | 0.0869 | 0.0105 | 0.0413 | 0.0240 | 0.1375 | 0.0188 | 2.2 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1) | 0.0257 | 0.0192 | 0.0836 | 0.0227 | 0.0407 | 0.0234 | 0.1296 | 0.0378 | 2.3 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=0, logq=1) | 0.0221 | 0.0151 | 0.0707 | 0.0106 | 0.0376 | 0.0199 | 0.1201 | 0.0188 | 2.3 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=0) | 0.0012 | 0.0010 | 0.0062 | 0.4151 | 0.0028 | 0.0015 | 0.0132 | 0.5920 | 2.2 |
