| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | ndcg@10 95% CI | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0007 | 0.0007 | 0.0029 | 0.4547 | 0.0008 | 0.0008 | 0.0050 | 0.7030 | [0.0001, 0.0017] | 0.7 |
| popularity() | 0.0226 | 0.0140 | 0.0582 | 0.0011 | 0.0368 | 0.0185 | 0.0955 | 0.0018 | [0.0113, 0.0169] | 0.5 |
| recent_popularity(days=180) | 0.0177 | 0.0121 | 0.0523 | 0.0008 | 0.0341 | 0.0169 | 0.0921 | 0.0013 | [0.0096, 0.0149] | 0.4 |
| itemknn(neighbors=50, shrink=200) | 0.0220 | 0.0154 | 0.0620 | 0.0116 | 0.0367 | 0.0196 | 0.0992 | 0.0247 | [0.0121, 0.0185] | 0.4 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0237 | 0.0162 | 0.0645 | 0.0096 | 0.0383 | 0.0207 | 0.1039 | 0.0164 | [0.0131, 0.0197] | 0.7 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 0.0215 | 0.0141 | 0.0603 | 0.0164 | 0.0365 | 0.0184 | 0.0963 | 0.0273 | [0.0115, 0.0172] | 0.7 |

| model | Δndcg@10 vs popularity() | 95% CI | relative | significant |
|---|---|---|---|---|
| random(seed=42) | -0.0133 | [-0.0163, -0.0105] | -95% | yes |
| recent_popularity(days=180) | -0.0018 | [-0.0042, +0.0005] | -13% | no |
| itemknn(neighbors=50, shrink=200) | +0.0014 | [-0.0009, +0.0036] | +10% | no |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | +0.0023 | [+0.0003, +0.0043] | +16% | yes |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | +0.0001 | [-0.0019, +0.0025] | +1% | no |
