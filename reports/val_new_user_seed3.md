## new_user

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | ndcg@10 95% CI | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0000 | 0.0000 | 0.0000 | 0.1723 | 0.0000 | 0.0000 | 0.0000 | 0.3165 | [0.0000, 0.0000] | 0.3 |
| popularity() | 0.0348 | 0.0252 | 0.0929 | 0.0004 | 0.0516 | 0.0297 | 0.1342 | 0.0006 | [0.0182, 0.0326] | 0.1 |
| recent_popularity(days=180) | 0.0361 | 0.0260 | 0.1018 | 0.0004 | 0.0607 | 0.0330 | 0.1475 | 0.0006 | [0.0190, 0.0336] | 0.2 |
| itemknn(neighbors=50, shrink=200) | 0.0328 | 0.0246 | 0.0782 | 0.0260 | 0.0619 | 0.0330 | 0.1313 | 0.0487 | [0.0174, 0.0326] | 0.1 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0406 | 0.0274 | 0.0929 | 0.0088 | 0.0620 | 0.0336 | 0.1475 | 0.0160 | [0.0201, 0.0346] | 0.3 |
| text_profile(history=30) | 0.0021 | 0.0014 | 0.0088 | 0.1180 | 0.0081 | 0.0033 | 0.0221 | 0.1929 | [0.0004, 0.0028] | 0.3 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 0.0331 | 0.0250 | 0.0841 | 0.0040 | 0.0585 | 0.0326 | 0.1357 | 0.0077 | [0.0177, 0.0330] | 0.4 |

| model | Δndcg@10 vs popularity() | 95% CI | relative | significant |
|---|---|---|---|---|
| random(seed=42) | -0.0252 | [-0.0326, -0.0182] | -100% | yes |
| recent_popularity(days=180) | +0.0008 | [-0.0045, +0.0064] | +3% | no |
| itemknn(neighbors=50, shrink=200) | -0.0006 | [-0.0076, +0.0062] | -2% | no |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | +0.0022 | [-0.0036, +0.0081] | +9% | no |
| text_profile(history=30) | -0.0238 | [-0.0312, -0.0166] | -95% | yes |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | -0.0002 | [-0.0061, +0.0053] | -1% | no |
