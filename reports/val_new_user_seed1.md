## new_user

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | ndcg@10 95% CI | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0004 | 0.0002 | 0.0007 | 0.1960 | 0.0004 | 0.0002 | 0.0007 | 0.3516 | [0.0000, 0.0005] | 0.7 |
| popularity() | 0.0415 | 0.0254 | 0.0768 | 0.0003 | 0.0638 | 0.0317 | 0.1157 | 0.0006 | [0.0212, 0.0293] | 0.5 |
| recent_popularity(days=180) | 0.0396 | 0.0254 | 0.0768 | 0.0003 | 0.0632 | 0.0320 | 0.1157 | 0.0006 | [0.0209, 0.0296] | 0.6 |
| itemknn(neighbors=50, shrink=200) | 0.0418 | 0.0289 | 0.0648 | 0.1853 | 0.0560 | 0.0330 | 0.0902 | 0.3269 | [0.0238, 0.0341] | 0.3 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0408 | 0.0258 | 0.0732 | 0.0302 | 0.0591 | 0.0310 | 0.1044 | 0.0565 | [0.0214, 0.0301] | 0.9 |
| text_profile(history=30) | 0.0129 | 0.0099 | 0.0191 | 0.3352 | 0.0164 | 0.0108 | 0.0244 | 0.5090 | [0.0070, 0.0133] | 1.1 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 0.0410 | 0.0244 | 0.0743 | 0.0067 | 0.0630 | 0.0308 | 0.1129 | 0.0112 | [0.0204, 0.0282] | 1.6 |

| model | Δndcg@10 vs popularity() | 95% CI | relative | significant |
|---|---|---|---|---|
| random(seed=42) | -0.0252 | [-0.0292, -0.0210] | -99% | yes |
| recent_popularity(days=180) | -0.0001 | [-0.0027, +0.0025] | -0% | no |
| itemknn(neighbors=50, shrink=200) | +0.0035 | [-0.0012, +0.0085] | +14% | no |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | +0.0004 | [-0.0027, +0.0035] | +1% | no |
| text_profile(history=30) | -0.0155 | [-0.0204, -0.0105] | -61% | yes |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | -0.0010 | [-0.0035, +0.0012] | -4% | no |
