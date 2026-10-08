## new_user

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | ndcg@10 95% CI | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0000 | 0.0004 | 0.0035 | 0.0757 | 0.0012 | 0.0007 | 0.0071 | 0.1457 | [0.0000, 0.0012] | 0.1 |
| popularity() | 0.0405 | 0.0287 | 0.1170 | 0.0004 | 0.0634 | 0.0336 | 0.1702 | 0.0007 | [0.0183, 0.0403] | 0.1 |
| recent_popularity(days=180) | 0.0293 | 0.0254 | 0.1135 | 0.0004 | 0.0605 | 0.0335 | 0.1667 | 0.0006 | [0.0163, 0.0360] | 0.0 |
| itemknn(neighbors=50, shrink=200) | 0.0447 | 0.0311 | 0.1241 | 0.0075 | 0.0688 | 0.0379 | 0.1879 | 0.0154 | [0.0195, 0.0447] | 0.0 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0467 | 0.0332 | 0.1241 | 0.0041 | 0.0646 | 0.0372 | 0.1738 | 0.0072 | [0.0213, 0.0475] | 0.1 |
| text_profile(history=30) | 0.0057 | 0.0032 | 0.0213 | 0.0501 | 0.0099 | 0.0049 | 0.0355 | 0.0860 | [0.0007, 0.0059] | 0.1 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 0.0393 | 0.0270 | 0.1064 | 0.0031 | 0.0744 | 0.0361 | 0.1667 | 0.0060 | [0.0168, 0.0406] | 0.1 |

| model | Δndcg@10 vs popularity() | 95% CI | relative | significant |
|---|---|---|---|---|
| random(seed=42) | -0.0283 | [-0.0397, -0.0180] | -99% | yes |
| recent_popularity(days=180) | -0.0033 | [-0.0129, +0.0054] | -11% | no |
| itemknn(neighbors=50, shrink=200) | +0.0024 | [-0.0050, +0.0104] | +8% | no |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | +0.0045 | [-0.0032, +0.0122] | +16% | no |
| text_profile(history=30) | -0.0256 | [-0.0371, -0.0151] | -89% | yes |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | -0.0017 | [-0.0087, +0.0061] | -6% | no |
