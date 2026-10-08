## new_item

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | ndcg@10 95% CI | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0007 | 0.0022 | 0.0236 | 0.5357 | 0.0017 | 0.0022 | 0.0362 | 0.7809 | [0.0012, 0.0035] | 0.1 |
| random(seed=1) | 0.0006 | 0.0016 | 0.0181 | 0.5382 | 0.0008 | 0.0016 | 0.0264 | 0.7838 | [0.0008, 0.0026] | 0.1 |
| random(seed=2) | 0.0001 | 0.0012 | 0.0097 | 0.5347 | 0.0030 | 0.0019 | 0.0236 | 0.7857 | [0.0004, 0.0025] | 0.1 |
| newest() | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | [0.0000, 0.0000] | 0.0 |
| text_profile(history=30) | 0.0053 | 0.0051 | 0.0264 | 0.0874 | 0.0101 | 0.0061 | 0.0501 | 0.1419 | [0.0026, 0.0082] | 0.2 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 0.0005 | 0.0023 | 0.0223 | 0.0941 | 0.0025 | 0.0030 | 0.0431 | 0.1520 | [0.0012, 0.0035] | 0.2 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=1, id_dropout=0.0) | 0.0019 | 0.0039 | 0.0250 | 0.0773 | 0.0050 | 0.0044 | 0.0431 | 0.1275 | [0.0019, 0.0067] | 0.3 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=1, id_dropout=0.5) | 0.0032 | 0.0051 | 0.0236 | 0.0528 | 0.0041 | 0.0053 | 0.0431 | 0.0917 | [0.0024, 0.0089] | 0.3 |

| model | Δndcg@10 vs two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=0, id_dropout=0.0) | 95% CI | relative | significant |
|---|---|---|---|---|
| random(seed=42) | -0.0001 | [-0.0016, +0.0013] | -5% | no |
| random(seed=1) | -0.0007 | [-0.0022, +0.0008] | -30% | no |
| random(seed=2) | -0.0011 | [-0.0025, +0.0006] | -46% | no |
| newest() | -0.0023 | [-0.0035, -0.0012] | -100% | yes |
| text_profile(history=30) | +0.0028 | [+0.0001, +0.0059] | +119% | yes |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=1, id_dropout=0.0) | +0.0015 | [-0.0006, +0.0044] | +67% | no |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1, text=1, id_dropout=0.5) | +0.0028 | [-0.0000, +0.0065] | +122% | no |
