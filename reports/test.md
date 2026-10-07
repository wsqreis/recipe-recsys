| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | recall@20 | ndcg@20 | hit_rate@20 | coverage@20 | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| random(seed=42) | 0.0007 | 0.0007 | 0.0029 | 0.4547 | 0.0008 | 0.0008 | 0.0050 | 0.7030 | 1.5 |
| popularity() | 0.0226 | 0.0140 | 0.0582 | 0.0011 | 0.0368 | 0.0185 | 0.0955 | 0.0018 | 1.1 |
| recent_popularity(days=180) | 0.0177 | 0.0121 | 0.0523 | 0.0008 | 0.0341 | 0.0169 | 0.0921 | 0.0013 | 1.0 |
| itemknn(neighbors=50, shrink=10.0) | 0.0171 | 0.0129 | 0.0486 | 0.1625 | 0.0225 | 0.0143 | 0.0641 | 0.2790 | 1.0 |
| itemknn(neighbors=50, shrink=50.0) | 0.0218 | 0.0145 | 0.0540 | 0.0395 | 0.0320 | 0.0176 | 0.0884 | 0.0733 | 1.0 |
| itemknn(neighbors=50, shrink=200.0) | 0.0220 | 0.0154 | 0.0620 | 0.0116 | 0.0367 | 0.0196 | 0.0992 | 0.0247 | 0.9 |
| ials(factors=32, reg=100.0, alpha=5.0, iterations=10, seed=42) | 0.0237 | 0.0162 | 0.0645 | 0.0096 | 0.0383 | 0.0207 | 0.1039 | 0.0164 | 1.3 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=1) | 0.0209 | 0.0142 | 0.0616 | 0.0165 | 0.0365 | 0.0186 | 0.0963 | 0.0275 | 1.4 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=0, logq=1) | 0.0198 | 0.0124 | 0.0532 | 0.0101 | 0.0340 | 0.0170 | 0.0913 | 0.0175 | 1.4 |
| two_tower(dim=64, epochs=5, batch_size=1024, lr=0.001, temperature=0.05, max_history=30, min_token_count=10, seed=42, device=auto, content=1, logq=0) | 0.0018 | 0.0010 | 0.0054 | 0.2704 | 0.0038 | 0.0015 | 0.0092 | 0.4162 | 1.4 |
