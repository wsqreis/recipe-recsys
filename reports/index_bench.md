231,637 vectors, 1,000 queries one at a time, k=10, single thread.

| method | recall@10 | median latency | p95 latency | build | index size |
|---|---|---|---|---|---|
| numpy brute force (current) | 1.000 | 20.18 ms | 23.89 ms | 0.0 s | 339 MB |
| faiss Flat (exact) | 1.000 | 20.78 ms | 24.62 ms | 0.3 s | 339 MB |
| faiss HNSW32, efSearch=16 | 0.961 | 0.11 ms | 0.20 ms | 46.2 s | 399 MB |
| faiss HNSW32, efSearch=64 | 0.979 | 0.27 ms | 0.43 ms | 46.2 s | 399 MB |
| faiss HNSW32, efSearch=256 | 0.990 | 0.87 ms | 1.26 ms | 46.2 s | 399 MB |
| faiss IVF1024, nprobe=4 | 0.851 | 0.18 ms | 0.27 ms | 46.1 s | 343 MB |
| faiss IVF1024, nprobe=16 | 0.938 | 0.58 ms | 0.83 ms | 46.1 s | 343 MB |
| faiss IVF1024, nprobe=64 | 0.978 | 1.93 ms | 2.61 ms | 46.1 s | 343 MB |

| hard filters | catalog left | candidates an ANN index would need for k=10 |
|---|---|---|
| no filter | 100.0% | ~10 |
| vegetarian | 56.7% | ~18 |
| dairy-free, with chicken, <= 20 min | 0.8% | ~1,255 |
| vegan, gluten-free, nut-free | 9.8% | ~103 |
| vegan, gluten-free, <= 15 min | 4.8% | ~209 |
