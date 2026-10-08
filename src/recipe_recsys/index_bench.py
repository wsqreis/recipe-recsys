"""Does search need an approximate nearest-neighbor index?

Search ranks all 231k recipe embeddings against the query with one matrix-
vector product, then applies the hard filters. This module measures that
against FAISS indexes (exact, HNSW, IVF): latency per query, recall@k of the
approximate indexes with respect to exact search, build time and memory.

It also measures what an approximate index would cost in this design. An ANN
index returns the k nearest recipes; the filters (restrictions, time,
ingredients) come afterwards, so with a filter that keeps a fraction f of the
catalog the index must be asked for about k / f candidates to still return k.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchRow:
    method: str
    recall: float  # recall@k against exact search
    median_ms: float
    p95_ms: float
    build_s: float
    memory_mb: float


def _timed(search, queries: np.ndarray, k: int) -> tuple[np.ndarray, list[float]]:
    results, times = [], []
    for q in queries:
        start = time.perf_counter()
        results.append(search(q[None, :], k))
        times.append((time.perf_counter() - start) * 1000)
    return np.vstack(results), times


def _recall(found: np.ndarray, exact: np.ndarray) -> float:
    pairs = zip(found, exact, strict=True)
    return float(np.mean([len(set(f) & set(e)) / len(e) for f, e in pairs]))


def benchmark(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> list[BenchRow]:
    """One query at a time, as in serving. `vectors` and `queries` are unit-norm float32."""
    import faiss

    faiss.omp_set_num_threads(1)  # a fair comparison with single-threaded numpy per query

    def numpy_search(q: np.ndarray, k: int) -> np.ndarray:
        scores = vectors @ q[0]
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])][None, :]

    exact, times = _timed(numpy_search, queries, k)
    size_mb = vectors.nbytes / 2**20
    rows = [BenchRow("numpy brute force (current)", 1.0, *_stats(times), 0.0, size_mb)]

    dim = vectors.shape[1]
    candidates = {
        "faiss Flat (exact)": lambda: faiss.IndexFlatIP(dim),
        "faiss HNSW32": lambda: faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT),
        "faiss IVF1024": lambda: faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim), dim, 1024, faiss.METRIC_INNER_PRODUCT
        ),
    }
    for name, make in candidates.items():
        index = make()
        start = time.perf_counter()
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        build = time.perf_counter() - start
        memory = faiss.serialize_index(index).nbytes / 2**20
        settings = {
            "faiss HNSW32": [("efSearch", v) for v in (16, 64, 256)],
            "faiss IVF1024": [("nprobe", v) for v in (4, 16, 64)],
        }.get(name, [(None, None)])
        for param, value in settings:
            if param == "efSearch":
                index.hnsw.efSearch = value
            elif param == "nprobe":
                index.nprobe = value
            found, times = _timed(lambda q, k, index=index: index.search(q, k)[1], queries, k)
            label = f"{name}, {param}={value}" if param else name
            rows.append(BenchRow(label, _recall(found, exact), *_stats(times), build, memory))
    return rows


def _stats(times: list[float]) -> tuple[float, float]:
    return float(np.median(times)), float(np.percentile(times, 95))


def bench_markdown(rows: list[BenchRow], n_vectors: int, n_queries: int, k: int) -> str:
    lines = [
        f"{n_vectors:,} vectors, {n_queries:,} queries one at a time, k={k}, single thread.",
        "",
        f"| method | recall@{k} | median latency | p95 latency | build | index size |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.method} | {r.recall:.3f} | {r.median_ms:.2f} ms | {r.p95_ms:.2f} ms "
            f"| {r.build_s:.1f} s | {r.memory_mb:.0f} MB |"
        )
    return "\n".join(lines)
