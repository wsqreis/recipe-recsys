import numpy as np
import pytest

pytest.importorskip("faiss")
from recipe_recsys.index_bench import benchmark  # noqa: E402


def test_exact_methods_agree_and_approximate_ones_report_recall():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((2000, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rows = {r.method: r for r in benchmark(vectors, vectors[:20], k=5)}
    assert rows["numpy brute force (current)"].recall == 1.0
    assert rows["faiss Flat (exact)"].recall == pytest.approx(1.0)
    assert all(0 <= r.recall <= 1 for r in rows.values())
