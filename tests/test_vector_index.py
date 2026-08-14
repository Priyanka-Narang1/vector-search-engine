import numpy as np
import pytest
from app.index.vector_index import VectorIndex


def _brute_force_cosine_topk(query: np.ndarray, vectors: dict[int, np.ndarray], k: int) -> list[int]:
    # Ground truth to validate hnswlib's approximate results against -
    # exact cosine similarity computed directly, no shortcuts.
    scores = {}
    for item_id, vec in vectors.items():
        cos_sim = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
        scores[item_id] = cos_sim
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked[:k]]


def test_insert_and_search_matches_brute_force():
    dim = 16
    rng = np.random.default_rng(seed=42)  # fixed seed - reproducible test, not flaky

    index = VectorIndex(dim=dim, max_elements=1000)
    raw_vectors = {}

    for i in range(200):
        vec = rng.random(dim).astype(np.float32)
        item_id = index.insert(vec, metadata={"source": f"doc_{i}"})
        raw_vectors[item_id] = vec

    query = rng.random(dim).astype(np.float32)
    k = 5

    hnsw_results = index.search(query, k=k)
    hnsw_ids = [r["id"] for r in hnsw_results]

    expected_ids = _brute_force_cosine_topk(query, raw_vectors, k=k)

    # hnswlib is approximate, not exact - at small scale (200 items) with these
    # ef/M settings it should still match brute-force exactly, but we assert
    # on overlap rather than strict equality so this doesn't become flaky if
    # index params change later.
    overlap = len(set(hnsw_ids) & set(expected_ids))
    assert overlap >= k - 1, f"expected near-exact match, got {overlap}/{k} overlap"


def test_search_on_empty_index_returns_empty_list():
    index = VectorIndex(dim=8, max_elements=100)
    query = np.random.rand(8).astype(np.float32)

    results = index.search(query, k=5)

    assert results == []


def test_insert_rejects_wrong_dimension():
    index = VectorIndex(dim=8, max_elements=100)
    wrong_dim_vector = np.random.rand(4).astype(np.float32)

    with pytest.raises(ValueError):
        index.insert(wrong_dim_vector, metadata={})


def test_search_rejects_wrong_dimension():
    index = VectorIndex(dim=8, max_elements=100)
    index.insert(np.random.rand(8).astype(np.float32), metadata={})
    wrong_dim_query = np.random.rand(4).astype(np.float32)

    with pytest.raises(ValueError):
        index.search(wrong_dim_query, k=1)


def test_search_k_larger_than_index_size_does_not_crash():
    index = VectorIndex(dim=8, max_elements=100)
    index.insert(np.random.rand(8).astype(np.float32), metadata={"n": 1})
    index.insert(np.random.rand(8).astype(np.float32), metadata={"n": 2})

    results = index.search(np.random.rand(8).astype(np.float32), k=10)

    assert len(results) == 2  # capped to actual item count, not padded or crashed


def test_insert_beyond_capacity_raises_not_silently_fails():
    index = VectorIndex(dim=4, max_elements=2)
    index.insert(np.random.rand(4).astype(np.float32), metadata={})
    index.insert(np.random.rand(4).astype(np.float32), metadata={})

    with pytest.raises(RuntimeError):
        index.insert(np.random.rand(4).astype(np.float32), metadata={})
