import time
import numpy as np

from app.coordinator.coordinator import Coordinator
from app.core.shard_config import SHARD_ADDRESSES
from app.core.query_cache import QueryCache


def test_repeated_query_hits_cache_and_is_faster():
    # Requires shard servers running on the configured ports (same as
    # Phase 4's manual test) and Redis running via docker.
    coordinator = Coordinator(SHARD_ADDRESSES)
    rng = np.random.default_rng(seed=5)
    query_vector = rng.random(8).astype(np.float32).tolist()

    start_first = time.perf_counter()
    first_result = coordinator.search(query_vector, k=3)
    first_duration = time.perf_counter() - start_first

    start_second = time.perf_counter()
    second_result = coordinator.search(query_vector, k=3)
    second_duration = time.perf_counter() - start_second

    assert first_result["from_cache"] is False
    assert second_result["from_cache"] is True
    # Cache hit should be meaningfully faster - not asserting a specific
    # millisecond number since that varies by machine, but the relative
    # speedup should be clear and consistent.
    assert second_duration < first_duration


def test_search_works_when_cache_unavailable():
    # Simulates Redis being down by pointing at a port nothing is
    # listening on - QueryCache's own connect-time try/except should
    # degrade to 'no caching', not raise.
    dead_cache = QueryCache(redis_url="redis://localhost:9999", ttl_seconds=60)
    assert dead_cache._available is False

    coordinator = Coordinator(SHARD_ADDRESSES, cache=dead_cache)
    rng = np.random.default_rng(seed=6)
    query_vector = rng.random(8).astype(np.float32).tolist()

    # Should not raise, should still return real results, from_cache
    # should be False since there's no cache to hit.
    result = coordinator.search(query_vector, k=3)
    assert result["from_cache"] is False
    assert isinstance(result["results"], list)
