import hashlib
import json
import redis


class QueryCache:
    def __init__(self, redis_url="redis://localhost:6379", ttl_seconds=300):
        # TTL of 5 minutes is a starting point, not arbitrary forever - short
        # enough that stale results don't linger long after new inserts,
        # long enough to actually catch repeated queries in a demo/load test.
        # Documented here so it's a visible, tunable decision, not a magic number.
        self._ttl_seconds = ttl_seconds
        try:
            self._client = redis.from_url(redis_url, socket_connect_timeout=1)
            self._client.ping()
            self._available = True
        except redis.exceptions.RedisError:
            # Redis being down must not take the search engine down with it -
            # degrade to 'no caching' rather than raising here.
            self._client = None
            self._available = False

    def _make_key(self, query_vector, k):
        # Round to reduce float-noise cache misses (two near-identical
        # queries from float precision shouldn't be treated as different).
        rounded = [round(v, 4) for v in query_vector]
        raw = json.dumps({"vector": rounded, "k": k}, sort_keys=True)
        return "query_cache:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query_vector, k):
        if not self._available:
            return None
        try:
            key = self._make_key(query_vector, k)
            cached = self._client.get(key)
            return json.loads(cached) if cached else None
        except redis.exceptions.RedisError:
            # Redis dying mid-request (not just at startup) - same rule,
            # treat as a cache miss, don't let it break the search.
            return None

    def set(self, query_vector, k, result):
        if not self._available:
            return
        try:
            key = self._make_key(query_vector, k)
            self._client.set(key, json.dumps(result), ex=self._ttl_seconds)
        except redis.exceptions.RedisError:
            pass
