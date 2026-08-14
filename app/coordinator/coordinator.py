import grpc

from app.sharding.consistent_hash import ConsistentHashRing
from app.grpc_service import shard_pb2, shard_pb2_grpc
from app.core.query_cache import QueryCache


class Coordinator:
    def __init__(self, shard_addresses, timeout_seconds=2.0, cache=None):
        self._shard_addresses = shard_addresses
        self._timeout_seconds = timeout_seconds
        self._ring = ConsistentHashRing(virtual_nodes_per_shard=150)
        self._stubs = {}
        # Cache is injectable (default: real QueryCache) so tests can swap
        # in a fake/None cache without touching Redis - keeps tests fast
        # and independent of whether Docker is running.
        self._cache = cache if cache is not None else QueryCache()

        for shard_id, address in shard_addresses.items():
            self._ring.add_shard(shard_id)
            channel = grpc.insecure_channel(address)
            self._stubs[shard_id] = shard_pb2_grpc.ShardServiceStub(channel)

    def insert(self, item_key, vector, metadata):
        shard_id = self._ring.get_shard(item_key)
        stub = self._stubs[shard_id]

        try:
            request = shard_pb2.InsertRequest(vector=vector, metadata=metadata)
            response = stub.Insert(request, timeout=self._timeout_seconds)
            return {"shard_id": shard_id, "item_id": response.item_id}
        except grpc.RpcError as e:
            raise RuntimeError(
                "insert failed on " + shard_id + ": " + e.code().name + " - " + e.details()
            )

    def search(self, query_vector, k=5):
        cached = self._cache.get(query_vector, k)
        if cached is not None:
            cached["from_cache"] = True
            return cached

        all_results = []
        failed_shards = []

        for shard_id, stub in self._stubs.items():
            try:
                request = shard_pb2.SearchRequest(query_vector=query_vector, k=k)
                response = stub.Search(request, timeout=self._timeout_seconds)
                for r in response.results:
                    all_results.append({
                        "id": r.id,
                        "score": r.score,
                        "metadata": dict(r.metadata),
                        "shard_id": shard_id,
                    })
            except grpc.RpcError as e:
                failed_shards.append({"shard_id": shard_id, "error": e.code().name})

        all_results.sort(key=lambda r: r["score"])
        merged = all_results[:k]
        result = {"results": merged, "failed_shards": failed_shards, "from_cache": False}

        # Only cache clean results - a query that hit failed shards shouldn't
        # be cached as if it were complete, or a later successful retry
        # would be masked by a stale partial-failure result.
        if not failed_shards:
            self._cache.set(query_vector, k, result)

        return result
