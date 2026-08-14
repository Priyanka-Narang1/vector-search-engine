import grpc

from app.sharding.consistent_hash import ConsistentHashRing
from app.grpc_service import shard_pb2, shard_pb2_grpc


class Coordinator:
    def __init__(self, shard_addresses, timeout_seconds=2.0):
        # timeout_seconds has a concrete justification: it must be long
        # enough for a normal search under load, short enough that a
        # single unreachable shard doesn't stall the whole query for the
        # caller. 2s is a starting point, tuned later against real
        # latency numbers from the Phase 10 load test, not a guess left
        # unexamined.
        self._shard_addresses = shard_addresses
        self._timeout_seconds = timeout_seconds
        self._ring = ConsistentHashRing(virtual_nodes_per_shard=150)
        self._stubs = {}

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
            # Surfacing which shard failed and why, not swallowing it -
            # a caller (or the API layer above this) needs to know insert
            # didn't happen, not get a silent no-op.
            raise RuntimeError(
                "insert failed on " + shard_id + ": " + e.code().name + " - " + e.details()
            )

    def search(self, query_vector, k=5):
        # Fan-out: query ALL shards, since a query key has no natural
        # shard affinity the way an insert's item_key does - the nearest
        # neighbors could be on any shard. Results are merged after.
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
                # A slow/unreachable shard must not block the whole query -
                # record the failure, continue with the others, and report
                # partial results plus which shards were unreachable so the
                # caller can tell the difference between 'no results exist'
                # and 'some results may be missing'.
                failed_shards.append({"shard_id": shard_id, "error": e.code().name})

        # cosine distance from hnswlib: LOWER is more similar, so ascending sort.
        all_results.sort(key=lambda r: r["score"])
        merged = all_results[:k]

        return {"results": merged, "failed_shards": failed_shards}
