import grpc
from concurrent import futures
import numpy as np

from app.grpc_service import shard_pb2, shard_pb2_grpc
from app.index.vector_index import VectorIndex


class ShardServicer(shard_pb2_grpc.ShardServiceServicer):
    def __init__(self, dim, max_elements=100000):
        self._index = VectorIndex(dim=dim, max_elements=max_elements)

    def Insert(self, request, context):
        try:
            vector = np.array(request.vector, dtype=np.float32)
            metadata = dict(request.metadata)
            item_id = self._index.insert(vector, metadata)
            return shard_pb2.InsertResponse(item_id=item_id)
        except ValueError as e:
            # Client sent bad data (wrong dimension) - this is their error,
            # not a server fault, so INVALID_ARGUMENT rather than a generic 500.
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return shard_pb2.InsertResponse()
        except RuntimeError as e:
            # Index genuinely full - this is a real server-side limit being hit.
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(e))
            return shard_pb2.InsertResponse()

    def Search(self, request, context):
        try:
            query_vector = np.array(request.query_vector, dtype=np.float32)
            results = self._index.search(query_vector, k=request.k)

            proto_results = []
            for r in results:
                str_metadata = {k: str(v) for k, v in r["metadata"].items()}
                proto_results.append(
                    shard_pb2.SearchResult(
                        id=r["id"],
                        score=r["score"],
                        metadata=str_metadata,
                    )
                )
            return shard_pb2.SearchResponse(results=proto_results)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return shard_pb2.SearchResponse()

    def HealthCheck(self, request, context):
        # Reports current item count too, not just a boolean - useful for
        # the coordinator to detect a shard that's technically alive but
        # stuck/empty when it shouldn't be.
        return shard_pb2.HealthCheckResponse(
            healthy=True,
            item_count=self._index._next_id,
        )


def serve(port, dim, max_elements=100000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    shard_pb2_grpc.add_ShardServiceServicer_to_server(
        ShardServicer(dim=dim, max_elements=max_elements), server
    )
    server.add_insecure_port("[::]:" + str(port))
    server.start()
    print("Shard server running on port " + str(port))
    server.wait_for_termination()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    serve(port, dim)
