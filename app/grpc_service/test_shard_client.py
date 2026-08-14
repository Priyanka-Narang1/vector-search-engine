import grpc
import numpy as np

from app.grpc_service import shard_pb2, shard_pb2_grpc


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = shard_pb2_grpc.ShardServiceStub(channel)

    health = stub.HealthCheck(shard_pb2.HealthCheckRequest())
    print("Health check:", health.healthy, "item_count:", health.item_count)

    rng = np.random.default_rng(seed=1)
    vector = rng.random(8).astype(np.float32).tolist()
    insert_response = stub.Insert(
        shard_pb2.InsertRequest(vector=vector, metadata={"source": "test"})
    )
    print("Inserted item_id:", insert_response.item_id)

    query_vector = rng.random(8).astype(np.float32).tolist()
    search_response = stub.Search(
        shard_pb2.SearchRequest(query_vector=query_vector, k=3)
    )
    print("Search results:")
    for r in search_response.results:
        print("  id:", r.id, "score:", r.score, "metadata:", dict(r.metadata))


if __name__ == "__main__":
    run()
