import numpy as np

from app.coordinator.coordinator import Coordinator
from app.core.shard_config import SHARD_ADDRESSES


def run():
    coordinator = Coordinator(SHARD_ADDRESSES)
    rng = np.random.default_rng(seed=1)

    query_vector = rng.random(8).astype(np.float32).tolist()
    search_result = coordinator.search(query_vector, k=5)

    print("Top results:")
    for r in search_result["results"]:
        print("  id:", r["id"], "score:", r["score"], "from:", r["shard_id"])

    print("Failed shards:", search_result["failed_shards"])


if __name__ == "__main__":
    run()
