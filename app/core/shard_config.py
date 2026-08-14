# Explicit list of shard addresses, not auto-discovered. In Docker/k8s
# (Phase 9) this becomes a list of service DNS names instead of
# localhost ports, but the coordinator's logic doesn't change either way.

SHARD_ADDRESSES = {
    "shard-0": "localhost:50051",
    "shard-1": "localhost:50052",
    "shard-2": "localhost:50053",
}
