import hashlib
import bisect


class ConsistentHashRing:
    # Naive consistent hashing (one point per shard) skews badly with few
    # shards - a small number of shards can end up owning wildly uneven
    # slices of the ring purely by hash luck. Virtual nodes fix this by
    # mapping each physical shard to many points on the ring, averaging
    # out the imbalance. This is the same technique used in DynamoDB/Cassandra.

    def __init__(self, virtual_nodes_per_shard=150):
        self._virtual_nodes_per_shard = virtual_nodes_per_shard
        self._ring = {}
        self._sorted_keys = []

    def _hash(self, key):
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_shard(self, shard_id):
        for i in range(self._virtual_nodes_per_shard):
            vnode_key = shard_id + "#vnode" + str(i)
            hash_val = self._hash(vnode_key)
            self._ring[hash_val] = shard_id
            bisect.insort(self._sorted_keys, hash_val)

    def remove_shard(self, shard_id):
        to_remove = [i for i in range(self._virtual_nodes_per_shard)]
        for i in to_remove:
            vnode_key = shard_id + "#vnode" + str(i)
            hash_val = self._hash(vnode_key)
            if hash_val in self._ring:
                del self._ring[hash_val]
                idx = bisect.bisect_left(self._sorted_keys, hash_val)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == hash_val:
                    self._sorted_keys.pop(idx)

    def get_shard(self, item_key):
        if not self._ring:
            raise RuntimeError("no shards registered in the ring")

        hash_val = self._hash(item_key)
        idx = bisect.bisect(self._sorted_keys, hash_val)
        if idx == len(self._sorted_keys):
            idx = 0
        return self._ring[self._sorted_keys[idx]]
