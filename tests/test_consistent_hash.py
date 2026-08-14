from collections import Counter
from app.sharding.consistent_hash import ConsistentHashRing


def test_keys_distribute_across_shards_reasonably_evenly():
    ring = ConsistentHashRing(virtual_nodes_per_shard=150)
    for shard_id in ["shard-0", "shard-1", "shard-2", "shard-3"]:
        ring.add_shard(shard_id)

    counts = Counter()
    num_keys = 10000
    for i in range(num_keys):
        shard = ring.get_shard("item-" + str(i))
        counts[shard] += 1

    expected_per_shard = num_keys / 4
    for shard_id, count in counts.items():
        deviation = abs(count - expected_per_shard) / expected_per_shard
        # With vnodes, deviation should stay well under 15% - without vnodes
        # this same test would show far worse skew on some runs.
        assert deviation < 0.15, (
            shard_id + " got " + str(count) + " keys, expected ~" + str(expected_per_shard)
        )


def test_adding_a_shard_only_remaps_expected_fraction_of_keys():
    ring = ConsistentHashRing(virtual_nodes_per_shard=150)
    for shard_id in ["shard-0", "shard-1", "shard-2"]:
        ring.add_shard(shard_id)

    num_keys = 5000
    before = {}
    for i in range(num_keys):
        key = "item-" + str(i)
        before[key] = ring.get_shard(key)

    ring.add_shard("shard-3")

    remapped = 0
    for i in range(num_keys):
        key = "item-" + str(i)
        after = ring.get_shard(key)
        if after != before[key]:
            remapped += 1

    remap_fraction = remapped / num_keys
    # Adding a 4th shard to 3 should move roughly 1/4 of keys - this IS
    # the actual point of consistent hashing (vs. naive mod-N hashing,
    # which would remap close to 100% of keys on any shard count change).
    assert 0.15 < remap_fraction < 0.40, (
        "expected ~25% of keys to remap, got " + str(remap_fraction * 100) + "%"
    )


def test_missing_ring_raises_not_silently_fails():
    ring = ConsistentHashRing()
    try:
        ring.get_shard("item-1")
        assert False, "expected RuntimeError for empty ring"
    except RuntimeError:
        pass

def test_naive_modulo_hashing_remaps_nearly_everything_vs_consistent_hashing():
    # Comparison test, not testing our ring's correctness again - this
    # demonstrates WHY consistent hashing matters at all. Naive hashing
    # (key_hash % num_shards) has no stability guarantee: changing the
    # shard count changes almost every key's assignment, because the
    # modulo operation depends on the total count directly.
    num_keys = 5000
    old_shard_count = 3
    new_shard_count = 4

    def naive_shard(key, num_shards):
        return hash(key) % num_shards

    naive_before = {}
    for i in range(num_keys):
        key = "item-" + str(i)
        naive_before[key] = naive_shard(key, old_shard_count)

    naive_remapped = 0
    for i in range(num_keys):
        key = "item-" + str(i)
        after = naive_shard(key, new_shard_count)
        if after != naive_before[key]:
            naive_remapped += 1

    naive_remap_fraction = naive_remapped / num_keys

    # Consistent hashing (already proven above) remaps ~25% on the same
    # shard-count change. Naive modulo should remap the vast majority -
    # this asserts the contrast directly, in one test, side by side.
    assert naive_remap_fraction > 0.60, (
        "expected naive hashing to remap most keys, got only "
        + str(naive_remap_fraction * 100) + "%"
    )
