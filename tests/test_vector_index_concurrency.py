import threading
import numpy as np
from app.index.vector_index import VectorIndex


def test_concurrent_inserts_without_lock_corrupt_state():
    # This test is EXPECTED TO FAIL on the current unlocked VectorIndex.
    # It exists to prove the race condition, not to pass yet.
    dim = 8
    index = VectorIndex(dim=dim, max_elements=5000)

    num_threads = 20
    inserts_per_thread = 50
    errors = []
    inserted_ids = []
    ids_lock = threading.Lock()  # only protects the test's own bookkeeping list, not the index

    def insert_worker(thread_id):
        rng = np.random.default_rng(seed=thread_id)
        for i in range(inserts_per_thread):
            try:
                vec = rng.random(dim).astype(np.float32)
                item_id = index.insert(vec, metadata={"thread": thread_id, "i": i})
                with ids_lock:
                    inserted_ids.append(item_id)
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=insert_worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_total = num_threads * inserts_per_thread
    unique_ids = set(inserted_ids)

    # Without a lock, _next_id increments are not atomic - two threads can
    # read the same _next_id before either writes it back, causing either
    # duplicate ids (both threads got id=N) or skipped ids (N is never assigned).
    # This assertion is expected to FAIL on the current unlocked code -
    # that failure IS the evidence.
    assert len(unique_ids) == expected_total, (
        f"expected {expected_total} unique ids, got {len(unique_ids)} "
        f"({expected_total - len(unique_ids)} lost to race condition), "
        f"errors during insert: {len(errors)}"
    )


def test_concurrent_read_write_without_lock_causes_errors_or_bad_results():
    # Runs searches WHILE inserts are happening, unlocked.
    # hnswlib's add_items and knn_query are not guaranteed safe to call
    # concurrently from multiple threads - this test surfaces that directly
    # rather than assuming it's fine.
    dim = 8
    index = VectorIndex(dim=dim, max_elements=5000)
    rng = np.random.default_rng(seed=1)

    # Seed some initial data so searches have something to find
    for _ in range(50):
        index.insert(rng.random(dim).astype(np.float32), metadata={})

    search_errors = []
    insert_errors = []
    stop_flag = threading.Event()

    def writer():
        w_rng = np.random.default_rng(seed=99)
        for _ in range(300):
            try:
                index.insert(w_rng.random(dim).astype(np.float32), metadata={})
            except Exception as e:
                insert_errors.append(str(e))
        stop_flag.set()

    def reader():
        r_rng = np.random.default_rng(seed=7)
        while not stop_flag.is_set():
            try:
                index.search(r_rng.random(dim).astype(np.float32), k=5)
            except Exception as e:
                search_errors.append(str(e))

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(5)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    writer_thread.join()
    for t in reader_threads:
        t.join()

    # Expected to FAIL (i.e. errors present) on the current unlocked code -
    # that's the point of this test right now.
    assert len(search_errors) == 0 and len(insert_errors) == 0, (
        f"concurrent read/write without locking produced "
        f"{len(search_errors)} search errors, {len(insert_errors)} insert errors"
    )
