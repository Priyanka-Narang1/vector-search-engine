import hnswlib
import numpy as np
from app.core.rwlock import ReadWriteLock


class VectorIndex:
    def __init__(self, dim: int, max_elements: int = 100_000, space: str = "cosine"):
        # Fixed max_elements is an hnswlib constraint, not a design choice -
        # the index must be resized explicitly if this cap is hit (see insert()).
        self._dim = dim
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self._index.set_ef(50)
        self._next_id = 0
        self._id_to_metadata = {}
        self._lock = ReadWriteLock()

    def insert(self, vector, metadata):
        if vector.shape[0] != self._dim:
            raise ValueError(
                "vector dim " + str(vector.shape[0]) + " does not match index dim " + str(self._dim)
            )

        self._lock.acquire_write()
        try:
            if self._next_id >= self._index.get_max_elements():
                # Growing mid-run is expensive (rebuilds internal graph) - flagging
                # loudly rather than silently resizing, since it signals max_elements
                # was undersized for the dataset.
                raise RuntimeError(
                    "index capacity reached - increase max_elements at construction time"
                )

            item_id = self._next_id
            self._index.add_items(vector, item_id)
            self._id_to_metadata[item_id] = metadata
            self._next_id += 1
            return item_id
        finally:
            self._lock.release_write()

    def search(self, query_vector, k=5):
        if query_vector.shape[0] != self._dim:
            raise ValueError(
                "query dim " + str(query_vector.shape[0]) + " does not match index dim " + str(self._dim)
            )

        self._lock.acquire_read()
        try:
            if self._next_id == 0:
                # Empty index - hnswlib raises on knn_query against zero items,
                # so this is handled explicitly rather than left to bubble up.
                return []

            # Can't request more neighbors than exist yet.
            effective_k = min(k, self._next_id)
            labels, distances = self._index.knn_query(query_vector, k=effective_k)

            results = []
            for label, distance in zip(labels[0], distances[0]):
                results.append({
                    "id": int(label),
                    "score": float(distance),
                    "metadata": self._id_to_metadata.get(int(label), {}),
                })
            return results
        finally:
            self._lock.release_read()
