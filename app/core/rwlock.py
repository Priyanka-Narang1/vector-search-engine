import threading


class ReadWriteLock:
    # Standard readers-writer pattern: any number of readers can hold the
    # lock simultaneously, but a writer needs exclusive access. Built on
    # a plain Lock + counter rather than threading.RLock because RLock
    # doesn't distinguish reader vs writer intent - only one thread total
    # could hold it at a time, defeating the point of allowing concurrent reads.

    def __init__(self):
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._writer_lock = threading.Lock()

    def acquire_read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                # First reader blocks writers for the duration of any read activity.
                self._writer_lock.acquire()

    def release_read(self):
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                # Last reader releases the writer block.
                self._writer_lock.release()

    def acquire_write(self):
        self._writer_lock.acquire()

    def release_write(self):
        self._writer_lock.release()
