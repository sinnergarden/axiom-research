"""Internal lifecycle policy for the existing saved-Label preparation loop.

Call after releasing a shard's objects. Python's automatic cyclic collection
stays enabled; this only spaces explicit full collections. No Label values,
source admission, clocks or Core execution live here.
"""
import gc


class _LabelShardGC:
    def __init__(self, *, interval=32):
        if type(interval) is not int or not 1 <= interval <= 32:
            raise ValueError("Label GC interval must be an integer from 1 to 32")
        self.interval = interval
        self.pending = 0
        self.collections = 0
        self.closed = False

    def after_release(self, *, force=False):
        if self.closed:
            raise RuntimeError("Label preparation GC stage is closed")
        if type(force) is not bool:
            raise ValueError("force must be bool")
        self.pending += 1
        if force or self.pending >= self.interval:
            gc.collect()
            self.pending = 0
            self.collections += 1

    def close(self):
        if not self.closed:
            gc.collect()
            self.pending = 0
            self.collections += 1
            self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Label preparation GC stage is closed")
        return self

    def __exit__(self, *exc):
        self.close()
