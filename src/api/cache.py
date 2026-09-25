"""
MonsoonIQ In-Memory Inference Cache.
Provides low-latency response caching for frequently requested forecast dates and districts.
"""

from typing import Any, Optional
from collections import OrderedDict


class LRUCache:
    """Thread-safe fast in-memory LRU cache."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()
