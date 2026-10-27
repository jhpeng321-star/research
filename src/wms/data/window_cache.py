"""Bounded, per-loader observable-window cache; returned records remain caller-owned."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np


def nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nbytes(v) for v in value)
    return 0


class WindowCache:
    def __init__(self, limit=16 * 1024**2):
        self.limit, self.size = limit, 0
        self.items = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, load):
        if key in self.items:
            self.hits += 1
            value = self.items.pop(key)
            self.items[key] = value
            return deepcopy(value)
        self.misses += 1
        value = load()
        size = nbytes(value)
        if size <= self.limit:
            while self.items and self.size + size > self.limit:
                _, old = self.items.popitem(last=False)
                self.size -= nbytes(old)
            self.items[key] = deepcopy(value)
            self.size += size
        return value
