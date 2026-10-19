"""Refittable axis-aligned bounds hierarchy for local 3D scene queries."""
from dataclasses import dataclass
import numpy as np


@dataclass
class _Node:
    low: np.ndarray
    high: np.ndarray
    ids: object = None
    left: object = None
    right: object = None
    parent: object = None


class BoundsIndex:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float).reshape(-1, 3).copy()
        self.high = np.asarray(high, dtype=float).reshape(-1, 3).copy()
        if self.low.shape != self.high.shape:
            raise ValueError('bounds shapes differ')
        self.leaves = {}
        self.last_visited = 0
        self.root = self._build(np.arange(len(self.low))) if len(self.low) else None

    def _envelope(self, ids):
        valid = ids[np.all(self.low[ids] <= self.high[ids], axis=1)]
        if not len(valid):
            return np.full(3, np.inf), np.full(3, -np.inf)
        return self.low[valid].min(axis=0), self.high[valid].max(axis=0)

    def _build(self, ids, parent=None):
        low, high = self._envelope(ids)
        node = _Node(low, high, parent=parent)
        if len(ids) <= 16:
            node.ids = ids
            for index in ids:
                self.leaves[int(index)] = node
        else:
            # Invalid/hidden bounds still receive leaves and can become active
            # through refitting, without rebuilding the whole hierarchy.
            centres = np.where(np.isfinite(self.low[ids]), self.low[ids], 0.)*.5
            centres += np.where(np.isfinite(self.high[ids]), self.high[ids], 0.)*.5
            axis = int(np.argmax(np.ptp(centres, axis=0)))
            middle = len(ids)//2
            order = np.argpartition(centres[:, axis], middle)
            node.left = self._build(ids[order[:middle]], node)
            node.right = self._build(ids[order[middle:]], node)
        return node

    def update(self, ids, low, high):
        ids = np.asarray(list(ids), dtype=int)
        if not len(ids):
            return
        self.low[ids], self.high[ids] = low, high
        dirty = {id(self.leaves[int(index)]): self.leaves[int(index)] for index in ids}
        for leaf in dirty.values():
            leaf.low, leaf.high = self._envelope(leaf.ids)
            node = leaf.parent
            while node is not None:
                node.low = np.minimum(node.left.low, node.right.low)
                node.high = np.maximum(node.left.high, node.right.high)
                node = node.parent

    def query(self, low, high):
        low, high = np.asarray(low), np.asarray(high)
        result = []
        stack = [self.root] if self.root is not None else []
        self.last_visited = 0
        while stack:
            node = stack.pop()
            self.last_visited += 1
            a, b = node.low, node.high
            if (a[0] > high[0] or a[1] > high[1] or a[2] > high[2]
                    or b[0] < low[0] or b[1] < low[1] or b[2] < low[2]):
                continue
            if node.ids is None:
                stack.extend((node.left, node.right))
            else:
                ids = node.ids
                valid = np.all(self.low[ids] <= self.high[ids], axis=1)
                valid &= np.all(self.low[ids] <= high, axis=1) & np.all(self.high[ids] >= low, axis=1)
                result.extend(ids[valid].tolist())
        return np.asarray(sorted(result), dtype=int)
