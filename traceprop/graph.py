"""Lineage graph with performance-optimized node/edge types."""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict, deque
from typing import Iterator

from traceprop._c_ext import fast_bfs_ancestors as _fast_bfs_ancestors

_node_counter = itertools.count()
_edge_counter = itertools.count()


class TensorNode:
    """Represents a tensor in the lineage graph."""

    __slots__ = ("id", "shape", "dtype", "timestamp", "meta")

    def __init__(self, shape: tuple, dtype: str, meta: dict | None = None):
        self.id: int = next(_node_counter)
        self.shape = shape
        self.dtype = dtype
        self.timestamp: int = time.monotonic_ns()
        self.meta = meta

    def __repr__(self) -> str:
        return f"TensorNode(id={self.id}, shape={self.shape}, dtype={self.dtype})"


class OpEdge:
    """Represents an operation connecting input tensors to an output tensor."""

    __slots__ = ("id", "op_name", "input_ids", "output_id", "timestamp", "meta", "source_ids")

    def __init__(self, op_name: str, input_ids: tuple[int, ...], output_id: int, meta: dict | None = None, source_ids: tuple[str, ...] = ()):
        self.id: int = next(_edge_counter)
        self.op_name = op_name
        self.input_ids = input_ids
        self.output_id = output_id
        self.timestamp: int = time.monotonic_ns()
        self.meta = meta
        self.source_ids = source_ids

    def __repr__(self) -> str:
        return f"OpEdge(id={self.id}, op={self.op_name}, {self.input_ids}->{self.output_id})"


class LineageGraph:
    """DAG storing tensor nodes and operation edges."""

    __slots__ = ("nodes", "edges", "_forward", "_backward", "_lock")

    def __init__(self):
        self.nodes: dict[int, TensorNode] = {}
        self.edges: dict[int, OpEdge] = {}
        self._forward: dict[int, list[int]] = defaultdict(list)   # node_id -> [edge_ids producing from it]
        self._backward: dict[int, list[int]] = defaultdict(list)  # node_id -> [edge_ids producing it]
        self._lock = threading.Lock()

    def add_node(self, node: TensorNode) -> None:
        with self._lock:
            self.nodes[node.id] = node

    def add_edge(self, edge: OpEdge) -> None:
        with self._lock:
            self.edges[edge.id] = edge
            for inp_id in edge.input_ids:
                self._forward[inp_id].append(edge.id)
            self._backward[edge.output_id].append(edge.id)

    def ancestors(self, node_id: int) -> set[int]:
        """BFS backward to find all ancestor node IDs."""
        if _fast_bfs_ancestors is not None:
            return _fast_bfs_ancestors(dict(self._backward), self.edges, node_id)
        visited: set[int] = set()
        queue: deque[int] = deque([node_id])
        while queue:
            nid = queue.popleft()
            for eid in self._backward.get(nid, ()):
                edge = self.edges[eid]
                for inp_id in edge.input_ids:
                    if inp_id not in visited:
                        visited.add(inp_id)
                        queue.append(inp_id)
        return visited

    def descendants(self, node_id: int) -> set[int]:
        """BFS forward to find all descendant node IDs."""
        visited: set[int] = set()
        queue: deque[int] = deque([node_id])
        while queue:
            nid = queue.popleft()
            for eid in self._forward.get(nid, ()):
                edge = self.edges[eid]
                out = edge.output_id
                if out not in visited:
                    visited.add(out)
                    queue.append(out)
        return visited

    def edges_for_node(self, node_id: int) -> Iterator[OpEdge]:
        """Yield all edges where node_id is an input or output."""
        for eid in self._backward.get(node_id, ()):
            yield self.edges[eid]
        for eid in self._forward.get(node_id, ()):
            yield self.edges[eid]

    def stats(self) -> dict:
        """Return graph statistics."""
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._forward.clear()
        self._backward.clear()


_global_graph = LineageGraph()


def get_graph() -> LineageGraph:
    return _global_graph


def reset_graph() -> None:
    global _global_graph
    _global_graph = LineageGraph()
