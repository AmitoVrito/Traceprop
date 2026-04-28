"""ProvRC range compression for lineage graphs."""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Iterator

from traceprop.graph import LineageGraph, OpEdge


_RANGE_OPS = frozenset({
    "matmul", "dot", "einsum", "tensordot", "inner", "outer", "conv2d",
})

_MIN_DIM_FOR_RANGE = 100


@dataclasses.dataclass(slots=True)
class RangeDescriptor:
    """Describes a full-range dependency between input and output tensors."""

    op_name: str
    input_node_id: int
    output_node_id: int
    input_shape: tuple
    output_shape: tuple
    full_range: bool = True


class CompressedLineageGraph:
    """Wraps a LineageGraph and stores range descriptors for compressed ops."""

    __slots__ = ("_graph", "_range_descriptors")

    def __init__(self, graph: LineageGraph):
        self._graph = graph
        self._range_descriptors: dict[str, RangeDescriptor] = {}

    @property
    def graph(self) -> LineageGraph:
        return self._graph

    def add_range_descriptor(self, desc: RangeDescriptor) -> str:
        """Add a range descriptor and return its key."""
        key = f"{desc.input_node_id}->{desc.output_node_id}"
        self._range_descriptors[key] = desc
        return key

    def ancestors_compressed(self, node_id: int) -> set[int]:
        """Find ancestors, collapsing range-compressed edges."""
        visited: set[int] = set()
        queue: deque[int] = deque([node_id])

        # Build a reverse lookup: output_node_id -> list[RangeDescriptor]
        rd_by_output: dict[int, list[RangeDescriptor]] = {}
        for desc in self._range_descriptors.values():
            rd_by_output.setdefault(desc.output_node_id, []).append(desc)

        while queue:
            nid = queue.popleft()
            # Check range descriptors first
            for desc in rd_by_output.get(nid, ()):
                if desc.input_node_id not in visited:
                    visited.add(desc.input_node_id)
                    queue.append(desc.input_node_id)
            # Then check normal edges
            for eid in self._graph._backward.get(nid, ()):
                edge = self._graph.edges[eid]
                for inp_id in edge.input_ids:
                    if inp_id not in visited:
                        visited.add(inp_id)
                        queue.append(inp_id)
        return visited

    def stats(self) -> dict:
        """Return statistics about the compressed graph."""
        graph_stats = self._graph.stats()
        graph_stats["range_descriptors"] = len(self._range_descriptors)
        return graph_stats


def should_use_range_encoding(op_name: str, input_shapes: list[tuple]) -> bool:
    """Return True if the operation should use range encoding."""
    if op_name not in _RANGE_OPS:
        return False
    for shape in input_shapes:
        if any(d > _MIN_DIM_FOR_RANGE for d in shape):
            return True
    return False


def build_range_descriptor(
    op_name: str,
    input_node_id: int,
    output_node_id: int,
    input_shape: tuple,
    output_shape: tuple,
) -> RangeDescriptor:
    """Build a full-range descriptor for a compressed operation."""
    return RangeDescriptor(
        op_name=op_name,
        input_node_id=input_node_id,
        output_node_id=output_node_id,
        input_shape=input_shape,
        output_shape=output_shape,
        full_range=True,
    )
