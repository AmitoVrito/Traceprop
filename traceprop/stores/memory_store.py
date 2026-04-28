"""In-memory store backend."""

from __future__ import annotations

from traceprop.graph import LineageGraph, OpEdge, TensorNode


class MemoryStore:
    """Stores lineage data in memory (wraps a LineageGraph)."""

    __slots__ = ("_graph",)

    def __init__(self, graph: LineageGraph | None = None):
        self._graph = graph or LineageGraph()

    @property
    def graph(self) -> LineageGraph:
        return self._graph

    def save_node(self, node: TensorNode) -> None:
        self._graph.add_node(node)

    def save_edge(self, edge: OpEdge) -> None:
        self._graph.add_edge(edge)

    def get_node(self, node_id: int) -> TensorNode | None:
        return self._graph.nodes.get(node_id)

    def get_edge(self, edge_id: int) -> OpEdge | None:
        return self._graph.edges.get(edge_id)

    def all_nodes(self) -> list[TensorNode]:
        return list(self._graph.nodes.values())

    def all_edges(self) -> list[OpEdge]:
        return list(self._graph.edges.values())

    def clear(self) -> None:
        self._graph.clear()
