"""Thin facade over store backends."""

from __future__ import annotations

from traceprop.graph import get_graph
from traceprop.stores.memory_store import MemoryStore
from traceprop.stores.sqlite_store import SQLiteStore


def to_memory_store() -> MemoryStore:
    """Snapshot the current global graph into a MemoryStore."""
    return MemoryStore(get_graph())


def to_sqlite(db_path: str = ":memory:") -> SQLiteStore:
    """Export the current global graph to a SQLiteStore."""
    store = SQLiteStore(db_path)
    graph = get_graph()
    nodes = list(graph.nodes.values())
    edges = list(graph.edges.values())
    if nodes:
        store.save_nodes_batch(nodes)
    if edges:
        store.save_edges_batch(edges)
    return store
