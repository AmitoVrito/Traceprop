"""Tests for store backends."""

import pytest

from traceprop.graph import OpEdge, TensorNode, reset_graph
from traceprop.stores.memory_store import MemoryStore
from traceprop.stores.sqlite_store import SQLiteStore


@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


class TestMemoryStore:
    def test_save_and_get_node(self):
        store = MemoryStore()
        n = TensorNode((3,), "float64")
        store.save_node(n)
        assert store.get_node(n.id) is n

    def test_save_and_get_edge(self):
        store = MemoryStore()
        e = OpEdge("mul", (0,), 1)
        store.save_edge(e)
        assert store.get_edge(e.id) is e

    def test_clear(self):
        store = MemoryStore()
        store.save_node(TensorNode((1,), "f"))
        store.clear()
        assert len(store.all_nodes()) == 0


class TestSQLiteStore:
    def test_roundtrip_node(self):
        store = SQLiteStore()
        n = TensorNode((2, 3), "float32")
        store.save_node(n)
        loaded = store.get_node(n.id)
        assert loaded is not None
        assert loaded.id == n.id
        assert loaded.shape == n.shape
        assert loaded.dtype == n.dtype
        store.close()

    def test_roundtrip_edge(self):
        store = SQLiteStore()
        e = OpEdge("add", (10, 20), 30)
        store.save_edge(e)
        loaded = store.get_edge(e.id)
        assert loaded is not None
        assert loaded.op_name == "add"
        assert loaded.input_ids == (10, 20)
        assert loaded.output_id == 30
        store.close()

    def test_batch_nodes(self):
        store = SQLiteStore()
        nodes = [TensorNode((i,), "f") for i in range(10)]
        store.save_nodes_batch(nodes)
        assert len(store.all_nodes()) == 10
        store.close()

    def test_batch_edges(self):
        store = SQLiteStore()
        edges = [OpEdge(f"op{i}", (i,), i + 100) for i in range(5)]
        store.save_edges_batch(edges)
        assert len(store.all_edges()) == 5
        store.close()

    def test_clear(self):
        store = SQLiteStore()
        store.save_node(TensorNode((1,), "f"))
        store.clear()
        assert len(store.all_nodes()) == 0
        store.close()
