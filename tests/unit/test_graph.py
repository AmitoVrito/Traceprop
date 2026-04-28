"""Tests for the lineage graph."""

import pytest

from traceprop.graph import LineageGraph, OpEdge, TensorNode, reset_graph


@pytest.fixture
def graph():
    return LineageGraph()


class TestTensorNode:
    def test_creation(self):
        n = TensorNode(shape=(3, 4), dtype="float64")
        assert n.shape == (3, 4)
        assert n.dtype == "float64"
        assert isinstance(n.id, int)
        assert isinstance(n.timestamp, int)

    def test_unique_ids(self):
        a = TensorNode(shape=(1,), dtype="int64")
        b = TensorNode(shape=(2,), dtype="int64")
        assert a.id != b.id


class TestOpEdge:
    def test_creation(self):
        e = OpEdge(op_name="add", input_ids=(0, 1), output_id=2)
        assert e.op_name == "add"
        assert e.input_ids == (0, 1)
        assert e.output_id == 2
        assert e.source_ids == ()

    def test_source_ids(self):
        e = OpEdge(op_name="add", input_ids=(0, 1), output_id=2, source_ids=("batch_0", "batch_1"))
        assert e.source_ids == ("batch_0", "batch_1")


class TestLineageGraph:
    def test_add_and_retrieve(self, graph):
        n = TensorNode(shape=(5,), dtype="float32")
        graph.add_node(n)
        assert graph.nodes[n.id] is n

    def test_ancestors(self, graph):
        a = TensorNode((2,), "f"); b = TensorNode((2,), "f"); c = TensorNode((2,), "f")
        graph.add_node(a); graph.add_node(b); graph.add_node(c)
        e = OpEdge("add", (a.id, b.id), c.id)
        graph.add_edge(e)
        anc = graph.ancestors(c.id)
        assert a.id in anc
        assert b.id in anc
        assert c.id not in anc

    def test_descendants(self, graph):
        a = TensorNode((2,), "f"); b = TensorNode((2,), "f")
        graph.add_node(a); graph.add_node(b)
        e = OpEdge("neg", (a.id,), b.id)
        graph.add_edge(e)
        desc = graph.descendants(a.id)
        assert b.id in desc

    def test_clear(self, graph):
        n = TensorNode((1,), "f")
        graph.add_node(n)
        graph.clear()
        assert len(graph.nodes) == 0

    def test_ancestors_python_fallback(self, graph):
        """Test the pure-Python BFS path directly."""
        import traceprop.graph as gmod
        orig = gmod._fast_bfs_ancestors
        try:
            gmod._fast_bfs_ancestors = None
            a = TensorNode((2,), "f"); b = TensorNode((2,), "f"); c = TensorNode((2,), "f")
            graph.add_node(a); graph.add_node(b); graph.add_node(c)
            e = OpEdge("add", (a.id, b.id), c.id)
            graph.add_edge(e)
            anc = graph.ancestors(c.id)
            assert a.id in anc
            assert b.id in anc
        finally:
            gmod._fast_bfs_ancestors = orig

    def test_stats(self, graph):
        a = TensorNode((2,), "f"); b = TensorNode((2,), "f")
        graph.add_node(a); graph.add_node(b)
        e = OpEdge("neg", (a.id,), b.id)
        graph.add_edge(e)
        s = graph.stats()
        assert s["nodes"] == 2
        assert s["edges"] == 1

    def test_edges_for_node(self, graph):
        a = TensorNode((2,), "f"); b = TensorNode((2,), "f")
        graph.add_node(a); graph.add_node(b)
        e = OpEdge("neg", (a.id,), b.id)
        graph.add_edge(e)
        edges = list(graph.edges_for_node(a.id))
        assert len(edges) == 1
        assert edges[0].id == e.id
