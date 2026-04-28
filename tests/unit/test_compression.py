"""Tests for ProvRC range compression."""

from __future__ import annotations

import pytest

from traceprop.graph import LineageGraph, TensorNode, OpEdge, reset_graph
from traceprop.compression import (
    RangeDescriptor,
    CompressedLineageGraph,
    build_range_descriptor,
    should_use_range_encoding,
)


@pytest.fixture(autouse=True)
def _clean_graph():
    reset_graph()
    yield
    reset_graph()


class TestShouldUseRangeEncoding:
    def test_matmul_large(self):
        assert should_use_range_encoding("matmul", [(200, 300)]) is True

    def test_dot_large(self):
        assert should_use_range_encoding("dot", [(101,)]) is True

    def test_matmul_small(self):
        assert should_use_range_encoding("matmul", [(10, 10)]) is False

    def test_non_range_op(self):
        assert should_use_range_encoding("add", [(200, 300)]) is False

    def test_conv2d(self):
        assert should_use_range_encoding("conv2d", [(1, 3, 224, 224)]) is True

    def test_einsum(self):
        assert should_use_range_encoding("einsum", [(50, 50)]) is False

    def test_einsum_large(self):
        assert should_use_range_encoding("einsum", [(50, 200)]) is True


class TestRangeDescriptor:
    def test_creation(self):
        desc = RangeDescriptor(
            op_name="matmul",
            input_node_id=0,
            output_node_id=1,
            input_shape=(200, 300),
            output_shape=(200, 400),
        )
        assert desc.op_name == "matmul"
        assert desc.input_node_id == 0
        assert desc.output_node_id == 1
        assert desc.full_range is True

    def test_build_range_descriptor(self):
        desc = build_range_descriptor("dot", 5, 10, (200,), (1,))
        assert desc.op_name == "dot"
        assert desc.input_node_id == 5
        assert desc.output_node_id == 10
        assert desc.full_range is True


class TestCompressedLineageGraph:
    def _make_graph_with_chain(self):
        """Create a graph: n0 -> edge -> n1 -> edge -> n2."""
        g = LineageGraph()
        n0 = TensorNode(shape=(200, 300), dtype="float64")
        n1 = TensorNode(shape=(200, 400), dtype="float64")
        n2 = TensorNode(shape=(400,), dtype="float64")
        g.add_node(n0)
        g.add_node(n1)
        g.add_node(n2)
        e0 = OpEdge("matmul", (n0.id,), n1.id)
        e1 = OpEdge("sum", (n1.id,), n2.id)
        g.add_edge(e0)
        g.add_edge(e1)
        return g, n0, n1, n2

    def test_add_range_descriptor(self):
        g = LineageGraph()
        cg = CompressedLineageGraph(g)
        desc = build_range_descriptor("matmul", 0, 1, (200, 300), (200, 400))
        key = cg.add_range_descriptor(desc)
        assert key == "0->1"
        assert len(cg._range_descriptors) == 1

    def test_ancestors_compressed(self):
        g, n0, n1, n2 = self._make_graph_with_chain()
        cg = CompressedLineageGraph(g)
        # Add a range descriptor bypassing the normal edge for n0->n1
        desc = build_range_descriptor("matmul", n0.id, n1.id, n0.shape, n1.shape)
        cg.add_range_descriptor(desc)
        ancestors = cg.ancestors_compressed(n2.id)
        assert n1.id in ancestors
        assert n0.id in ancestors

    def test_stats(self):
        g, n0, n1, n2 = self._make_graph_with_chain()
        cg = CompressedLineageGraph(g)
        desc = build_range_descriptor("matmul", n0.id, n1.id, n0.shape, n1.shape)
        cg.add_range_descriptor(desc)
        stats = cg.stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2
        assert stats["range_descriptors"] == 1

    def test_graph_property(self):
        g = LineageGraph()
        cg = CompressedLineageGraph(g)
        assert cg.graph is g
