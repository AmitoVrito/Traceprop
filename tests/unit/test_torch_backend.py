"""Tests for PyTorch backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from traceprop.backends.torch_backend import ProvenanceTorchTensor, torch_tensor
from traceprop.graph import get_graph, reset_graph


@pytest.fixture(autouse=True)
def _clean_graph():
    reset_graph()
    yield
    reset_graph()


class TestProvenanceTorchTensor:
    def test_creation(self):
        t = torch_tensor([1.0, 2.0, 3.0])
        assert isinstance(t, ProvenanceTorchTensor)
        assert t._provenance_node_id is not None

    def test_source_id(self):
        t = torch_tensor([1.0], source_id="batch-1")
        assert t._source_id == "batch-1"

    def test_provenance_view(self):
        t = torch_tensor([1.0, 2.0])
        v = t.provenance
        assert v.node_id == t._provenance_node_id

    def test_node_in_graph(self):
        t = torch_tensor([1.0, 2.0])
        graph = get_graph()
        assert t._provenance_node_id in graph.nodes

    def test_operation_tracking(self):
        a = torch_tensor([1.0, 2.0])
        b = torch_tensor([3.0, 4.0])
        c = a + b
        graph = get_graph()
        assert len(graph.nodes) >= 3
