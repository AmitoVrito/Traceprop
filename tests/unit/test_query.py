"""Tests for ProvenanceView query API."""

import numpy as np
import pytest

import traceprop as tp
from traceprop.graph import reset_graph


@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


class TestProvenanceView:
    def test_node(self):
        a = tp.array([1, 2, 3])
        v = tp.provenance(a)
        assert v.node is not None
        assert v.node_id == a._provenance_node_id

    def test_sources_after_op(self):
        a = tp.array([1.0, 2.0])
        b = tp.array([3.0, 4.0])
        c = a + b
        v = tp.provenance(c)
        sources = v.sources()
        source_ids = {s.id for s in sources}
        assert a._provenance_node_id in source_ids
        assert b._provenance_node_id in source_ids

    def test_ops(self):
        a = tp.array([1.0])
        b = a * 2
        v = tp.provenance(b)
        ops = v.ops()
        assert len(ops) >= 1
        assert ops[0].op_name == "multiply"

    def test_ancestors(self):
        a = tp.array([1.0])
        b = tp.array([2.0])
        c = a + b
        d = c * a
        v = tp.provenance(d)
        anc = v.ancestors()
        assert a._provenance_node_id in anc
        assert b._provenance_node_id in anc
        assert c._provenance_node_id in anc

    def test_descendants(self):
        a = tp.array([1.0])
        b = a + a
        c = b * 2
        v = tp.provenance(a)
        desc = v.descendants()
        assert b._provenance_node_id in desc
        assert c._provenance_node_id in desc

    def test_source_ids_in_path(self):
        a = tp.array([1.0], source_id="batch_A")
        b = tp.array([2.0], source_id="batch_B")
        c = a + b
        d = c * a
        v = tp.provenance(d)
        sids = v.source_ids_in_path()
        assert "batch_A" in sids
        assert "batch_B" in sids

    def test_source_ids_in_path_empty(self):
        a = tp.array([1.0])
        b = a + a
        v = tp.provenance(b)
        sids = v.source_ids_in_path()
        assert sids == set()

    def test_no_provenance(self):
        tp.set_granularity(tp.Granularity.NONE)
        a = tp.array([1])
        v = tp.provenance(a)
        assert v.node is None
        assert v.sources() == []
        assert v.ancestors() == set()
        tp.set_granularity(tp.Granularity.TENSOR)
