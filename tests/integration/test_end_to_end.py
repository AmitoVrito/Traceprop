"""End-to-end integration tests."""

import tempfile

import numpy as np
import pytest

import traceprop as tp
from traceprop.graph import reset_graph
from traceprop.store import to_memory_store, to_sqlite


@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


def test_full_pipeline():
    """Create tensors, do ops, query lineage, store to SQLite."""
    # Create inputs
    x = tp.array([1.0, 2.0, 3.0])
    w = tp.array([0.5, 0.5, 0.5])

    # Compute
    y = x * w
    z = np.sum(y)

    # Query
    v = tp.provenance(z)
    assert v.node is not None
    anc = v.ancestors()
    assert x._provenance_node_id in anc
    assert w._provenance_node_id in anc

    # Store
    store = to_sqlite()
    nodes = store.all_nodes()
    edges = store.all_edges()
    assert len(nodes) >= 3  # x, w, y, z
    assert len(edges) >= 1
    store.close()


def test_csv_pipeline(tmp_path):
    """Load CSV, compute, query."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
    t = tp.from_csv(str(csv_file))
    assert t.shape == (2, 3)
    s = np.sum(t, axis=0)
    assert isinstance(s, tp.ProvenanceTensor)
    np.testing.assert_array_equal(s, [5, 7, 9])

    v = tp.provenance(s)
    sources = v.sources()
    assert len(sources) >= 1


def test_chained_operations():
    """Verify multi-step lineage chain."""
    a = tp.array([1.0, 2.0])
    b = a + 1
    c = b * 2
    d = np.sqrt(c)

    v = tp.provenance(d)
    anc = v.ancestors()
    assert a._provenance_node_id in anc
    assert b._provenance_node_id in anc
    assert c._provenance_node_id in anc


def test_memory_store():
    a = tp.array([1.0])
    b = a * 2
    store = to_memory_store()
    assert len(store.all_nodes()) >= 2


def test_enable_compression():
    """Test compression can be enabled and works end-to-end."""
    a = tp.array(np.random.randn(200, 200))
    b = tp.array(np.random.randn(200, 200))
    cg = tp.enable_compression()
    c = np.dot(a, b)
    stats = cg.stats()
    assert stats["range_descriptors"] >= 1


def test_version():
    assert tp.__version__ == "0.7.0"
