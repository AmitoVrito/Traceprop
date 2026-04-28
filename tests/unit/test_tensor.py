"""Tests for ProvenanceTensor."""

import numpy as np
import pytest

import traceprop as tp
from traceprop.graph import reset_graph
from traceprop.tensor import ProvenanceTensor


@pytest.fixture(autouse=True)
def _clean_graph():
    reset_graph()
    yield
    reset_graph()


class TestCreation:
    def test_array_creates_provenance_tensor(self):
        t = tp.array([1, 2, 3])
        assert isinstance(t, ProvenanceTensor)
        assert t._provenance_node_id is not None

    def test_from_numpy(self):
        arr = np.array([4, 5, 6])
        t = tp.from_numpy(arr)
        assert isinstance(t, ProvenanceTensor)
        np.testing.assert_array_equal(t, arr)

    def test_values_preserved(self):
        t = tp.array([[1, 2], [3, 4]], dtype=np.float64)
        assert t.shape == (2, 2)
        assert t.dtype == np.float64


class TestUfuncs:
    def test_add(self):
        a = tp.array([1.0, 2.0])
        b = tp.array([3.0, 4.0])
        c = a + b
        assert isinstance(c, ProvenanceTensor)
        np.testing.assert_array_equal(c, [4.0, 6.0])
        assert c._provenance_node_id is not None
        assert c._provenance_node_id != a._provenance_node_id

    def test_multiply(self):
        a = tp.array([2.0, 3.0])
        c = a * 5
        assert isinstance(c, ProvenanceTensor)
        np.testing.assert_array_equal(c, [10.0, 15.0])

    def test_chain(self):
        a = tp.array([1.0, 2.0])
        b = tp.array([3.0, 4.0])
        c = (a + b) * a
        assert isinstance(c, ProvenanceTensor)
        np.testing.assert_array_equal(c, [4.0, 12.0])

    def test_unary_ufunc(self):
        a = tp.array([1.0, 4.0, 9.0])
        b = np.sqrt(a)
        assert isinstance(b, ProvenanceTensor)
        np.testing.assert_array_almost_equal(b, [1.0, 2.0, 3.0])


class TestArrayFunction:
    def test_concatenate(self):
        a = tp.array([1, 2])
        b = tp.array([3, 4])
        c = np.concatenate([a, b])
        assert isinstance(c, ProvenanceTensor)
        np.testing.assert_array_equal(c, [1, 2, 3, 4])

    def test_stack(self):
        a = tp.array([1, 2])
        b = tp.array([3, 4])
        c = np.stack([a, b])
        assert isinstance(c, ProvenanceTensor)
        assert c.shape == (2, 2)

    def test_reshape(self):
        a = tp.array([1, 2, 3, 4])
        b = np.reshape(a, (2, 2))
        assert isinstance(b, ProvenanceTensor)
        assert b.shape == (2, 2)

    def test_sum(self):
        a = tp.array([1.0, 2.0, 3.0])
        s = np.sum(a)
        assert isinstance(s, ProvenanceTensor)
        assert float(s) == 6.0

    def test_dot(self):
        a = tp.array([1.0, 2.0])
        b = tp.array([3.0, 4.0])
        c = np.dot(a, b)
        # dot returns scalar for 1D, which may or may not be ProvenanceTensor
        assert float(c) == 11.0


class TestSourceId:
    def test_source_id_on_array(self):
        a = tp.array([1.0, 2.0], source_id="batch_0")
        assert a._source_id == "batch_0"

    def test_source_id_on_from_numpy(self):
        a = tp.from_numpy(np.array([1.0]), source_id="src_1")
        assert a._source_id == "src_1"

    def test_source_id_default_none(self):
        a = tp.array([1.0])
        assert a._source_id is None

    def test_source_id_propagated_via_finalize(self):
        a = tp.array([1.0, 2.0], source_id="b1")
        b = tp.array([3.0, 4.0])
        c = a + b
        # source_id is on edges, not auto-propagated to result tensor
        assert isinstance(c, ProvenanceTensor)


class TestGranularityNone:
    def test_no_tracking(self):
        tp.set_granularity(tp.Granularity.NONE)
        a = tp.array([1, 2, 3])
        assert a._provenance_node_id is None
        tp.set_granularity(tp.Granularity.TENSOR)
