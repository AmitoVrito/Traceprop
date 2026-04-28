"""Tests for JAX backend (skipped if JAX not installed)."""

import pytest

jax = pytest.importorskip("jax")

import traceprop as tp
from traceprop.backends.jax_backend import TrackedJaxArray, jax_array
from traceprop.graph import reset_graph


@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


class TestTrackedJaxArray:
    def test_creation(self):
        a = jax_array([1.0, 2.0, 3.0])
        assert isinstance(a, TrackedJaxArray)
        assert a._provenance_node_id is not None
        assert a.shape == (3,)

    def test_add(self):
        a = jax_array([1.0, 2.0])
        b = jax_array([3.0, 4.0])
        c = a + b
        assert isinstance(c, TrackedJaxArray)
        assert c._provenance_node_id is not None

    def test_sub(self):
        a = jax_array([5.0, 6.0])
        b = jax_array([1.0, 2.0])
        c = a - b
        assert isinstance(c, TrackedJaxArray)

    def test_mul(self):
        a = jax_array([2.0, 3.0])
        b = jax_array([4.0, 5.0])
        c = a * b
        assert isinstance(c, TrackedJaxArray)

    def test_truediv(self):
        a = jax_array([10.0, 20.0])
        b = jax_array([2.0, 4.0])
        c = a / b
        assert isinstance(c, TrackedJaxArray)

    def test_matmul(self):
        a = jax_array([[1.0, 2.0], [3.0, 4.0]])
        b = jax_array([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        assert isinstance(c, TrackedJaxArray)
        assert c.shape == (2, 2)

    def test_mean(self):
        a = jax_array([1.0, 2.0, 3.0])
        m = a.mean()
        assert isinstance(m, TrackedJaxArray)

    def test_std(self):
        a = jax_array([1.0, 2.0, 3.0])
        s = a.std()
        assert isinstance(s, TrackedJaxArray)

    def test_provenance_property(self):
        a = jax_array([1.0, 2.0])
        b = jax_array([3.0, 4.0])
        c = a + b
        view = c.provenance
        sources = view.sources()
        source_ids = {s.id for s in sources}
        assert a._provenance_node_id in source_ids
        assert b._provenance_node_id in source_ids

    def test_source_id(self):
        a = jax_array([1.0, 2.0], source_id="batch_1")
        assert a._source_id == "batch_1"

    def test_from_jax(self):
        a = tp.from_jax([1.0, 2.0], source_id="src")
        assert isinstance(a, TrackedJaxArray)
        assert a._source_id == "src"
