"""JAX backend — TrackedJaxArray wrapper with provenance tracking."""

from __future__ import annotations

from typing import Any

from traceprop.exceptions import BackendNotInstalledError

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def is_available() -> bool:
    return _JAX_AVAILABLE


def _ensure_jax():
    if not _JAX_AVAILABLE:
        raise BackendNotInstalledError("JAX is not installed. Install with: pip install traceprop[jax]")


class TrackedJaxArray:
    """Wrapper around a JAX array with provenance tracking.

    JAX arrays cannot be subclassed, so we wrap them instead.
    """

    __slots__ = ("_data", "_provenance_node_id", "_source_id")

    def __init__(self, data, node_id: int | None = None, source_id: str | None = None):
        _ensure_jax()
        if isinstance(data, TrackedJaxArray):
            self._data = data._data
        else:
            self._data = jnp.asarray(data)
        self._source_id = source_id

        if node_id is not None:
            self._provenance_node_id = node_id
        else:
            from traceprop.granularity import Granularity, get_granularity
            import traceprop.graph as _graph_mod

            if get_granularity() >= Granularity.OP:
                try:
                    node = _graph_mod.TensorNode(shape=self._data.shape, dtype=str(self._data.dtype))
                    _graph_mod._global_graph.add_node(node)
                    self._provenance_node_id = node.id
                except Exception:
                    self._provenance_node_id = None
            else:
                self._provenance_node_id = None

    def _binop(self, other, op_name, jnp_fn):
        from traceprop.interceptor import record_op

        other_data = other._data if isinstance(other, TrackedJaxArray) else other
        result_data = jnp_fn(self._data, other_data)
        inputs = (self, other) if isinstance(other, TrackedJaxArray) else (self,)
        node_id = record_op(op_name, inputs, result_data)
        return TrackedJaxArray(result_data, node_id=node_id)

    def __add__(self, other):
        return self._binop(other, "add", jnp.add)

    def __sub__(self, other):
        return self._binop(other, "subtract", jnp.subtract)

    def __mul__(self, other):
        return self._binop(other, "multiply", jnp.multiply)

    def __truediv__(self, other):
        return self._binop(other, "true_divide", jnp.divide)

    def __matmul__(self, other):
        return self._binop(other, "matmul", jnp.matmul)

    def mean(self, axis=None):
        from traceprop.interceptor import record_op

        result_data = self._data.mean(axis=axis)
        node_id = record_op("mean", (self,), result_data)
        return TrackedJaxArray(result_data, node_id=node_id)

    def std(self, axis=None):
        from traceprop.interceptor import record_op

        result_data = self._data.std(axis=axis)
        node_id = record_op("std", (self,), result_data)
        return TrackedJaxArray(result_data, node_id=node_id)

    @property
    def provenance(self):
        from traceprop.query import ProvenanceView
        return ProvenanceView(self)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __repr__(self):
        return f"TrackedJaxArray(shape={self.shape}, dtype={self.dtype}, node_id={self._provenance_node_id})"


def jax_array(data: Any, source_id: str | None = None) -> TrackedJaxArray:
    """Create a TrackedJaxArray with provenance tracking."""
    _ensure_jax()
    return TrackedJaxArray(data, source_id=source_id)
