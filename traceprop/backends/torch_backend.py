"""PyTorch backend for provenance tracking."""

from __future__ import annotations

import functools
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from traceprop.exceptions import safe_provenance
from traceprop.granularity import Granularity, get_granularity
import traceprop.graph as _graph_mod
from traceprop.query import ProvenanceView


def _torch_available() -> bool:
    return torch is not None


class ProvenanceTorchTensor(torch.Tensor if torch is not None else object):  # type: ignore[misc]
    """A torch.Tensor subclass with provenance tracking."""

    _provenance_node_id: int | None
    _source_id: str | None

    @staticmethod
    def __new__(cls, data, *, source_id: str | None = None, node_id: int | None = None, **kwargs):
        if torch is None:
            raise ImportError("PyTorch is required for ProvenanceTorchTensor")
        if isinstance(data, torch.Tensor):
            instance = data.clone().detach().requires_grad_(data.requires_grad)
            instance.__class__ = cls
        else:
            instance = torch.Tensor._make_subclass(cls, torch.as_tensor(data))
        instance._provenance_node_id = node_id
        instance._source_id = source_id
        return instance

    def _ensure_node(self) -> int:
        """Lazily create a TensorNode if needed."""
        if self._provenance_node_id is not None:
            return self._provenance_node_id
        if get_granularity() < Granularity.OP:
            return -1
        node = _graph_mod.TensorNode(
            shape=tuple(self.shape),
            dtype=str(self.dtype),
            meta={"source_id": self._source_id} if self._source_id else None,
        )
        _graph_mod._global_graph.add_node(node)
        self._provenance_node_id = node.id
        return node.id

    @property
    def provenance(self) -> ProvenanceView:
        return ProvenanceView(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, torch.Tensor):
            result = _record_torch_op(func.__name__, args, result)
        return result


@safe_provenance
def _record_torch_op(op_name: str, inputs: tuple, result: torch.Tensor) -> ProvenanceTorchTensor:
    """Record a torch operation in the lineage graph."""
    if get_granularity() < Granularity.OP:
        if not isinstance(result, ProvenanceTorchTensor):
            out = ProvenanceTorchTensor(result)
            out._provenance_node_id = None
            return out
        return result

    # Collect input node IDs
    input_ids: list[int] = []
    source_ids_set: set[str] = set()
    for inp in _flatten_tensors(inputs):
        nid = getattr(inp, "_provenance_node_id", None)
        if nid is None:
            nid_val = getattr(inp, "_ensure_node", lambda: None)()
            if nid_val is not None and nid_val >= 0:
                nid = nid_val
        if nid is not None:
            input_ids.append(nid)
        sid = getattr(inp, "_source_id", None)
        if sid is not None:
            source_ids_set.add(sid)

    # Create output node
    out_node = _graph_mod.TensorNode(
        shape=tuple(result.shape),
        dtype=str(result.dtype),
    )
    graph = _graph_mod._global_graph
    graph.add_node(out_node)

    # Create edge
    if input_ids:
        source_ids = tuple(sorted(source_ids_set)) if source_ids_set else ()
        edge = _graph_mod.OpEdge(
            op_name=op_name,
            input_ids=tuple(input_ids),
            output_id=out_node.id,
            source_ids=source_ids,
        )
        graph.add_edge(edge)

    if not isinstance(result, ProvenanceTorchTensor):
        out = ProvenanceTorchTensor(result, node_id=out_node.id)
    else:
        out = result
        out._provenance_node_id = out_node.id
    return out


def _flatten_tensors(args) -> list:
    """Recursively extract tensors from args."""
    result = []
    if isinstance(args, (list, tuple)):
        for a in args:
            result.extend(_flatten_tensors(a))
    elif torch is not None and isinstance(args, torch.Tensor):
        result.append(args)
    return result


def torch_tensor(data: Any, source_id: str | None = None) -> ProvenanceTorchTensor:
    """Factory function to create a ProvenanceTorchTensor."""
    if torch is None:
        raise ImportError("PyTorch is required: pip install traceprop[torch]")
    t = ProvenanceTorchTensor(data, source_id=source_id)
    t._ensure_node()
    return t
