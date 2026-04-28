"""ProvenanceTensor — NumPy ndarray subclass with lineage tracking."""

from __future__ import annotations

import numpy as np

from traceprop.granularity import Granularity, get_granularity
import traceprop.graph as _graph_mod
from traceprop.interceptor import record_op

# NumPy functions we handle via __array_function__
HANDLED_FUNCTIONS: dict = {}


def implements(np_function):
    """Register an __array_function__ implementation."""
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


_ndarray_ufunc = np.ndarray.__array_ufunc__


class ProvenanceTensor(np.ndarray):
    """NumPy ndarray subclass that tracks computational lineage."""

    _provenance_node_id: int | None
    _source_id: str | None

    def __new__(cls, input_array, node_id: int | None = None, source_id: str | None = None):
        obj = np.asarray(input_array).view(cls)
        obj._source_id = source_id
        if node_id is not None:
            obj._provenance_node_id = node_id
        elif get_granularity() >= Granularity.OP:
            try:
                node = _graph_mod.TensorNode(shape=obj.shape, dtype=str(obj.dtype))
                _graph_mod._global_graph.add_node(node)
                obj._provenance_node_id = node.id
            except Exception:
                obj._provenance_node_id = None
        else:
            obj._provenance_node_id = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._provenance_node_id = getattr(obj, "_provenance_node_id", None)
        self._source_id = getattr(obj, "_source_id", None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Fast path for common binary ufuncs (2 inputs, no out)
        n = len(inputs)
        if out is None:
            if n == 2:
                a, b = inputs
                ra = a.view(np.ndarray) if type(a) is ProvenanceTensor else a
                rb = b.view(np.ndarray) if type(b) is ProvenanceTensor else b
                result = _ndarray_ufunc(self, ufunc, method, ra, rb, **kwargs)
            elif n == 1:
                a = inputs[0]
                ra = a.view(np.ndarray) if type(a) is ProvenanceTensor else a
                result = _ndarray_ufunc(self, ufunc, method, ra, **kwargs)
            else:
                raw_inputs = tuple(
                    x.view(np.ndarray) if type(x) is ProvenanceTensor else x
                    for x in inputs
                )
                result = _ndarray_ufunc(self, ufunc, method, *raw_inputs, **kwargs)
        else:
            raw_inputs = tuple(
                x.view(np.ndarray) if type(x) is ProvenanceTensor else x
                for x in inputs
            )
            raw_out = tuple(
                x.view(np.ndarray) if type(x) is ProvenanceTensor else x
                for x in out
            )
            result = _ndarray_ufunc(self, ufunc, method, *raw_inputs, out=raw_out, **kwargs)

        if result is NotImplemented:
            return NotImplemented

        op_name = ufunc.__name__
        if isinstance(result, np.ndarray):
            node_id = record_op(op_name, inputs, result)
            result = result.view(ProvenanceTensor)
            result._provenance_node_id = node_id
        elif isinstance(result, tuple):
            wrapped = []
            for r in result:
                if isinstance(r, np.ndarray):
                    node_id = record_op(op_name, inputs, r)
                    r = r.view(ProvenanceTensor)
                    r._provenance_node_id = node_id
                wrapped.append(r)
            result = tuple(wrapped)

        return result

    def __array_function__(self, func, types, args, kwargs):
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        # Fallback: run the function, then wrap the result
        raw_args = _strip_args(args)
        result = func(*raw_args, **kwargs)
        if isinstance(result, np.ndarray):
            all_inputs = _collect_inputs(args)
            node_id = record_op(func.__name__, tuple(all_inputs), result)
            result = result.view(ProvenanceTensor)
            result._provenance_node_id = node_id
        elif np.isscalar(result) or (isinstance(result, np.generic)):
            # Wrap scalar results (e.g. from np.sum on 1D)
            arr = np.asarray(result)
            all_inputs = _collect_inputs(args)
            node_id = record_op(func.__name__, tuple(all_inputs), arr)
            result = arr.view(ProvenanceTensor)
            result._provenance_node_id = node_id
        return result


def _strip_args(args):
    """Recursively strip ProvenanceTensor from args."""
    out = []
    for a in args:
        if type(a) is ProvenanceTensor:
            out.append(a.view(np.ndarray))
        elif isinstance(a, (list, tuple)):
            out.append(type(a)(_strip_args(a)))
        else:
            out.append(a)
    return out


def _collect_inputs(args) -> list:
    """Collect all ProvenanceTensor inputs from args (single pass)."""
    found = []
    for a in args:
        if type(a) is ProvenanceTensor:
            found.append(a)
        elif isinstance(a, (list, tuple)):
            found.extend(_collect_inputs(a))
    return found


# Register common numpy functions
@implements(np.concatenate)
def _concatenate(arrays, axis=0, out=None, **kwargs):
    raw = [a.view(np.ndarray) if type(a) is ProvenanceTensor else a for a in arrays]
    result = np.concatenate(raw, axis=axis, out=out, **kwargs)
    all_inputs = [a for a in arrays if type(a) is ProvenanceTensor]
    node_id = record_op("concatenate", tuple(all_inputs), result)
    result = result.view(ProvenanceTensor)
    result._provenance_node_id = node_id
    return result


@implements(np.stack)
def _stack(arrays, axis=0, out=None, **kwargs):
    raw = [a.view(np.ndarray) if type(a) is ProvenanceTensor else a for a in arrays]
    result = np.stack(raw, axis=axis, out=out, **kwargs)
    all_inputs = [a for a in arrays if type(a) is ProvenanceTensor]
    node_id = record_op("stack", tuple(all_inputs), result)
    result = result.view(ProvenanceTensor)
    result._provenance_node_id = node_id
    return result


@implements(np.reshape)
def _reshape(a, *args, **kwargs):
    raw = a.view(np.ndarray) if type(a) is ProvenanceTensor else a
    result = np.reshape(raw, *args, **kwargs)
    node_id = record_op("reshape", (a,), result)
    result = result.view(ProvenanceTensor)
    result._provenance_node_id = node_id
    return result
