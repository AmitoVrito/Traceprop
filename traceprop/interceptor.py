"""Hot-path operation recording. No decorators, minimal indirection."""

from __future__ import annotations

import itertools
import time

from traceprop.granularity import Granularity, get_granularity
import traceprop.graph as _graph_mod
from traceprop._c_ext import fast_collect_input_ids as _fast_collect
from traceprop._c_ext import fast_record_op as _fast_record_op

_dtype_cache: dict[object, str] = {}
_node_counter = _graph_mod._node_counter
_edge_counter = _graph_mod._edge_counter
_TensorNode = _graph_mod.TensorNode
_OpEdge = _graph_mod.OpEdge

_compressed_graph = None  # Set by enable_compression()


def record_op(op_name: str, inputs: tuple, output_array, meta: dict | None = None) -> int | None:
    """Record an operation in the global lineage graph.

    Returns the output node ID, or None if tracking is disabled.
    This is the single hot-path entry point — kept minimal.
    """
    if get_granularity() < Granularity.OP:
        return None

    try:
        graph = _graph_mod._global_graph

        # Fast C path: delegate entire body when in OP mode and no compression
        # BATCH mode and above use the Python path for source_id propagation
        if _fast_record_op is not None and _compressed_graph is None and get_granularity() <= Granularity.OP:
            result = _fast_record_op(
                op_name, inputs, output_array,
                graph.nodes, graph.edges,
                graph._forward, graph._backward,
                _node_counter, _edge_counter,
                _dtype_cache, _TensorNode, _OpEdge, meta,
            )
            return result if result >= 0 else None

        # Fallback: pure Python path (also used when compression is active)

        # Single-pass: collect input node IDs from ProvenanceTensors
        if _fast_collect is not None:
            input_ids = _fast_collect(inputs)
        else:
            input_ids: list[int] = []
            for inp in inputs:
                nid = getattr(inp, "_provenance_node_id", None)
                if nid is not None:
                    input_ids.append(nid)

        # Collect source_ids from inputs
        source_ids_set: set[str] = set()
        for inp in inputs:
            sid = getattr(inp, "_source_id", None)
            if sid is not None:
                source_ids_set.add(sid)
        source_ids = tuple(sorted(source_ids_set)) if source_ids_set else ()

        # Create output node — cache dtype str conversion
        dt = output_array.dtype
        dtype_str = _dtype_cache.get(dt)
        if dtype_str is None:
            dtype_str = str(dt)
            _dtype_cache[dt] = dtype_str

        out_node = _TensorNode.__new__(_TensorNode)
        out_node.id = next(_node_counter)
        out_node.shape = output_array.shape
        out_node.dtype = dtype_str
        out_node.timestamp = time.monotonic_ns()
        out_node.meta = meta
        graph.nodes[out_node.id] = out_node

        # Check for range compression
        if _compressed_graph is not None and input_ids:
            from traceprop.compression import should_use_range_encoding, build_range_descriptor
            shapes = []
            for inp in inputs:
                s = getattr(inp, "shape", None)
                if s is not None:
                    shapes.append(s)
            if should_use_range_encoding(op_name, shapes):
                for inp_id in input_ids:
                    inp_node = graph.nodes.get(inp_id)
                    inp_shape = inp_node.shape if inp_node else ()
                    desc = build_range_descriptor(
                        op_name, inp_id, out_node.id, inp_shape, out_node.shape,
                    )
                    _compressed_graph.add_range_descriptor(desc)
                # Fall through to also create the normal edge

        # Create edge
        if input_ids:
            edge = _OpEdge.__new__(_OpEdge)
            edge.id = next(_edge_counter)
            edge.op_name = op_name
            edge.input_ids = tuple(input_ids)
            edge.output_id = out_node.id
            edge.timestamp = time.monotonic_ns()
            edge.meta = meta
            edge.source_ids = source_ids
            graph.edges[edge.id] = edge
            for inp_id in edge.input_ids:
                graph._forward[inp_id].append(edge.id)
            graph._backward[out_node.id].append(edge.id)

        return out_node.id
    except Exception:
        return None
