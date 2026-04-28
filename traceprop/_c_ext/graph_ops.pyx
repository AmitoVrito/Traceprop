# cython: language_level=3
"""C-accelerated hot-path helpers for lineage graph operations."""

import time
from collections import deque


def fast_collect_input_ids(inputs):
    """Collect _provenance_node_id ints from inputs (C-accelerated)."""
    cdef list result = []
    for inp in inputs:
        nid = getattr(inp, "_provenance_node_id", None)
        if nid is not None:
            result.append(nid)
    return result


def fast_record_op(str op_name, tuple inputs, output_array,
                   graph_nodes, graph_edges,
                   graph_forward, graph_backward,
                   node_counter, edge_counter,
                   dtype_cache,
                   TensorNode_cls, OpEdge_cls, meta):
    """C-accelerated record_op hot path.

    Returns output node ID (int), or -1 on failure.
    """
    cdef list input_ids = []
    cdef set source_ids_set = set()
    cdef int nid_val
    cdef str sid

    # Fused single-pass over inputs: collect both node IDs and source IDs
    for inp in inputs:
        nid = getattr(inp, "_provenance_node_id", None)
        if nid is not None:
            input_ids.append(nid)
        s = getattr(inp, "_source_id", None)
        if s is not None:
            source_ids_set.add(s)

    source_ids = tuple(sorted(source_ids_set)) if source_ids_set else ()

    # Cache dtype string conversion
    dt = output_array.dtype
    dtype_str = dtype_cache.get(dt)
    if dtype_str is None:
        dtype_str = str(dt)
        dtype_cache[dt] = dtype_str

    # Single timestamp for both node and edge
    cdef long long ts = time.monotonic_ns()

    # Create output node via __new__ + direct slot assignment
    out_node = TensorNode_cls.__new__(TensorNode_cls)
    out_node.id = next(node_counter)
    out_node.shape = output_array.shape
    out_node.dtype = dtype_str
    out_node.timestamp = ts
    out_node.meta = meta
    graph_nodes[out_node.id] = out_node

    # Create edge (only if there are inputs)
    if input_ids:
        edge = OpEdge_cls.__new__(OpEdge_cls)
        edge.id = next(edge_counter)
        edge.op_name = op_name
        edge.input_ids = tuple(input_ids)
        edge.output_id = out_node.id
        edge.timestamp = ts
        edge.meta = meta
        edge.source_ids = source_ids
        graph_edges[edge.id] = edge
        for inp_id in edge.input_ids:
            graph_forward[inp_id].append(edge.id)
        graph_backward[out_node.id].append(edge.id)

    return out_node.id


def fast_bfs_ancestors(dict backward, dict edges, int start_id):
    """BFS backward through the lineage graph to find all ancestor node IDs.

    Parameters
    ----------
    backward : dict[int, list[int]]
        Mapping from node_id to list of edge_ids that produce it.
    edges : dict[int, object]
        Mapping from edge_id to OpEdge objects (must have .input_ids attribute).
    start_id : int
        The node ID to start BFS from.

    Returns
    -------
    set[int]
        All ancestor node IDs (not including start_id unless it's its own ancestor).
    """
    cdef set visited = set()
    queue = deque()
    queue.append(start_id)

    while queue:
        nid = queue.popleft()
        edge_ids = backward.get(nid)
        if edge_ids is None:
            continue
        for eid in edge_ids:
            edge = edges[eid]
            for inp_id in edge.input_ids:
                if inp_id not in visited:
                    visited.add(inp_id)
                    queue.append(inp_id)
    return visited
