"""Optional Cython-accelerated graph operations."""

try:
    from traceprop._c_ext.graph_ops import fast_bfs_ancestors, fast_collect_input_ids, fast_record_op
except ImportError:
    fast_bfs_ancestors = None
    fast_collect_input_ids = None
    fast_record_op = None
