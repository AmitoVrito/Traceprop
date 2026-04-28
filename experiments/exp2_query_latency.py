"""Experiment 2: Lineage Query Latency vs graph depth."""

import time
import numpy as np
import traceprop as tp
from traceprop.graph import reset_graph, get_graph
import json

DEPTHS = [10, 50, 100, 500, 1000, 5000]
QUERY_ITERS = 100
results = {}

for depth in DEPTHS:
    reset_graph()
    a = tp.array(np.random.rand(1000), source_id="latency_test")
    for _ in range(depth):
        a = a * 1.001 + 0.001

    view = tp.provenance(a)
    graph = get_graph()
    graph_st = graph.stats()

    # Measure .sources() latency
    t0 = time.perf_counter()
    for _ in range(QUERY_ITERS):
        _ = view.sources()
    sources_ms = (time.perf_counter() - t0) / QUERY_ITERS * 1000

    # Measure .ops() latency
    t0 = time.perf_counter()
    for _ in range(QUERY_ITERS):
        _ = view.ops()
    ops_ms = (time.perf_counter() - t0) / QUERY_ITERS * 1000

    # Measure .ancestors() latency
    t0 = time.perf_counter()
    for _ in range(QUERY_ITERS):
        _ = view.ancestors()
    ancestors_ms = (time.perf_counter() - t0) / QUERY_ITERS * 1000

    results[depth] = {
        "graph_nodes": graph_st.get("nodes"),
        "graph_edges": graph_st.get("edges"),
        "sources_query_ms": round(sources_ms, 4),
        "ops_query_ms": round(ops_ms, 4),
        "ancestors_query_ms": round(ancestors_ms, 4),
    }
    print(
        f"depth={depth:5}  nodes={graph_st.get('nodes'):6}  "
        f"sources={sources_ms:.3f}ms  ops={ops_ms:.3f}ms  ancestors={ancestors_ms:.3f}ms"
    )

with open("results/exp2_query_latency.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp2_query_latency.json")
