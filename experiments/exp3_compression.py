"""Experiment 3: ProvRC Compression Ratio for matmul operations."""

import numpy as np
import traceprop as tp
from traceprop.graph import reset_graph, get_graph
import json

SIZES = [100, 250, 500, 1000]
results = {}

for n in SIZES:
    # Without compression
    reset_graph()
    A = tp.array(np.random.rand(n, n), source_id="A")
    B = tp.array(np.random.rand(n, n), source_id="B")
    C = A @ B
    uncompressed = get_graph().stats()

    # With compression
    reset_graph()
    cg = tp.enable_compression()
    A2 = tp.array(np.random.rand(n, n), source_id="A")
    B2 = tp.array(np.random.rand(n, n), source_id="B")
    C2 = A2 @ B2
    compressed = cg.stats()
    range_descs = compressed.get("range_descriptors", 0)

    uc_edges = uncompressed["edges"]
    c_total = compressed["edges"] + range_descs
    ratio = uc_edges / max(1, c_total)

    results[n] = {
        "uncompressed_edges": uc_edges,
        "compressed_edges": compressed["edges"],
        "range_descriptors": range_descs,
        "compression_ratio": round(ratio, 2),
    }
    print(
        f"n={n:5}: uncompressed={uc_edges} edges  "
        f"compressed={compressed['edges']} edges + {range_descs} descriptors  "
        f"ratio={ratio:.1f}x"
    )

with open("results/exp3_compression.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp3_compression.json")
