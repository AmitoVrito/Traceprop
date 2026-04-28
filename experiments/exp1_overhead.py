"""Experiment 1: Computation Lineage Overhead — op and batch modes vs vanilla NumPy."""

import time
import numpy as np
import traceprop as tp
from traceprop.graph import reset_graph
from traceprop.granularity import Granularity, set_granularity
import json

SIZES = [1_000, 10_000, 100_000, 1_000_000]
MODES = {
    "op": Granularity.OP,
    "batch": Granularity.BATCH,
}
ITERS = 200
WARMUP = 20


def measure(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000  # ms per iteration


results = {}

for n in SIZES:
    raw = np.random.rand(n).astype(np.float64)
    results[n] = {}

    # Vanilla NumPy baseline
    def vanilla(r=raw):
        _ = r * 2.0 + 1.0
        _ = r - r.mean()

    baseline_ms = measure(vanilla, ITERS, WARMUP)
    results[n]["vanilla_ms"] = round(baseline_ms, 4)

    for mode_name, mode_enum in MODES.items():
        reset_graph()
        pt_arr = tp.from_numpy(raw.copy())
        set_granularity(mode_enum)

        def tp_op(a=pt_arr):
            _ = a * 2.0 + 1.0
            _ = a - a.mean()

        tp_ms = measure(tp_op, ITERS, WARMUP)
        set_granularity(Granularity.TENSOR)  # restore default

        overhead = tp_ms / baseline_ms
        results[n][f"{mode_name}_ms"] = round(tp_ms, 4)
        results[n][f"{mode_name}_overhead"] = round(overhead, 3)
        print(
            f"n={n:>9}  mode={mode_name:6}  vanilla={baseline_ms:.3f}ms  "
            f"traceprop={tp_ms:.3f}ms  overhead={overhead:.3f}x"
        )

# Matmul test
print("\n--- Matmul overhead ---")
matmul_results = {}
for n in [100, 500, 1000]:
    A_raw = np.random.rand(n, n)
    B_raw = np.random.rand(n, n)

    reset_graph()
    A_tp = tp.from_numpy(A_raw.copy(), source_id="A")
    B_tp = tp.from_numpy(B_raw.copy(), source_id="B")

    baseline_ms = measure(lambda a=A_raw, b=B_raw: a @ b, 50, 5)
    tp_ms = measure(lambda a=A_tp, b=B_tp: a @ b, 50, 5)
    overhead = tp_ms / baseline_ms
    matmul_results[n] = {
        "baseline_ms": round(baseline_ms, 4),
        "tp_ms": round(tp_ms, 4),
        "overhead": round(overhead, 3),
    }
    print(
        f"matmul ({n}x{n}): vanilla={baseline_ms:.3f}ms  "
        f"traceprop={tp_ms:.3f}ms  overhead={overhead:.3f}x"
    )

with open("results/exp1_overhead.json", "w") as f:
    json.dump({"element_wise": results, "matmul": matmul_results}, f, indent=2)
print("\nSaved to results/exp1_overhead.json")
