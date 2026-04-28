"""Benchmark: measure overhead of provenance tracking vs raw NumPy."""

import time

import numpy as np

import traceprop as tp
from traceprop._c_ext import fast_record_op
from traceprop.granularity import Granularity
from traceprop.graph import reset_graph


def bench_raw_numpy(n_ops: int, size: int) -> float:
    a = np.random.randn(size)
    b = np.random.randn(size)
    start = time.perf_counter()
    for _ in range(n_ops):
        c = a + b
        a = c * b
    return time.perf_counter() - start


def bench_provenance(n_ops: int, size: int, granularity: Granularity = Granularity.OP) -> float:
    reset_graph()
    prev = tp.get_granularity()
    tp.set_granularity(granularity)
    a = tp.array(np.random.randn(size))
    b = tp.array(np.random.randn(size))
    start = time.perf_counter()
    for _ in range(n_ops):
        c = a + b
        a = c * b
    elapsed = time.perf_counter() - start
    tp.set_granularity(prev)
    return elapsed


def bench_no_tracking(n_ops: int, size: int) -> float:
    tp.set_granularity(Granularity.NONE)
    a = tp.array(np.random.randn(size))
    b = tp.array(np.random.randn(size))
    start = time.perf_counter()
    for _ in range(n_ops):
        c = a + b
        a = c * b
    elapsed = time.perf_counter() - start
    tp.set_granularity(Granularity.TENSOR)
    return elapsed


def main():
    n_ops = 500
    size = 1_000_000  # Realistic ML tensor size

    c_ext_active = fast_record_op is not None
    print(f"C extension (fast_record_op): {'ACTIVE' if c_ext_active else 'NOT AVAILABLE'}")
    print(f"Benchmark: {n_ops} ops on arrays of size {size}")
    print("-" * 50)

    t_raw = bench_raw_numpy(n_ops, size)
    t_prov = bench_provenance(n_ops, size)
    t_none = bench_no_tracking(n_ops, size)

    overhead = t_prov / t_raw
    overhead_none = t_none / t_raw

    print(f"Raw NumPy:        {t_raw:.4f}s")
    print(f"With provenance:  {t_prov:.4f}s  ({overhead:.2f}x overhead)")
    print(f"Tracking off:     {t_none:.4f}s  ({overhead_none:.2f}x overhead)")
    print("-" * 50)

    if c_ext_active:
        if overhead < 1.05:
            print(f"PASS: Overhead {overhead:.3f}x < 1.05x (C-accelerated)")
        elif overhead < 2.0:
            print(f"PASS: Overhead {overhead:.2f}x < 2.0x")
        else:
            print(f"WARN: Overhead is {overhead:.2f}x (target < 1.05x with C extension)")
        assert overhead < 1.05, f"C-accelerated overhead {overhead:.3f}x exceeds 1.05x target"
    else:
        if overhead < 10:
            print("PASS: Overhead < 10x (pure Python)")
        else:
            print(f"WARN: Overhead is {overhead:.1f}x (target < 10x)")


if __name__ == "__main__":
    main()
