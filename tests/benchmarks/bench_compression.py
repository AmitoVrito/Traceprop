"""Benchmark: compression savings for large matmul operations."""

from __future__ import annotations

import time

import numpy as np

import traceprop as tp
from traceprop.compression import CompressedLineageGraph, should_use_range_encoding, build_range_descriptor
from traceprop.graph import get_graph, reset_graph


def bench_compression_savings():
    """Compare graph size with and without range compression."""
    reset_graph()
    N = 200

    # Without compression
    a = tp.array(np.random.randn(N, N))
    b = tp.array(np.random.randn(N, N))
    c = np.dot(a, b)
    graph = get_graph()
    normal_stats = graph.stats()

    # With compression
    reset_graph()
    a = tp.array(np.random.randn(N, N))
    b = tp.array(np.random.randn(N, N))
    cg = tp.enable_compression()
    c = np.dot(a, b)
    compressed_stats = cg.stats()

    print(f"Normal graph:     {normal_stats}")
    print(f"Compressed graph: {compressed_stats}")
    print(f"Range encoding applicable for matmul({N}x{N}): {should_use_range_encoding('matmul', [(N, N)])}")

    # Benchmark should_use_range_encoding speed
    start = time.perf_counter()
    for _ in range(100_000):
        should_use_range_encoding("matmul", [(N, N)])
    elapsed = time.perf_counter() - start
    print(f"should_use_range_encoding: {elapsed*1e6/100_000:.2f} µs/call")


if __name__ == "__main__":
    bench_compression_savings()
