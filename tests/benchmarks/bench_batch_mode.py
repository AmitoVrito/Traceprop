"""Benchmark: measure overhead of batch/source_id tracking."""

import time

import numpy as np

import traceprop as tp
from traceprop.graph import reset_graph
from traceprop.query import ProvenanceView


def bench_batch_ops(n_batches: int, batch_size: int, n_ops: int) -> float:
    """Benchmark creating batches with source_ids and performing ops."""
    reset_graph()
    start = time.perf_counter()

    for i in range(n_batches):
        a = tp.array(np.random.randn(batch_size), source_id=f"batch_{i}")
        b = tp.array(np.random.randn(batch_size), source_id=f"batch_{i}")
        for _ in range(n_ops):
            c = a + b
            a = c * b

    elapsed = time.perf_counter() - start
    return elapsed


def bench_source_id_query(n_batches: int, batch_size: int) -> float:
    """Benchmark source_id querying after building a lineage graph."""
    reset_graph()

    results = []
    for i in range(n_batches):
        a = tp.array(np.random.randn(batch_size), source_id=f"batch_{i}")
        b = tp.array(np.random.randn(batch_size), source_id=f"batch_{i}")
        c = a + b
        d = c * b
        results.append(d)

    start = time.perf_counter()
    for r in results:
        view = ProvenanceView(r)
        view.source_ids_in_path()
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    n_batches = 50
    batch_size = 100
    n_ops = 20

    print(f"Batch benchmark: {n_batches} batches, size {batch_size}, {n_ops} ops each")
    print("-" * 60)

    t_batch = bench_batch_ops(n_batches, batch_size, n_ops)
    print(f"Batch ops:         {t_batch:.4f}s")

    t_query = bench_source_id_query(n_batches, batch_size)
    print(f"Source ID queries:  {t_query:.4f}s ({n_batches} queries)")
    print("-" * 60)


if __name__ == "__main__":
    main()
