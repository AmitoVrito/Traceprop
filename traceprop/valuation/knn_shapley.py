"""
KNN-Shapley data valuation using projected gradients as feature vectors.

KNN-Shapley approximates the Shapley value of each training point as its
contribution to k-nearest-neighbor accuracy on the validation set.
In gradient space, proximity ~ influence similarity.

Reference: Jia et al., "Efficient Task-Specific Data Valuation for
Nearest Neighbor Algorithms", VLDB 2019.

Traceprop extension: aggregate values by source_id and preprocessing op
using the Phase 1-3 lineage graph.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.exceptions import safe_provenance, logger


class ValuationResult:
    """Result of data valuation computation."""

    def __init__(
        self,
        shapley_values: np.ndarray,
        gradient_store: GradientStore,
        lineage_graph: Any = None,
    ) -> None:
        self._values = shapley_values
        self._store = gradient_store
        self._graph = lineage_graph
        self._sorted_entries = sorted(
            gradient_store._entries.values(), key=lambda e: e.sample_index
        )

    def top_samples(self, k: int = 10) -> list[dict]:
        """Return top-k most valuable training samples."""
        if len(self._values) == 0:
            return []
        top_idx = np.argsort(self._values)[::-1][:k]
        results = []
        for idx in top_idx:
            if idx < len(self._sorted_entries):
                entry = self._sorted_entries[idx]
                results.append({
                    "rank": len(results) + 1,
                    "sample_index": entry.sample_index,
                    "source_id": entry.source_id,
                    "source_node_id": entry.source_node_id,
                    "shapley_value": float(self._values[idx]),
                })
        return results

    def by_source(self) -> dict[str, dict]:
        """Aggregate Shapley values by source_id."""
        source_agg: dict[str, dict] = {}
        for i, entry in enumerate(self._sorted_entries):
            if i >= len(self._values):
                break
            sid = entry.source_id or "_unknown"
            if sid not in source_agg:
                source_agg[sid] = {"total_value": 0.0, "n_samples": 0, "values": []}
            source_agg[sid]["total_value"] += float(self._values[i])
            source_agg[sid]["n_samples"] += 1
            source_agg[sid]["values"].append(float(self._values[i]))

        for sid, agg in source_agg.items():
            agg["mean_value"] = agg["total_value"] / agg["n_samples"] if agg["n_samples"] else 0.0
            del agg["values"]

        return source_agg

    def by_preprocessing_op(self) -> dict[str, float]:
        """Aggregate Shapley values by preprocessing operation using lineage graph.

        This query is novel — it requires the Phase 1-3 lineage graph.
        For each unique op_name in the lineage of valued samples, compute
        the mean Shapley value of samples that passed through that op.
        """
        if self._graph is None:
            return {}

        op_values: dict[str, list[float]] = {}
        for i, entry in enumerate(self._sorted_entries):
            if i >= len(self._values):
                break
            nid = entry.source_node_id
            if nid is None:
                continue
            # Walk backward edges from this node to find ops
            visited = set()
            queue = [nid]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                for eid in self._graph._backward.get(current, []):
                    edge = self._graph.edges.get(eid)
                    if edge is None:
                        continue
                    op = edge.op_name
                    if op not in op_values:
                        op_values[op] = []
                    op_values[op].append(float(self._values[i]))
                    for inp_id in edge.input_ids:
                        queue.append(inp_id)

        return {op: np.mean(vals) for op, vals in op_values.items() if vals}

    @property
    def values(self) -> np.ndarray:
        return self._values


@safe_provenance
def knn_shapley_values(
    gradient_store: GradientStore,
    val_gradients: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute KNN-Shapley values using projected gradients as feature vectors.

    Args:
        gradient_store: GradientStore with training gradients.
        val_gradients: Validation gradients, shape (n_val, grad_dim) or (n_val, proj_dim).
            If grad_dim != proj_dim, they are projected using the store's projection.
        k: Number of nearest neighbors for KNN-Shapley.

    Returns:
        shapley_values: shape (n_train_samples,)
    """
    train_matrix = gradient_store.get_projected_matrix()
    n_train = len(train_matrix)
    if n_train == 0:
        return np.array([], dtype=np.float32)

    # Project val gradients if needed
    proj_dim = gradient_store._proj_dim
    if val_gradients.ndim == 1:
        val_gradients = val_gradients.reshape(1, -1)

    if val_gradients.shape[1] != proj_dim and gradient_store._projection is not None:
        val_proj = np.stack([
            gradient_store._projection.project(g) for g in val_gradients
        ])
    else:
        val_proj = val_gradients.astype(np.float32)

    n_val = len(val_proj)
    shapley = np.zeros(n_train, dtype=np.float64)

    # For each validation point, compute KNN-Shapley contributions
    # KNN-Shapley: value of training point i for validation point j is
    # based on its rank in the distance ordering
    for j in range(n_val):
        # Compute distances (negative dot product = larger distance)
        similarities = train_matrix @ val_proj[j]
        sorted_idx = np.argsort(-similarities)  # most similar first

        # KNN-Shapley weight: decreasing with rank, zero after k
        for rank, idx in enumerate(sorted_idx[:k]):
            # Weight = 1/k * (1 - rank/k) — triangular weighting
            weight = (1.0 / k) * (1.0 - rank / k)
            shapley[idx] += weight / n_val

    return shapley.astype(np.float32)


def aggregate_by_source(
    shapley_values: np.ndarray,
    gradient_store: GradientStore,
    lineage_graph: Any = None,
) -> dict[str, dict]:
    """Aggregate Shapley values by source_id using the Phase 1-3 lineage graph."""
    result = ValuationResult(shapley_values, gradient_store, lineage_graph)
    return result.by_source()
