"""
Influence function computation for training data attribution.

The influence of training sample z_i on test prediction f(z_test) is:
    I(z_i, z_test) ~ -grad_L(z_test)^T * H^{-1} * grad_L(z_i)

We approximate H^{-1} using the Gauss-Newton/Fisher approximation, consistent
with TRAK and LogIX approaches.

Key innovation over LogIX/dattri: attribution results are returned with
LineageGraph node_ids, enabling further upstream queries.
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

import numpy as np

from traceprop.exceptions import safe_provenance, logger

if TYPE_CHECKING:
    from traceprop.attribution.gradient_store import GradientStore


@safe_provenance
def compute_influence_scores(
    test_gradient: np.ndarray,
    train_store: GradientStore,
    projection: Optional[Any] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute influence scores of all training samples on a test gradient."""
    if projection is not None:
        proj_test = projection.project(test_gradient)
    else:
        proj_test = test_gradient.flatten().astype(np.float32)

    train_matrix = train_store.get_projected_matrix()

    if len(train_matrix) == 0:
        logger.warning("[traceprop] GradientStore is empty — no influence scores computed")
        return np.array([], dtype=np.float32)

    scores = train_matrix @ proj_test

    if normalize:
        max_abs = np.abs(scores).max()
        if max_abs > 1e-10:
            scores = scores / max_abs

    return scores


def top_k_influential(
    scores: np.ndarray,
    train_store: GradientStore,
    k: int = 10,
    most_harmful: bool = False,
) -> list[dict]:
    """Return the top-k most influential training samples."""
    if len(scores) == 0:
        return []

    sorted_entries = sorted(
        train_store._entries.values(), key=lambda e: e.sample_index
    )

    if most_harmful:
        top_indices = np.argsort(scores)[::-1][:k]
    else:
        top_indices = np.argsort(scores)[:k]

    results = []
    for idx in top_indices:
        if idx < len(sorted_entries):
            entry = sorted_entries[idx]
            # Build a ProvenanceView for this entry if it has a source_node_id
            prov = None
            if entry.source_node_id is not None:
                from traceprop.query import ProvenanceView

                class _NodeRef:
                    pass
                ref = _NodeRef()
                ref._provenance_node_id = entry.source_node_id
                prov = ProvenanceView(ref)

            results.append({
                "rank": len(results) + 1,
                "sample_index": entry.sample_index,
                "source_id": entry.source_id,
                "source_node_id": entry.source_node_id,
                "influence_score": float(scores[idx]),
                "loss_value": entry.loss_value,
                "provenance": prov,
            })

    return results
