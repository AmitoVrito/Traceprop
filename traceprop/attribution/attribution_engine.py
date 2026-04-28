"""
AttributionEngine: the unified interface for end-to-end attribution queries.

Connects the computation lineage graph (Phases 1-3) with gradient-level
training data attribution (Phase 4). This is the unique contribution of
Traceprop over LogIX and dattri — neither of those connect attribution
results back to a computation lineage graph.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.influence import compute_influence_scores, top_k_influential
from traceprop.exceptions import safe_provenance, logger
from traceprop.graph import get_graph


class AttributionResult:
    """Returned by AttributionEngine.attribute().

    Each result entry carries a source_node_id linking back to the
    Phase 1-3 lineage graph for further upstream queries.
    """

    def __init__(self, entries: list[dict], graph: Any) -> None:
        self._entries = entries
        self._graph = graph

    def top(self, k: int = 10) -> list[dict]:
        return self._entries[:k]

    def __len__(self) -> int:
        return len(self._entries)

    def sources(self) -> list[str]:
        """Return unique source_ids from all attribution entries."""
        return list({e["source_id"] for e in self._entries if e.get("source_id")})

    def trace_to_file(self, rank: int = 0) -> dict:
        """For the rank-th most influential sample, trace its full lineage
        back through the Phase 1-3 computation graph to its source file."""
        if rank >= len(self._entries):
            return {}
        entry = self._entries[rank]
        node_id = entry.get("source_node_id")
        if not node_id:
            return {"error": "No source_node_id — was TrainingContext used?"}

        from traceprop.query import ProvenanceView

        # Create a dummy object with _provenance_node_id for ProvenanceView
        class _NodeRef:
            pass
        ref = _NodeRef()
        ref._provenance_node_id = node_id

        view = ProvenanceView(ref)
        return {
            "rank": rank,
            "influence_score": entry["influence_score"],
            "sources": view.sources(),
            "ops": view.ops(),
        }


class AttributionEngine:
    """End-to-end attribution: from model output back to source files."""

    def __init__(self, gradient_store: GradientStore) -> None:
        self.gradient_store = gradient_store

    @safe_provenance
    def attribute(
        self,
        test_gradient: np.ndarray,
        top_k: int = 20,
        most_harmful: bool = False,
    ) -> AttributionResult:
        """Given the gradient of a test loss, find the most influential training samples."""
        if self.gradient_store._projection is None:
            logger.warning(
                "[traceprop] GradientStore has no projection matrix. "
                "Was TrainingContext.step() called?"
            )
            return AttributionResult([], get_graph())

        scores = compute_influence_scores(
            test_gradient=test_gradient,
            train_store=self.gradient_store,
            projection=self.gradient_store._projection,
        )

        if scores is None or len(scores) == 0:
            return AttributionResult([], get_graph())

        entries = top_k_influential(
            scores=scores,
            train_store=self.gradient_store,
            k=top_k,
            most_harmful=most_harmful,
        )

        return AttributionResult(entries or [], get_graph())
