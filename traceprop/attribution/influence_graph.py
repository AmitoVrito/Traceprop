"""
InfluenceGraph: extends LineageGraph with gradient-level edges for unified queries.

Merges computation lineage (Phase 1-3) with gradient attribution (Phase 4) into
a single queryable graph. Gradient edges connect training samples to model outputs
via influence scores.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from traceprop.graph import LineageGraph, TensorNode


@dataclass
class GradientEdge:
    """An edge representing gradient-level influence between a training sample and output."""
    id: int
    source_node_id: int
    target_node_id: int
    influence_score: float
    source_id: Optional[str] = None
    sample_index: int = -1
    metadata: dict = field(default_factory=dict)


class InfluenceGraph:
    """Extends LineageGraph with gradient-level influence edges.

    Provides unified queries across computation lineage (op edges)
    and gradient influence (gradient edges).
    """

    def __init__(self, lineage_graph: LineageGraph) -> None:
        self._lineage = lineage_graph
        self._gradient_edges: dict[int, GradientEdge] = {}
        self._grad_forward: dict[int, list[int]] = defaultdict(list)
        self._grad_backward: dict[int, list[int]] = defaultdict(list)
        self._edge_counter = 0

    @property
    def lineage(self) -> LineageGraph:
        return self._lineage

    def add_gradient_edge(
        self,
        source_node_id: int,
        target_node_id: int,
        influence_score: float,
        source_id: Optional[str] = None,
        sample_index: int = -1,
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a gradient influence edge from a training sample node to an output node."""
        eid = self._edge_counter
        self._edge_counter += 1
        edge = GradientEdge(
            id=eid,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            influence_score=influence_score,
            source_id=source_id,
            sample_index=sample_index,
            metadata=metadata or {},
        )
        self._gradient_edges[eid] = edge
        self._grad_forward[source_node_id].append(eid)
        self._grad_backward[target_node_id].append(eid)
        return eid

    def gradient_ancestors(self, node_id: int) -> list[GradientEdge]:
        """Return all gradient edges that influence the given node."""
        return [
            self._gradient_edges[eid]
            for eid in self._grad_backward.get(node_id, [])
        ]

    def gradient_descendants(self, node_id: int) -> list[GradientEdge]:
        """Return all gradient edges originating from the given node."""
        return [
            self._gradient_edges[eid]
            for eid in self._grad_forward.get(node_id, [])
        ]

    def full_trace(self, node_id: int) -> dict:
        """Unified trace: both computation ancestors and gradient influences."""
        comp_ancestors = self._lineage.ancestors(node_id)
        grad_edges = self.gradient_ancestors(node_id)
        return {
            "computation_ancestors": comp_ancestors,
            "gradient_influences": [
                {
                    "source_node_id": e.source_node_id,
                    "influence_score": e.influence_score,
                    "source_id": e.source_id,
                    "sample_index": e.sample_index,
                }
                for e in grad_edges
            ],
        }

    def stats(self) -> dict:
        lineage_stats = self._lineage.stats()
        return {
            **lineage_stats,
            "gradient_edges": len(self._gradient_edges),
        }
