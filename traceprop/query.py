"""Query API for inspecting provenance of a ProvenanceTensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from traceprop.graph import OpEdge, TensorNode, get_graph

if TYPE_CHECKING:
    from traceprop.tensor import ProvenanceTensor


class ProvenanceView:
    """Read-only view into the lineage of a tensor."""

    __slots__ = ("_node_id",)

    def __init__(self, tensor):
        # Duck-type: accept anything with _provenance_node_id
        self._node_id = getattr(tensor, "_provenance_node_id", None)

    @property
    def node_id(self) -> int | None:
        return self._node_id

    @property
    def node(self) -> TensorNode | None:
        if self._node_id is None:
            return None
        return get_graph().nodes.get(self._node_id)

    def sources(self) -> list[TensorNode]:
        """Return immediate input nodes that produced this tensor."""
        if self._node_id is None:
            return []
        graph = get_graph()
        result = []
        for eid in graph._backward.get(self._node_id, ()):
            edge = graph.edges[eid]
            for inp_id in edge.input_ids:
                node = graph.nodes.get(inp_id)
                if node is not None:
                    result.append(node)
        return result

    def ops(self) -> list[OpEdge]:
        """Return all edges (ops) connected to this tensor."""
        if self._node_id is None:
            return []
        return list(get_graph().edges_for_node(self._node_id))

    def ancestors(self) -> set[int]:
        """Return all ancestor node IDs."""
        if self._node_id is None:
            return set()
        return get_graph().ancestors(self._node_id)

    def descendants(self) -> set[int]:
        """Return all descendant node IDs."""
        if self._node_id is None:
            return set()
        return get_graph().descendants(self._node_id)

    def training_attribution(
        self,
        gradient_store=None,
        test_gradient=None,
        top_k: int = 10,
        most_harmful: bool = False,
    ):
        """Query gradient-level training data attribution for this tensor.

        Args:
            gradient_store: A GradientStore with logged training gradients.
                If None, uses the most recently created TrainingContext's store.
            test_gradient: Gradient of the test loss. If None, a zero gradient is used.
            top_k: Number of top influential samples to return.
            most_harmful: If True, return samples that most increased loss.

        Returns:
            AttributionResult with lineage pointers back to source files.
        """
        from traceprop.attribution.attribution_engine import AttributionEngine

        if gradient_store is None:
            from traceprop.attribution.training_context import _active_gradient_store
            gradient_store = _active_gradient_store()
            if gradient_store is None:
                return None

        import numpy as np
        if test_gradient is None:
            test_gradient = np.zeros(gradient_store._proj_dim, dtype=np.float32)

        engine = AttributionEngine(gradient_store)
        return engine.attribute(
            test_gradient=test_gradient,
            top_k=top_k,
            most_harmful=most_harmful,
        )

    def source_ids_in_path(self) -> set[str]:
        """Return all source_ids found in edges along the ancestor path."""
        if self._node_id is None:
            return set()
        graph = get_graph()
        result: set[str] = set()
        # Check edges leading to this node and all ancestors
        ancestor_ids = graph.ancestors(self._node_id)
        all_ids = ancestor_ids | {self._node_id}
        for nid in all_ids:
            for eid in graph._backward.get(nid, ()):
                edge = graph.edges[eid]
                for sid in getattr(edge, "source_ids", ()):
                    result.add(sid)
        return result
