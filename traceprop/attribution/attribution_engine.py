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
from traceprop.attribution.influence import (
    compute_influence_scores,
    compute_trak_scores,
    compute_influence_scores_multi_checkpoint,
    precompute_gram_factor,
    top_k_influential,
)
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
    """End-to-end attribution: from model output back to source files.

    estimator options:
      "dot"   — raw dot product (default, fast)
      "trak"  — TRAK estimator φ_test @ (ΦᵀΦ + λI)⁻¹ @ Φ_train.T
    """

    def __init__(
        self,
        gradient_store: GradientStore,
        estimator: str = "dot",
        lambda_factor: float = 1e-3,
    ) -> None:
        self.gradient_store = gradient_store
        self.estimator = estimator
        self._gram_factor = None
        if estimator == "trak":
            self._gram_factor = precompute_gram_factor(gradient_store, lambda_factor)
            self._lambda_factor = lambda_factor

    def attribute_scores(self, test_gradient: np.ndarray) -> np.ndarray:
        """Return raw influence scores (n_train,) for a test gradient."""
        if self.gradient_store._projection is None:
            return np.array([], dtype=np.float32)
        if self.estimator == "trak":
            return compute_trak_scores(
                test_gradient=test_gradient,
                train_store=self.gradient_store,
                lambda_factor=self._lambda_factor,
                gram_factor=self._gram_factor,
            )
        return compute_influence_scores(
            test_gradient=test_gradient,
            train_store=self.gradient_store,
            projection=self.gradient_store._projection,
        )

    def attribute_scores_batch(self, test_gradients: np.ndarray) -> np.ndarray:
        """Batch version of attribute_scores — uses all CPU cores via BLAS.

        Args:
            test_gradients: (n_test, grad_dim) array of raw test gradients.
        Returns:
            (n_test, n_train) influence matrix — one big matrix multiply,
            parallelised across all CPU cores by OpenBLAS/MKL/Accelerate.
        """
        import scipy.linalg
        proj = self.gradient_store._projection
        if proj is None:
            return np.zeros((len(test_gradients), 0), dtype=np.float32)

        # Project all test gradients at once: (n_test, proj_dim)
        G = test_gradients.reshape(len(test_gradients), -1).astype(np.float32)
        Phi_te = G @ proj._matrix.T           # single BLAS dgemm, all cores

        Phi_tr = self.gradient_store.get_projected_matrix()  # (n_train, proj_dim)
        if len(Phi_tr) == 0:
            return np.zeros((len(test_gradients), 0), dtype=np.float32)

        if self.estimator == "trak":
            # Batch Cholesky solve: cho_solve accepts matrix RHS
            V = scipy.linalg.cho_solve(
                self._gram_factor,
                Phi_te.T.astype(np.float64),
            )                                  # (proj_dim, n_test)
            scores = (Phi_tr.astype(np.float64) @ V).T  # (n_test, n_train)
        else:
            scores = (Phi_te @ Phi_tr.T).astype(np.float64)  # (n_test, n_train)

        # L-inf normalise each row independently
        max_abs = np.abs(scores).max(axis=1, keepdims=True)
        scores = np.where(max_abs > 1e-10, scores / max_abs, scores)
        return scores.astype(np.float32)

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

        if self.estimator == "trak":
            scores = compute_trak_scores(
                test_gradient=test_gradient,
                train_store=self.gradient_store,
                lambda_factor=self._lambda_factor,
                gram_factor=self._gram_factor,
            )
        else:
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


class MultiCheckpointAttributionEngine:
    """Averages attribution scores across K checkpoint GradientStores.

    Each checkpoint store should use a different random seed so its JL
    projection is independent — this mirrors TRAK's multi-checkpoint approach.

    Args:
        checkpoint_stores: List of GradientStore, one per training checkpoint.
        use_trak:          Apply TRAK estimator within each checkpoint.
        lambda_factor:     Regularisation strength for TRAK estimator.
    """

    def __init__(
        self,
        checkpoint_stores: list[GradientStore],
        use_trak: bool = False,
        lambda_factor: float = 1e-3,
    ) -> None:
        self.checkpoint_stores = checkpoint_stores
        self.use_trak = use_trak
        self._lambda_factor = lambda_factor
        self._gram_factors = None
        if use_trak:
            self._gram_factors = [
                precompute_gram_factor(s, lambda_factor) for s in checkpoint_stores
            ]

    def attribute_scores_batch(self, test_gradients: np.ndarray) -> np.ndarray:
        """Batch version — one large matmul per checkpoint, uses all CPU cores.

        Args:
            test_gradients: (n_test, grad_dim)
        Returns:
            (n_test, n_train) averaged influence matrix.
        """
        import scipy.linalg
        all_scores = []
        G = test_gradients.reshape(len(test_gradients), -1).astype(np.float32)

        for k, store in enumerate(self.checkpoint_stores):
            proj = store._projection
            if proj is None:
                continue
            Phi_te = G @ proj._matrix.T                    # (n_test, proj_dim)
            Phi_tr = store.get_projected_matrix()           # (n_train, proj_dim)
            if len(Phi_tr) == 0:
                continue

            if self.use_trak:
                gf = self._gram_factors[k] if self._gram_factors else None
                if gf is None:
                    continue
                V = scipy.linalg.cho_solve(
                    gf, Phi_te.T.astype(np.float64)
                )                                           # (proj_dim, n_test)
                scores_k = (Phi_tr.astype(np.float64) @ V).T
            else:
                scores_k = (Phi_te @ Phi_tr.T).astype(np.float64)

            max_abs = np.abs(scores_k).max(axis=1, keepdims=True)
            scores_k = np.where(max_abs > 1e-10, scores_k / max_abs, scores_k)
            all_scores.append(scores_k.astype(np.float32))

        if not all_scores:
            return np.zeros((len(test_gradients), 0), dtype=np.float32)
        return np.mean(all_scores, axis=0).astype(np.float32)

    def attribute_scores(self, test_gradient: np.ndarray) -> np.ndarray:
        """Return raw averaged influence scores (n_train,) for a test gradient."""
        return compute_influence_scores_multi_checkpoint(
            test_gradient=test_gradient,
            checkpoint_stores=self.checkpoint_stores,
            use_trak=self.use_trak,
            lambda_factor=self._lambda_factor,
            gram_factors=self._gram_factors,
        )

    def attribute(
        self,
        test_gradient: np.ndarray,
        top_k: int = 20,
        most_harmful: bool = False,
    ) -> AttributionResult:
        """Return AttributionResult for the test gradient."""
        scores = self.attribute_scores(test_gradient)
        if len(scores) == 0:
            return AttributionResult([], get_graph())
        entries = top_k_influential(
            scores=scores,
            train_store=self.checkpoint_stores[-1],  # use last ckpt for metadata
            k=top_k,
            most_harmful=most_harmful,
        )
        return AttributionResult(entries or [], get_graph())
