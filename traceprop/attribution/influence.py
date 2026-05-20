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


def compute_trak_scores(
    test_gradient: np.ndarray,
    train_store: "GradientStore",
    lambda_factor: float = 1e-3,
    gram_factor: Optional[Any] = None,
    normalize: bool = True,
) -> np.ndarray:
    """TRAK estimator: φ_test @ (ΦᵀΦ + λI)⁻¹ @ Φ_train.T

    Replaces the raw dot product with a regularised inverse-Gram product,
    approximating the inverse Hessian in the projected gradient space.
    Pre-computed gram_factor (Cholesky) can be passed for efficiency when
    attributing many test samples against the same store.
    """
    import scipy.linalg

    proj = train_store._projection
    if proj is None:
        return np.array([], dtype=np.float32)

    phi_test = proj.project(test_gradient).astype(np.float64)  # (d,)
    Phi = train_store.get_projected_matrix().astype(np.float64)  # (n, d)
    if len(Phi) == 0:
        return np.array([], dtype=np.float32)

    if gram_factor is None:
        d = Phi.shape[1]
        G = Phi.T @ Phi                                    # (d, d)
        lam = lambda_factor * np.trace(G) / d
        G += lam * np.eye(d)
        gram_factor = scipy.linalg.cho_factor(G)

    v = scipy.linalg.cho_solve(gram_factor, phi_test)     # (d,)
    scores = (Phi @ v).astype(np.float32)                 # (n,)

    if normalize:
        max_abs = np.abs(scores).max()
        if max_abs > 1e-10:
            scores = scores / max_abs

    return scores


def precompute_gram_factor(
    train_store: "GradientStore",
    lambda_factor: float = 1e-3,
) -> Any:
    """Precompute the Cholesky factor of (ΦᵀΦ + λI) for reuse across test samples."""
    import scipy.linalg

    Phi = train_store.get_projected_matrix().astype(np.float64)
    if len(Phi) == 0:
        return None
    d = Phi.shape[1]
    G = Phi.T @ Phi
    lam = lambda_factor * np.trace(G) / d
    G += lam * np.eye(d)
    return scipy.linalg.cho_factor(G)


def compute_influence_scores_multi_checkpoint(
    test_gradient: np.ndarray,
    checkpoint_stores: list,
    use_trak: bool = False,
    lambda_factor: float = 1e-3,
    gram_factors: Optional[list] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Average influence scores across K checkpoint GradientStores.

    Each store should have been created with a different random seed so
    its JL projection is independent (same as TRAK's multi-checkpoint approach).

    Args:
        test_gradient:     Raw gradient vector for the test sample.
        checkpoint_stores: List of GradientStore, one per checkpoint.
        use_trak:          If True, use TRAK estimator per checkpoint.
        lambda_factor:     Regularisation for TRAK estimator.
        gram_factors:      Pre-computed Cholesky factors (one per store).
        normalize:         L-inf normalise before averaging.
    """
    all_scores = []
    for k, store in enumerate(checkpoint_stores):
        if use_trak:
            gf = gram_factors[k] if gram_factors else None
            scores_k = compute_trak_scores(
                test_gradient, store, lambda_factor=lambda_factor,
                gram_factor=gf, normalize=normalize,
            )
        else:
            proj = store._projection
            if proj is None:
                continue
            phi_test = proj.project(test_gradient)
            Phi = store.get_projected_matrix()
            if len(Phi) == 0:
                continue
            scores_k = (Phi @ phi_test).astype(np.float32)
            if normalize:
                m = np.abs(scores_k).max()
                if m > 1e-10:
                    scores_k = scores_k / m
        if len(scores_k) > 0:
            all_scores.append(scores_k)

    if not all_scores:
        return np.array([], dtype=np.float32)
    return np.mean(all_scores, axis=0).astype(np.float32)


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
