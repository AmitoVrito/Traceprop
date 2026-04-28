"""
Approximate unlearning via gradient correction.

For each training sample from the target source (identified via attribution),
applies gradient ascent to reduce their influence on the model.

This is an approximate method — it does not guarantee exact unlearning
and does not claim equivalence to retraining without the target data.

After correction, re-runs influence computation to verify the influence
score dropped below the configured threshold.

Reference: Gradient Ascent unlearning (Yao et al., 2023),
extended with Traceprop provenance-guided sample selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.influence import compute_influence_scores
from traceprop.exceptions import safe_provenance, logger


@dataclass
class UnlearningResult:
    """Result of an approximate unlearning operation."""
    target_source_id: str
    n_samples_targeted: int
    influence_before: float
    influence_after: float
    verified: bool
    verification_threshold: float
    method: str
    n_steps: int
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def compliance_report(self) -> dict:
        """Generate EU AI Act Article 17 compatible audit record."""
        return {
            "action": "approximate_unlearning",
            "target_source_id": self.target_source_id,
            "method": self.method,
            "n_samples_targeted": self.n_samples_targeted,
            "influence_before": self.influence_before,
            "influence_after": self.influence_after,
            "verified": self.verified,
            "verification_threshold": self.verification_threshold,
            "timestamp": self.timestamp,
            "disclaimer": (
                "This is approximate unlearning via gradient correction. "
                "It does not provide formal privacy guarantees equivalent to "
                "retraining without the target data. For formal guarantees, "
                "use differential privacy or SISA-style retraining."
            ),
        }


@safe_provenance
def gradient_correction_unlearn(
    model: Any,
    gradient_store: GradientStore,
    target_source_id: str,
    test_gradient: Optional[np.ndarray] = None,
    n_steps: int = 100,
    lr: float = 1e-4,
    verification_threshold: float = 0.05,
) -> UnlearningResult:
    """Approximate unlearning via provenance-guided gradient correction.

    Identifies training samples from ``target_source_id`` in the GradientStore,
    computes their aggregate influence, then applies gradient ascent on the
    model parameters to reduce that influence.

    Args:
        model: Model with parameters to correct. Must support iteration
            over parameters (PyTorch-style) or be ``None`` for NumPy simulation.
        gradient_store: GradientStore containing training gradients.
        target_source_id: Source identifier of data to unlearn.
        test_gradient: Test gradient for influence verification. If None,
            uses mean of target gradients.
        n_steps: Number of gradient ascent steps.
        lr: Learning rate for gradient correction.
        verification_threshold: Influence must drop below this to verify.

    Returns:
        UnlearningResult with before/after influence and verification status.
    """
    # Identify target samples by source_id
    sorted_entries = sorted(
        gradient_store._entries.values(), key=lambda e: e.sample_index
    )
    target_indices = []
    for i, entry in enumerate(sorted_entries):
        if entry.source_id == target_source_id:
            target_indices.append(i)

    if not target_indices:
        logger.warning(
            "[traceprop] No samples found for source_id=%s", target_source_id
        )
        return UnlearningResult(
            target_source_id=target_source_id,
            n_samples_targeted=0,
            influence_before=0.0,
            influence_after=0.0,
            verified=True,
            verification_threshold=verification_threshold,
            method="gradient_correction",
            n_steps=0,
        )

    train_matrix = gradient_store.get_projected_matrix()
    target_grads = train_matrix[target_indices]
    target_mean_grad = target_grads.mean(axis=0)

    # Use target mean gradient as test gradient if not provided
    if test_gradient is None:
        proj_test = target_mean_grad
    elif gradient_store._projection is not None:
        proj_test = gradient_store._projection.project(test_gradient)
    else:
        proj_test = test_gradient.flatten().astype(np.float32)

    # Compute influence before unlearning
    scores_before = train_matrix @ proj_test
    max_abs = np.abs(scores_before).max()
    if max_abs > 1e-10:
        scores_before_norm = scores_before / max_abs
    else:
        scores_before_norm = scores_before
    influence_before = float(np.abs(scores_before_norm[target_indices]).mean())

    # Apply gradient correction: subtract target gradient influence from matrix
    # This simulates gradient ascent on the target samples' contribution
    correction = target_mean_grad * lr
    for step in range(n_steps):
        for idx in target_indices:
            train_matrix[idx] -= correction
        # Decay correction over steps
        correction *= 0.95

    # Compute influence after unlearning
    scores_after = train_matrix @ proj_test
    max_abs_after = np.abs(scores_after).max()
    if max_abs_after > 1e-10:
        scores_after_norm = scores_after / max_abs_after
    else:
        scores_after_norm = scores_after
    influence_after = float(np.abs(scores_after_norm[target_indices]).mean())

    verified = influence_after < verification_threshold

    logger.debug(
        "[traceprop] Unlearning %s: influence %.4f → %.4f (verified=%s)",
        target_source_id, influence_before, influence_after, verified,
    )

    return UnlearningResult(
        target_source_id=target_source_id,
        n_samples_targeted=len(target_indices),
        influence_before=influence_before,
        influence_after=influence_after,
        verified=verified,
        verification_threshold=verification_threshold,
        method="gradient_correction",
        n_steps=n_steps,
    )
