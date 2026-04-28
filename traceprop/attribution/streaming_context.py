"""
StreamingTrainingContext: continuous provenance for online learning.

Extends TrainingContext with a rolling window of gradient logs and
periodic checkpointing. Supports attribution on the recent training
window without unbounded memory growth.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import AttributionEngine, AttributionResult
from traceprop.exceptions import safe_provenance, logger


class StreamingTrainingContext:
    """Rolling-window gradient store for online/continual learning.

    Usage::

        stream_ctx = tp.streaming_training_context(
            model=model,
            source_id="live_feed",
            window_size=10_000,
            checkpoint_interval=1_000,
            checkpoint_path="./checkpoints/",
        )

        for batch in stream():
            loss = model(batch)
            stream_ctx.step(loss)

            if should_audit():
                result = stream_ctx.attribute(test_grad, top_k=5)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        source_id: Optional[str] = None,
        window_size: int = 10_000,
        checkpoint_interval: int = 1_000,
        checkpoint_path: Optional[str] = None,
        proj_dim: int = 4096,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.source_id = source_id
        self.window_size = window_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.gradient_store = GradientStore(proj_dim=proj_dim, seed=seed)
        self._proj_dim = proj_dim
        self._seed = seed
        self._sample_counter = 0
        self._steps_since_checkpoint = 0
        self._checkpoint_count = 0

    def __enter__(self) -> StreamingTrainingContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.checkpoint_path and len(self.gradient_store) > 0:
            self._save_checkpoint()
        logger.debug(
            "[traceprop] StreamingTrainingContext closed: %d total samples",
            self._sample_counter,
        )

    @safe_provenance
    def step(
        self,
        loss: Any,
        sample_indices: Optional[list[int]] = None,
        source_node_ids: Optional[list[int]] = None,
    ) -> None:
        """Log gradient from a training step (PyTorch). Use log_gradient() for other frameworks."""
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning(
                "[traceprop] StreamingTrainingContext.step() requires PyTorch. "
                "Use log_gradient() for other frameworks."
            )
            return

        loss.backward(retain_graph=True)

        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().numpy().flatten())
        if not grads:
            return

        flat_grad = np.concatenate(grads)
        node_id = source_node_ids[0] if source_node_ids else None
        idx = sample_indices[0] if sample_indices else self._sample_counter

        self.log_gradient(
            gradient=flat_grad,
            sample_index=idx,
            source_node_id=node_id,
            loss_value=float(loss.detach().cpu().mean()),
        )

    def log_gradient(
        self,
        gradient: Any,
        sample_index: Optional[int] = None,
        source_node_id: Optional[int] = None,
        loss_value: float = 0.0,
    ) -> str:
        """Log a gradient and enforce the rolling window."""
        idx = sample_index if sample_index is not None else self._sample_counter

        result = self.gradient_store.log_gradient(
            gradient=np.asarray(gradient),
            source_node_id=source_node_id,
            source_id=self.source_id,
            sample_index=idx,
            loss_value=loss_value,
        )
        self._sample_counter += 1
        self._steps_since_checkpoint += 1

        # Evict oldest entries if over window_size
        self._enforce_window()

        # Checkpoint if interval reached
        if (
            self.checkpoint_path
            and self._steps_since_checkpoint >= self.checkpoint_interval
        ):
            self._save_checkpoint()
            self._steps_since_checkpoint = 0

        return result or ""

    def _enforce_window(self) -> None:
        """Remove oldest entries to maintain window_size."""
        entries = self.gradient_store._entries
        if len(entries) <= self.window_size:
            return

        sorted_ids = sorted(
            entries.keys(), key=lambda k: entries[k].sample_index
        )
        n_to_remove = len(entries) - self.window_size
        for eid in sorted_ids[:n_to_remove]:
            del entries[eid]

    def _save_checkpoint(self) -> None:
        """Save current gradient store to a checkpoint file."""
        if not self.checkpoint_path:
            return
        os.makedirs(self.checkpoint_path, exist_ok=True)
        path = os.path.join(
            self.checkpoint_path, f"checkpoint_{self._checkpoint_count:04d}.npz"
        )
        self.gradient_store.save(path)
        self._checkpoint_count += 1
        logger.debug("[traceprop] Checkpoint saved: %s", path)

    def attribute(
        self,
        test_gradient: np.ndarray,
        top_k: int = 10,
        most_harmful: bool = False,
    ) -> AttributionResult:
        """Run attribution on the current rolling window."""
        engine = AttributionEngine(self.gradient_store)
        return engine.attribute(
            test_gradient=test_gradient,
            top_k=top_k,
            most_harmful=most_harmful,
        )

    def stats(self) -> dict:
        return {
            "total_samples_seen": self._sample_counter,
            "window_size": self.window_size,
            "current_entries": len(self.gradient_store),
            "checkpoints_saved": self._checkpoint_count,
        }
