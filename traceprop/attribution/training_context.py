"""
Context manager that hooks into a training loop and logs per-sample gradients.
Supports PyTorch natively. NumPy and JAX require manual gradient passing.
"""
from __future__ import annotations

from typing import Optional, Any

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.exceptions import safe_provenance, logger


_last_gradient_store: Optional[GradientStore] = None


def _active_gradient_store() -> Optional[GradientStore]:
    """Return the most recently created TrainingContext's gradient store."""
    return _last_gradient_store


class TrainingContext:
    """Wraps a training loop to record per-sample gradients for later attribution.

    Supports context manager protocol::

        with tp.training_context(model, source_id="data_v3") as ctx:
            for batch in loader:
                loss = model(batch)
                ctx.step(loss)

    Or use directly::

        ctx = tp.training_context(source_id="data_v3")
        for i, gradient in enumerate(grads):
            ctx.log_gradient(gradient, sample_index=i)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        X_train: Optional[Any] = None,
        y_train: Optional[Any] = None,
        source_id: Optional[str] = None,
        proj_dim: int = 4096,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.source_id = source_id
        self.gradient_store = GradientStore(proj_dim=proj_dim, seed=seed)
        self._sample_counter = 0
        global _last_gradient_store
        _last_gradient_store = self.gradient_store

    def __enter__(self) -> TrainingContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logger.debug(
            "[traceprop] TrainingContext closed: %d gradients logged",
            len(self.gradient_store),
        )

    @safe_provenance
    def step(
        self,
        loss: Any,
        sample_indices: Optional[list[int]] = None,
        source_node_ids: Optional[list[int]] = None,
    ) -> None:
        """Call after computing loss for a batch. Extracts and logs per-sample gradients.

        Requires PyTorch. For other frameworks, use log_gradient() directly.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning(
                "[traceprop] TrainingContext.step() requires PyTorch. "
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

        batch_size = 1
        if hasattr(loss, "shape") and loss.shape:
            batch_size = loss.shape[0]

        indices = sample_indices or list(range(
            self._sample_counter, self._sample_counter + batch_size
        ))

        node_id = source_node_ids[0] if source_node_ids else None

        self.gradient_store.log_gradient(
            gradient=flat_grad,
            source_node_id=node_id,
            source_id=self.source_id,
            sample_index=indices[0] if indices else self._sample_counter,
            loss_value=float(loss.detach().cpu().mean()),
        )
        self._sample_counter += len(indices)

    def log_gradient(
        self,
        gradient: Any,
        sample_index: Optional[int] = None,
        source_node_id: Optional[int] = None,
        loss_value: float = 0.0,
    ) -> str:
        """Manually log a gradient. Use for NumPy/JAX pipelines."""
        idx = sample_index if sample_index is not None else self._sample_counter
        result = self.gradient_store.log_gradient(
            gradient=np.asarray(gradient),
            source_node_id=source_node_id,
            source_id=self.source_id,
            sample_index=idx,
            loss_value=loss_value,
        )
        self._sample_counter += 1
        return result or ""
