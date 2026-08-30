"""
Stores per-sample gradient logs needed for influence function computation.
Uses random projection to make storage tractable.

Design follows the approach from:
- TRAK (Park et al., ICML 2023): random projection of gradients
- LogIX (ICLR 2025): efficient gradient logging for billion-scale models

Key difference from both: GradientStore integrates with Traceprop's LineageGraph
so every stored gradient entry carries a node_id linking it back to the
computation lineage of the training sample that produced it.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from traceprop.exceptions import safe_provenance, logger


@dataclass
class GradientLogEntry:
    """A compressed gradient record for one training sample."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proj_gradient: Optional[np.ndarray] = None
    source_node_id: Optional[int] = None
    source_id: Optional[str] = None
    sample_index: int = -1
    loss_value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class RandomProjection:
    """Stable random projection matrix for gradient compression.

    Uses the same projection across all samples (required for influence computation).
    Seeded for reproducibility.
    """

    def __init__(self, input_dim: int, proj_dim: int = 4096, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        rng = np.random.default_rng(seed)
        # Sparse Johnson-Lindenstrauss projection (memory-efficient)
        self._matrix = rng.choice(
            [-1.0, 0.0, 1.0],
            size=(proj_dim, input_dim),
            p=[1/6, 2/3, 1/6],
        ).astype(np.float32) * np.sqrt(3.0 / proj_dim)

    def project(self, gradient: np.ndarray) -> np.ndarray:
        """Project a flat gradient vector into proj_dim dimensions."""
        flat = gradient.flatten().astype(np.float32)
        if len(flat) != self.input_dim:
            raise ValueError(
                f"Gradient dim {len(flat)} != projection input_dim {self.input_dim}"
            )
        return self._matrix @ flat

    def project_batch(self, gradients: np.ndarray) -> np.ndarray:
        """Project an (n, input_dim) batch in a single BLAS matmul.

        Returns float32 to match :meth:`project`.
        """
        if gradients.ndim != 2 or gradients.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected (n, {self.input_dim}); got {gradients.shape}"
            )
        g = gradients.astype(np.float32, copy=False)
        # (n, p) @ (p, k) -> (n, k), explicitly float32
        return (g @ self._matrix.T).astype(np.float32, copy=False)


class GradientStore:
    """In-memory store of compressed per-sample gradient logs."""

    def __init__(self, proj_dim: int = 4096, seed: int = 42) -> None:
        self._entries: dict[str, GradientLogEntry] = {}
        self._proj_dim = proj_dim
        self._seed = seed
        self._projection: Optional[RandomProjection] = None

    @safe_provenance
    def log_gradient(
        self,
        gradient: np.ndarray,
        source_node_id: Optional[int] = None,
        source_id: Optional[str] = None,
        sample_index: int = -1,
        loss_value: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record a compressed gradient for one training sample."""
        flat = gradient.flatten()

        if self._projection is None:
            self._projection = RandomProjection(
                input_dim=len(flat),
                proj_dim=self._proj_dim,
                seed=self._seed,
            )

        proj_grad = self._projection.project(flat)
        entry = GradientLogEntry(
            proj_gradient=proj_grad,
            source_node_id=source_node_id,
            source_id=source_id,
            sample_index=sample_index,
            loss_value=loss_value,
            metadata=metadata or {},
        )
        self._entries[entry.id] = entry
        return entry.id

    def log_batch(
        self,
        gradients: np.ndarray,
        source_id: Optional[str] = None,
        sample_index_offset: int = 0,
        source_node_ids: Optional[list] = None,
    ) -> list[str]:
        """Vectorised batched gradient logging.

        Accepts an (n, p) array of per-sample gradients and projects them
        in a single BLAS matmul, then stores each row as a GradientLogEntry.
        Avoids the per-sample Python interpreter dispatch that dominates
        log_gradient() in tight training loops.
        """
        if gradients.ndim == 1:
            gradients = gradients[None, :]
        n, p = gradients.shape

        if self._projection is None:
            self._projection = RandomProjection(
                input_dim=p,
                proj_dim=self._proj_dim,
                seed=self._seed,
            )

        # One matmul for the whole batch (sparse JL projection).
        proj = self._projection.project_batch(gradients) \
            if hasattr(self._projection, "project_batch") \
            else np.stack([self._projection.project(g) for g in gradients])

        ids = []
        for i in range(n):
            entry = GradientLogEntry(
                proj_gradient=proj[i],
                source_node_id=(source_node_ids[i] if source_node_ids else None),
                source_id=source_id,
                sample_index=sample_index_offset + i,
                loss_value=0.0,
                metadata={},
            )
            self._entries[entry.id] = entry
            ids.append(entry.id)
        return ids

    def add_projected_batch(
        self,
        proj_gradients: np.ndarray,
        source_id: Optional[str] = None,
        sample_index_offset: int = 0,
        source_node_ids: Optional[list] = None,
    ) -> list[str]:
        """Store gradients that are *already* projected to ``proj_dim``.

        Used by the inline LLM logger, which performs the sparse-JL projection
        on-device (GPU) in the training framework and hands back only the small
        (n, proj_dim) result — avoiding a per-step host round-trip of the full
        per-sample gradient. No projection matrix is built here.
        """
        if proj_gradients.ndim == 1:
            proj_gradients = proj_gradients[None, :]
        n, k = proj_gradients.shape
        if k != self._proj_dim:
            raise ValueError(
                f"add_projected_batch: got proj_dim {k}, store expects {self._proj_dim}"
            )
        proj = np.ascontiguousarray(proj_gradients, dtype=np.float32)
        ids = []
        for i in range(n):
            entry = GradientLogEntry(
                proj_gradient=proj[i],
                source_node_id=(source_node_ids[i] if source_node_ids else None),
                source_id=source_id,
                sample_index=sample_index_offset + i,
            )
            self._entries[entry.id] = entry
            ids.append(entry.id)
        return ids

    def get_projected_matrix(self) -> np.ndarray:
        """Build the full (n_samples, proj_dim) gradient matrix for influence computation."""
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.sample_index)
        if not sorted_entries:
            return np.zeros((0, self._proj_dim), dtype=np.float32)
        return np.stack([e.proj_gradient for e in sorted_entries])

    def get_entry_by_index(self, sample_index: int) -> Optional[GradientLogEntry]:
        for e in self._entries.values():
            if e.sample_index == sample_index:
                return e
        return None

    def save(self, path: str) -> None:
        """Persist gradient store to disk as .npz + metadata."""
        matrix = self.get_projected_matrix()
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.sample_index)
        np.savez(
            path,
            gradient_matrix=matrix,
            source_node_ids=np.array([e.source_node_id if e.source_node_id is not None else -1 for e in sorted_entries]),
            source_ids=np.array([e.source_id or "" for e in sorted_entries]),
            sample_indices=np.array([e.sample_index for e in sorted_entries]),
            loss_values=np.array([e.loss_value for e in sorted_entries]),
        )
        logger.debug("[traceprop] GradientStore saved to %s (%d entries)", path, len(sorted_entries))

    @classmethod
    def load(cls, path: str, proj_dim: int = 4096) -> GradientStore:
        """Load a saved GradientStore from disk."""
        store = cls(proj_dim=proj_dim)
        data = np.load(path, allow_pickle=False)
        matrix = data["gradient_matrix"]
        for i, row in enumerate(matrix):
            nid = int(data["source_node_ids"][i])
            entry = GradientLogEntry(
                proj_gradient=row,
                source_node_id=nid if nid >= 0 else None,
                source_id=str(data["source_ids"][i]) or None,
                sample_index=int(data["sample_indices"][i]),
                loss_value=float(data["loss_values"][i]),
            )
            store._entries[entry.id] = entry
        store._proj_dim = proj_dim
        return store

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "proj_dim": self._proj_dim,
            "has_projection": self._projection is not None,
        }
