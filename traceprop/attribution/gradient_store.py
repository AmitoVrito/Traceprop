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
