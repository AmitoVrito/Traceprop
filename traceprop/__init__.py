"""Traceprop — Computation-level lineage tracking for NumPy."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from traceprop.granularity import Granularity, get_granularity, set_granularity
from traceprop.graph import get_graph, reset_graph
from traceprop.query import ProvenanceView
from traceprop.tensor import ProvenanceTensor

__version__ = "0.7.0"


def array(data: Any, dtype=None, source_id: str | None = None, **kwargs) -> ProvenanceTensor:
    """Create a ProvenanceTensor (like np.array but with lineage tracking)."""
    arr = np.array(data, dtype=dtype, **kwargs)
    return ProvenanceTensor(arr, source_id=source_id)


def from_numpy(arr: np.ndarray, source_id: str | None = None) -> ProvenanceTensor:
    """Wrap an existing NumPy array with provenance tracking."""
    return ProvenanceTensor(arr, source_id=source_id)


def from_csv(path: str, dtype=None, delimiter: str = ",", skip_header: bool = True, source_id: str | None = None) -> ProvenanceTensor:
    """Load a CSV file as a ProvenanceTensor."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            rows.append([float(v) for v in row])
    arr = np.array(rows, dtype=dtype)
    t = ProvenanceTensor(arr, source_id=source_id)
    return t


def from_jax(data: Any, source_id: str | None = None):
    """Create a TrackedJaxArray with provenance tracking.

    Requires: jax
    """
    from traceprop.backends.jax_backend import jax_array
    return jax_array(data, source_id=source_id)


def from_torch(data, source_id: str | None = None):
    """Create a ProvenanceTorchTensor with provenance tracking.

    Requires: torch
    """
    from traceprop.backends.torch_backend import torch_tensor
    return torch_tensor(data, source_id=source_id)


def enable_compression():
    """Enable ProvRC range compression for large operations."""
    import traceprop.interceptor as _int_mod
    from traceprop.compression import CompressedLineageGraph
    cg = CompressedLineageGraph(get_graph())
    _int_mod._compressed_graph = cg
    return cg


def compliance_report(tensor, system_name: str, system_version: str, deployer_name: str, high_risk_category: str, output_path: str | None = None) -> dict | None:
    """Generate an EU AI Act compliance report for a tensor."""
    from traceprop.compliance.eu_ai_act import generate_compliance_report
    node_id = getattr(tensor, "_provenance_node_id", None)
    if node_id is None:
        return None
    return generate_compliance_report(
        node_id=node_id,
        graph=get_graph(),
        system_name=system_name,
        system_version=system_version,
        deployer_name=deployer_name,
        high_risk_category=high_risk_category,
        output_path=output_path,
    )


def training_context(
    model: Any | None = None,
    X_train: Any | None = None,
    y_train: Any | None = None,
    source_id: str | None = None,
    proj_dim: int = 4096,
):
    """Create a TrainingContext for recording per-sample gradients during training.

    Can be used as a context manager::

        with tp.training_context(model, X_train, y_train, source_id="data") as ctx:
            train(model, X_train, y_train)
    """
    from .attribution.training_context import TrainingContext
    return TrainingContext(
        model=model, X_train=X_train, y_train=y_train,
        source_id=source_id, proj_dim=proj_dim,
    )


def attribution_engine(gradient_store: Any, estimator: str = "dot", lambda_factor: float = 1e-3):
    """Create an AttributionEngine from a saved or in-memory GradientStore.

    estimator: "dot" (default, fast) or "trak" (regularised inverse-Gram, higher quality)
    """
    from .attribution.attribution_engine import AttributionEngine
    return AttributionEngine(gradient_store=gradient_store, estimator=estimator, lambda_factor=lambda_factor)


def unlearn(
    model: Any = None,
    gradient_store: Any = None,
    source_id: str = "",
    method: str = "gradient_correction",
    test_gradient: Any = None,
    n_steps: int = 100,
    lr: float = 1e-4,
    verification_threshold: float = 0.05,
):
    """Approximate surgical unlearning guided by provenance lineage.

    Identifies training samples from ``source_id``, computes their influence,
    and performs gradient correction to reduce it below the threshold.
    """
    from .unlearning.gradient_correction import gradient_correction_unlearn
    if gradient_store is None:
        from .attribution.training_context import _active_gradient_store
        gradient_store = _active_gradient_store()
    return gradient_correction_unlearn(
        model=model,
        gradient_store=gradient_store,
        target_source_id=source_id,
        test_gradient=test_gradient,
        n_steps=n_steps,
        lr=lr,
        verification_threshold=verification_threshold,
    )


def data_valuation(
    gradient_store: Any = None,
    val_gradients: Any = None,
    method: str = "knn_shapley",
    k: int = 10,
    lineage_graph: Any = None,
    **kwargs: Any,
):
    """Provenance-guided data valuation.

    Computes per-sample Shapley values and returns a ValuationResult
    that supports aggregation by source and preprocessing op.
    """
    from .valuation.knn_shapley import knn_shapley_values, ValuationResult
    import numpy as np
    if gradient_store is None:
        from .attribution.training_context import _active_gradient_store
        gradient_store = _active_gradient_store()
    if lineage_graph is None:
        lineage_graph = get_graph()
    if val_gradients is None:
        val_gradients = np.zeros((1, gradient_store._proj_dim), dtype=np.float32)
    values = knn_shapley_values(gradient_store, np.asarray(val_gradients), k=k)
    if values is None:
        values = np.array([], dtype=np.float32)
    return ValuationResult(values, gradient_store, lineage_graph)


def streaming_training_context(
    model: Any | None = None,
    source_id: str | None = None,
    window_size: int = 10_000,
    checkpoint_interval: int = 1_000,
    checkpoint_path: str | None = None,
    proj_dim: int = 4096,
):
    """Create a StreamingTrainingContext for online/continual learning."""
    from .attribution.streaming_context import StreamingTrainingContext
    return StreamingTrainingContext(
        model=model,
        source_id=source_id,
        window_size=window_size,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
        proj_dim=proj_dim,
    )


def provenance(tensor) -> ProvenanceView:
    """Get a ProvenanceView for querying the lineage of a tensor."""
    return ProvenanceView(tensor)


__all__ = [
    "array",
    "from_numpy",
    "from_csv",
    "from_jax",
    "from_torch",
    "enable_compression",
    "compliance_report",
    "provenance",
    "ProvenanceTensor",
    "ProvenanceView",
    "Granularity",
    "get_granularity",
    "set_granularity",
    "get_graph",
    "reset_graph",
    "training_context",
    "attribution_engine",
    "unlearn",
    "data_valuation",
    "streaming_training_context",
]
