"""Tests to cover uncovered lines in attribution modules.

Targets:
  - AttributionEngine: TRAK estimator, batch scoring, most_harmful, empty-store paths
  - MultiCheckpointAttributionEngine: all paths
  - influence.py: compute_trak_scores, precompute_gram_factor, multi-checkpoint
  - training_context.py: step() without torch, log_gradient with all params
  - streaming_context.py: step() without torch, eviction, checkpoint
"""
from __future__ import annotations

import numpy as np
import pytest

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import (
    AttributionEngine,
    AttributionResult,
    MultiCheckpointAttributionEngine,
)
from traceprop.attribution.influence import (
    compute_trak_scores,
    precompute_gram_factor,
    compute_influence_scores_multi_checkpoint,
)
import traceprop as tp


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_store(n: int = 20, grad_dim: int = 16, proj_dim: int = 32, seed: int = 0) -> GradientStore:
    rng = np.random.RandomState(seed)
    store = GradientStore(proj_dim=proj_dim, seed=seed)
    for i in range(n):
        g = rng.randn(grad_dim).astype(np.float32)
        store.log_gradient(g, sample_index=i, source_id="src")
    return store


def _test_grad(grad_dim: int = 16, seed: int = 99) -> np.ndarray:
    return np.random.RandomState(seed).randn(grad_dim).astype(np.float32)


# ── AttributionEngine — TRAK estimator ────────────────────────────────────────

class TestAttributionEngineTRAK:
    def test_attribute_scores_trak(self):
        store = _make_store()
        engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)
        scores = engine.attribute_scores(_test_grad())
        assert scores.shape == (20,)
        assert scores.dtype == np.float32

    def test_attribute_trak(self):
        store = _make_store()
        engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)
        result = engine.attribute(_test_grad(), top_k=5)
        assert len(result.top(5)) == 5
        assert all("influence_score" in e for e in result.top(5))

    def test_attribute_most_harmful(self):
        store = _make_store()
        engine = AttributionEngine(store, estimator="dot")
        result = engine.attribute(_test_grad(), top_k=5, most_harmful=True)
        assert len(result.top(5)) == 5

    def test_attribute_scores_trak_empty_store(self):
        store = GradientStore(proj_dim=32, seed=0)
        engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)
        scores = engine.attribute_scores(_test_grad())
        assert len(scores) == 0

    def test_attribute_no_projection(self):
        store = GradientStore(proj_dim=32, seed=0)
        engine = AttributionEngine(store, estimator="dot")
        result = engine.attribute(_test_grad(), top_k=5)
        assert len(result) == 0

    def test_attribute_scores_dot_no_projection(self):
        store = GradientStore(proj_dim=32, seed=0)
        engine = AttributionEngine(store, estimator="dot")
        scores = engine.attribute_scores(_test_grad())
        assert len(scores) == 0


# ── AttributionEngine — batch scoring ─────────────────────────────────────────

class TestAttributionEngineBatch:
    def test_batch_dot(self):
        store = _make_store(n=20, grad_dim=16)
        engine = AttributionEngine(store, estimator="dot")
        G = np.random.randn(5, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (5, 20)

    def test_batch_trak(self):
        store = _make_store(n=20, grad_dim=16)
        engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)
        G = np.random.randn(5, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (5, 20)

    def test_batch_empty_store(self):
        store = GradientStore(proj_dim=32, seed=0)
        engine = AttributionEngine(store, estimator="dot")
        G = np.random.randn(5, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape[0] == 5
        assert scores.shape[1] == 0

    def test_batch_no_projection(self):
        store = GradientStore(proj_dim=32, seed=0)
        engine = AttributionEngine(store, estimator="dot")
        G = np.random.randn(3, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (3, 0)


# ── AttributionResult ─────────────────────────────────────────────────────────

class TestAttributionResult:
    def test_sources(self):
        store = _make_store()
        engine = AttributionEngine(store, estimator="dot")
        result = engine.attribute(_test_grad(), top_k=10)
        srcs = result.sources()
        assert isinstance(srcs, list)
        assert "src" in srcs

    def test_trace_to_file_out_of_range(self):
        store = _make_store()
        engine = AttributionEngine(store, estimator="dot")
        result = engine.attribute(_test_grad(), top_k=5)
        trace = result.trace_to_file(rank=999)
        assert trace == {}

    def test_trace_to_file_no_node_id(self):
        result = AttributionResult(
            [{"rank": 0, "influence_score": 0.5, "sample_index": 0}],
            tp.get_graph(),
        )
        trace = result.trace_to_file(rank=0)
        assert "error" in trace


# ── MultiCheckpointAttributionEngine ─────────────────────────────────────────

class TestMultiCheckpoint:
    def _make_stores(self, n_ckpts: int = 3) -> list[GradientStore]:
        return [_make_store(n=20, grad_dim=16, seed=k) for k in range(n_ckpts)]

    def test_attribute_scores_dot(self):
        stores = self._make_stores()
        engine = MultiCheckpointAttributionEngine(stores, use_trak=False)
        scores = engine.attribute_scores(_test_grad())
        assert scores.shape == (20,)

    def test_attribute_scores_trak(self):
        stores = self._make_stores()
        engine = MultiCheckpointAttributionEngine(stores, use_trak=True, lambda_factor=1e-3)
        scores = engine.attribute_scores(_test_grad())
        assert scores.shape == (20,)

    def test_attribute_batch_dot(self):
        stores = self._make_stores()
        engine = MultiCheckpointAttributionEngine(stores, use_trak=False)
        G = np.random.randn(4, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (4, 20)

    def test_attribute_batch_trak(self):
        stores = self._make_stores()
        engine = MultiCheckpointAttributionEngine(stores, use_trak=True, lambda_factor=1e-3)
        G = np.random.randn(4, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (4, 20)

    def test_attribute_result(self):
        stores = self._make_stores()
        engine = MultiCheckpointAttributionEngine(stores, use_trak=False)
        result = engine.attribute(_test_grad(), top_k=5)
        assert len(result.top(5)) == 5

    def test_attribute_empty_stores(self):
        empty = [GradientStore(proj_dim=32, seed=0)]
        engine = MultiCheckpointAttributionEngine(empty, use_trak=False)
        scores = engine.attribute_scores(_test_grad())
        assert len(scores) == 0

    def test_attribute_batch_empty_stores(self):
        empty = [GradientStore(proj_dim=32, seed=0)]
        engine = MultiCheckpointAttributionEngine(empty, use_trak=False)
        G = np.random.randn(3, 16).astype(np.float32)
        scores = engine.attribute_scores_batch(G)
        assert scores.shape == (3, 0)


# ── influence.py direct functions ────────────────────────────────────────────

class TestInfluenceFunctions:
    def test_compute_trak_scores(self):
        store = _make_store()
        scores = compute_trak_scores(_test_grad(), store, lambda_factor=1e-3)
        assert scores.shape == (20,)

    def test_compute_trak_scores_with_gram_factor(self):
        store = _make_store()
        gf = precompute_gram_factor(store, lambda_factor=1e-3)
        scores = compute_trak_scores(_test_grad(), store, lambda_factor=1e-3, gram_factor=gf)
        assert scores.shape == (20,)

    def test_compute_trak_scores_no_normalize(self):
        store = _make_store()
        scores = compute_trak_scores(_test_grad(), store, normalize=False)
        assert scores.shape == (20,)

    def test_compute_trak_scores_empty_store(self):
        store = GradientStore(proj_dim=32, seed=0)
        scores = compute_trak_scores(_test_grad(), store)
        assert len(scores) == 0

    def test_compute_trak_scores_no_projection(self):
        store = GradientStore(proj_dim=32, seed=0)
        scores = compute_trak_scores(_test_grad(), store)
        assert len(scores) == 0

    def test_precompute_gram_factor(self):
        store = _make_store()
        gf = precompute_gram_factor(store, lambda_factor=1e-3)
        assert gf is not None

    def test_precompute_gram_factor_empty(self):
        store = GradientStore(proj_dim=32, seed=0)
        gf = precompute_gram_factor(store)
        assert gf is None

    def test_multi_checkpoint_dot(self):
        stores = [_make_store(seed=k) for k in range(3)]
        scores = compute_influence_scores_multi_checkpoint(_test_grad(), stores, use_trak=False)
        assert scores.shape == (20,)

    def test_multi_checkpoint_trak(self):
        stores = [_make_store(seed=k) for k in range(3)]
        gram_factors = [precompute_gram_factor(s) for s in stores]
        scores = compute_influence_scores_multi_checkpoint(
            _test_grad(), stores, use_trak=True, gram_factors=gram_factors
        )
        assert scores.shape == (20,)

    def test_multi_checkpoint_empty(self):
        stores = [GradientStore(proj_dim=32, seed=0)]
        scores = compute_influence_scores_multi_checkpoint(_test_grad(), stores)
        assert len(scores) == 0


# ── training_context.py — step() without torch ───────────────────────────────

class TestTrainingContextCoverage:
    def test_step_without_torch(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from traceprop.attribution.training_context import TrainingContext
        ctx = TrainingContext(source_id="test", proj_dim=32)

        class FakeLoss:
            def backward(self, retain_graph=False):
                pass
            def detach(self):
                return self
            def cpu(self):
                return self
            def mean(self):
                return 0.5
            @property
            def shape(self):
                return ()

        ctx.step(FakeLoss())
        assert len(ctx.gradient_store) == 0

    def test_log_gradient_with_source_node_id(self):
        from traceprop.attribution.training_context import TrainingContext
        ctx = TrainingContext(source_id="test", proj_dim=32)
        g = np.random.randn(16).astype(np.float32)
        ctx.log_gradient(g, sample_index=0, source_node_id=42)
        assert len(ctx.gradient_store) == 1


# ── streaming_context.py — step() without torch ──────────────────────────────

class TestStreamingContextCoverage:
    def test_step_without_torch(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from traceprop.attribution.streaming_context import StreamingTrainingContext
        ctx = StreamingTrainingContext(source_id="test", proj_dim=32, window_size=10)

        class FakeLoss:
            def backward(self, retain_graph=False):
                pass
            def detach(self):
                return self
            def cpu(self):
                return self
            def mean(self):
                return 0.5
            @property
            def shape(self):
                return ()

        ctx.step(FakeLoss())
        assert len(ctx.gradient_store) == 0
