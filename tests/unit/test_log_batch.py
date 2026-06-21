"""Tests for the vectorised log_batch / project_batch APIs added in 0.7.0."""
from __future__ import annotations

import numpy as np
import pytest

from traceprop.attribution.gradient_store import GradientStore, RandomProjection


def test_random_projection_project_batch_shape():
    rp = RandomProjection(input_dim=100, proj_dim=64, seed=0)
    g = np.random.default_rng(0).standard_normal((32, 100)).astype(np.float32)
    out = rp.project_batch(g)
    assert out.shape == (32, 64)
    assert out.dtype == np.float32


def test_random_projection_project_batch_matches_per_sample():
    rp = RandomProjection(input_dim=50, proj_dim=16, seed=42)
    g = np.random.default_rng(1).standard_normal((10, 50)).astype(np.float32)
    batched = rp.project_batch(g)
    per_sample = np.stack([rp.project(g[i]) for i in range(len(g))])
    np.testing.assert_allclose(batched, per_sample, atol=1e-5)


def test_random_projection_project_batch_validates_shape():
    rp = RandomProjection(input_dim=20, proj_dim=8, seed=0)
    with pytest.raises(ValueError):
        rp.project_batch(np.zeros((4, 99), dtype=np.float32))   # wrong cols
    with pytest.raises(ValueError):
        rp.project_batch(np.zeros((4,), dtype=np.float32))      # not 2D


def test_random_projection_project_batch_casts_dtype():
    rp = RandomProjection(input_dim=10, proj_dim=4, seed=0)
    g_f64 = np.random.default_rng(0).standard_normal((3, 10))   # float64
    out = rp.project_batch(g_f64)
    assert out.shape == (3, 4)
    assert out.dtype == np.float32


def test_gradient_store_log_batch_basic():
    store = GradientStore(proj_dim=32, seed=0)
    g = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    ids = store.log_batch(g, source_id="ds1", sample_index_offset=0)
    assert len(ids) == 16
    assert len(store) == 16


def test_gradient_store_log_batch_offset():
    store = GradientStore(proj_dim=32, seed=0)
    g = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    store.log_batch(g, source_id="ds1", sample_index_offset=100)
    matrix = store.get_projected_matrix()
    assert matrix.shape == (8, 32)
    # Entries should have sample_index 100..107
    indices = sorted(e.sample_index for e in store._entries.values())
    assert indices == list(range(100, 108))


def test_gradient_store_log_batch_1d_promoted():
    store = GradientStore(proj_dim=32, seed=0)
    g = np.random.default_rng(0).standard_normal(64).astype(np.float32)   # 1D
    ids = store.log_batch(g, source_id="ds1")
    assert len(ids) == 1
    assert len(store) == 1


def test_gradient_store_log_batch_equivalent_to_log_gradient():
    """Vectorised path produces the same projected gradients as the
    per-sample loop (within float32 tolerance)."""
    rng = np.random.default_rng(7)
    g = rng.standard_normal((20, 100)).astype(np.float32)

    s_batch = GradientStore(proj_dim=64, seed=99)
    s_batch.log_batch(g, source_id="x", sample_index_offset=0)

    s_loop = GradientStore(proj_dim=64, seed=99)
    for i in range(20):
        s_loop.log_gradient(g[i], source_id="x", sample_index=i)

    m_batch = s_batch.get_projected_matrix()
    m_loop  = s_loop.get_projected_matrix()
    np.testing.assert_allclose(m_batch, m_loop, atol=1e-4)


def test_gradient_store_log_batch_with_source_node_ids():
    store = GradientStore(proj_dim=16, seed=0)
    g = np.random.default_rng(0).standard_normal((5, 32)).astype(np.float32)
    node_ids = [10, 20, 30, 40, 50]
    ids = store.log_batch(g, source_id="ds1", source_node_ids=node_ids,
                          sample_index_offset=0)
    assert len(ids) == 5
    # Spot-check that one of them carries the right source_node_id
    by_idx = {e.sample_index: e for e in store._entries.values()}
    assert by_idx[2].source_node_id == 30


def test_gradient_store_log_batch_lazy_projection_init():
    """log_batch creates the projection on first call with the
    correct input_dim, just like log_gradient."""
    store = GradientStore(proj_dim=16, seed=0)
    assert store._projection is None
    g = np.random.default_rng(0).standard_normal((3, 25)).astype(np.float32)
    store.log_batch(g, source_id="x")
    assert store._projection is not None
    assert store._projection.input_dim == 25
    assert store._projection.proj_dim  == 16
