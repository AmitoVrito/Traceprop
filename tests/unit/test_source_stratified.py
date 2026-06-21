"""Coverage for compute_source_stratified_scores + _resolve_source_key
+ top_k_influential — exercised by SAB v1 in research but not by the
core unit tests previously."""
from __future__ import annotations

import numpy as np
import pytest

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.influence import (
    compute_source_stratified_scores,
    top_k_influential,
    precompute_gram_factor,
    compute_influence_scores_multi_checkpoint,
)


@pytest.fixture
def small_store() -> GradientStore:
    """GradientStore with 12 samples across 3 source files."""
    store = GradientStore(proj_dim=16, seed=0)
    rng = np.random.default_rng(0)
    for i in range(12):
        g = rng.standard_normal(32).astype(np.float32)
        src = ["a.csv", "b.csv", "c.csv"][i % 3]
        store.log_gradient(g, source_id=src, sample_index=i)
    return store


def test_source_stratified_basic(small_store):
    test_grad = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    result = compute_source_stratified_scores(test_grad, small_store)
    assert "per_sample_scores" in result
    assert "per_source" in result
    assert "source_ranking" in result
    assert set(result["per_source"].keys()) == {"a.csv", "b.csv", "c.csv"}
    for k, v in result["per_source"].items():
        assert "mean_influence" in v
        assert "std_influence"  in v
        assert v["n_samples"] == 4
    assert len(result["source_ranking"]) == 3


def test_source_stratified_normalize(small_store):
    test_grad = np.random.default_rng(2).standard_normal(32).astype(np.float32)
    result = compute_source_stratified_scores(test_grad, small_store, normalize=True)
    means = [abs(v["mean_influence"]) for v in result["per_source"].values()]
    assert max(means) == pytest.approx(1.0, abs=1e-6)


def test_source_stratified_with_trak(small_store):
    test_grad = np.random.default_rng(3).standard_normal(32).astype(np.float32)
    result = compute_source_stratified_scores(
        test_grad, small_store, use_trak=True, lambda_factor=1e-3
    )
    assert len(result["per_source"]) == 3


def test_source_stratified_empty_store():
    store = GradientStore(proj_dim=8, seed=0)
    test_grad = np.zeros(16, dtype=np.float32)
    result = compute_source_stratified_scores(test_grad, store)
    assert result["per_source"] == {}
    assert result["source_ranking"] == []


def test_top_k_influential(small_store):
    scores = np.random.default_rng(4).standard_normal(len(small_store)).astype(np.float32)
    top = top_k_influential(scores, small_store, k=5)
    assert len(top) == 5
    # default `most_harmful=False` returns ascending (most negative first)
    influence = [r["influence_score"] for r in top]
    assert influence == sorted(influence)
    assert top[0]["rank"] == 1


def test_top_k_influential_most_harmful(small_store):
    scores = np.random.default_rng(4).standard_normal(len(small_store)).astype(np.float32)
    top = top_k_influential(scores, small_store, k=3, most_harmful=True)
    assert len(top) == 3
    # descending
    influence = [r["influence_score"] for r in top]
    assert influence == sorted(influence, reverse=True)


def test_top_k_influential_k_larger_than_store(small_store):
    scores = np.zeros(len(small_store), dtype=np.float32)
    top = top_k_influential(scores, small_store, k=100)
    assert len(top) == 12   # only 12 samples in store


def test_top_k_influential_empty_scores():
    store = GradientStore(proj_dim=8, seed=0)
    top = top_k_influential(np.array([], dtype=np.float32), store, k=5)
    assert top == []


def test_precompute_gram_factor(small_store):
    factor = precompute_gram_factor(small_store, lambda_factor=1e-3)
    # cho_factor returns a tuple (factor, lower)
    assert factor is not None
    assert len(factor) == 2


def test_multi_checkpoint_influence(small_store):
    # Two snapshot copies of the same store treated as two checkpoints
    test_grad = np.random.default_rng(5).standard_normal(32).astype(np.float32)
    scores = compute_influence_scores_multi_checkpoint(
        test_grad, [small_store, small_store]
    )
    assert len(scores) == len(small_store)
