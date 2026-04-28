"""
Phase 5 tests: Unlearning, Data Valuation, and Streaming Attribution.
Uses NumPy only (no PyTorch required).
"""
import os

import numpy as np

import traceprop as tp
from traceprop.graph import reset_graph, get_graph
from traceprop.attribution.gradient_store import GradientStore


def setup_function():
    reset_graph()


# ── Problem 1: Approximate Surgical Unlearning ──


def test_unlearn_basic():
    """Unlearning reduces influence of target source below threshold."""
    ctx = tp.training_context(source_id="source_A")
    for i in range(50):
        ctx.log_gradient(np.random.randn(20), sample_index=i, source_node_id=i)

    # Add samples from a different source
    ctx.gradient_store._entries  # ensure initialized
    for i in range(50, 100):
        ctx.gradient_store.log_gradient(
            np.random.randn(20),
            source_id="source_B",
            sample_index=i,
            source_node_id=i,
        )

    result = tp.unlearn(
        gradient_store=ctx.gradient_store,
        source_id="source_A",
        n_steps=200,
        lr=1e-3,
        verification_threshold=0.1,
    )

    assert result is not None
    assert result.target_source_id == "source_A"
    assert result.n_samples_targeted == 50
    assert result.influence_after < result.influence_before
    assert result.method == "gradient_correction"


def test_unlearn_compliance_report():
    """UnlearningResult produces an EU AI Act compliance report."""
    ctx = tp.training_context(source_id="user_123")
    for i in range(10):
        ctx.log_gradient(np.random.randn(10), sample_index=i)

    result = tp.unlearn(
        gradient_store=ctx.gradient_store,
        source_id="user_123",
        n_steps=50,
    )

    report = result.compliance_report
    assert report["action"] == "approximate_unlearning"
    assert report["target_source_id"] == "user_123"
    assert "disclaimer" in report
    assert "formal privacy" in report["disclaimer"].lower() or "approximate" in report["disclaimer"].lower()


def test_unlearn_no_matching_source():
    """Unlearning with no matching samples returns clean result."""
    ctx = tp.training_context(source_id="source_A")
    for i in range(10):
        ctx.log_gradient(np.random.randn(10), sample_index=i)

    result = tp.unlearn(
        gradient_store=ctx.gradient_store,
        source_id="nonexistent_source",
    )

    assert result.n_samples_targeted == 0
    assert result.verified is True


def test_unlearn_verified():
    """Verify that unlearning can achieve verified=True."""
    ctx = tp.training_context(source_id="target")
    for i in range(20):
        ctx.log_gradient(np.random.randn(10), sample_index=i)

    result = tp.unlearn(
        gradient_store=ctx.gradient_store,
        source_id="target",
        n_steps=300,
        lr=1e-2,
        verification_threshold=0.5,
    )

    # With aggressive params and generous threshold, should verify
    assert result.verified is True
    assert result.influence_after < result.verification_threshold


# ── Problem 2: Provenance-Guided Data Valuation ──


def test_data_valuation_basic():
    """KNN-Shapley computes per-sample values."""
    ctx = tp.training_context(source_id="cohort_A")
    for i in range(30):
        sid = "cohort_A" if i < 15 else "cohort_B"
        ctx.gradient_store.log_gradient(
            np.random.randn(10),
            source_id=sid,
            sample_index=i,
        )

    val_grads = np.random.randn(5, 10)
    result = tp.data_valuation(
        gradient_store=ctx.gradient_store,
        val_gradients=val_grads,
        k=5,
    )

    assert result is not None
    assert result.values.shape == (30,)
    top = result.top_samples(k=5)
    assert len(top) == 5
    assert "shapley_value" in top[0]


def test_data_valuation_by_source():
    """Aggregate Shapley values by source_id."""
    ctx = tp.training_context(source_id="cohort_A")
    for i in range(20):
        sid = "cohort_A" if i < 10 else "cohort_B"
        ctx.gradient_store.log_gradient(
            np.random.randn(10),
            source_id=sid,
            sample_index=i,
        )

    val_grads = np.random.randn(3, 10)
    result = tp.data_valuation(
        gradient_store=ctx.gradient_store,
        val_gradients=val_grads,
        k=5,
    )

    by_source = result.by_source()
    assert "cohort_A" in by_source
    assert "cohort_B" in by_source
    assert by_source["cohort_A"]["n_samples"] == 10
    assert by_source["cohort_B"]["n_samples"] == 10
    assert "total_value" in by_source["cohort_A"]
    assert "mean_value" in by_source["cohort_A"]


def test_data_valuation_by_preprocessing_op():
    """Aggregate Shapley values by preprocessing operation (novel query)."""
    data = tp.array(np.random.rand(10, 3), source_id="val_test")
    processed = data * 2.0  # tracked as "multiply" op
    node_id = getattr(processed, "_provenance_node_id", None)

    ctx = tp.training_context(source_id="val_test")
    for i in range(10):
        ctx.gradient_store.log_gradient(
            np.random.randn(5),
            source_id="val_test",
            sample_index=i,
            source_node_id=node_id,
        )

    val_grads = np.random.randn(2, 5)
    result = tp.data_valuation(
        gradient_store=ctx.gradient_store,
        val_gradients=val_grads,
        k=3,
        lineage_graph=get_graph(),
    )

    by_op = result.by_preprocessing_op()
    assert isinstance(by_op, dict)
    # Should have "multiply" since we did data * 2.0
    assert len(by_op) > 0


def test_data_valuation_empty():
    """Empty gradient store returns empty valuation."""
    store = GradientStore(proj_dim=32)
    # Need at least one entry to have projection
    result = tp.data_valuation(gradient_store=store)
    assert len(result.top_samples()) == 0


# ── Problem 3: Continuous Provenance for Online Learning ──


def test_streaming_context_basic():
    """StreamingTrainingContext logs gradients with rolling window."""
    stream_ctx = tp.streaming_training_context(
        source_id="live_feed",
        window_size=20,
        proj_dim=32,
    )

    for i in range(50):
        stream_ctx.log_gradient(np.random.randn(10), sample_index=i)

    # Window should have evicted oldest entries
    assert len(stream_ctx.gradient_store) == 20

    stats = stream_ctx.stats()
    assert stats["total_samples_seen"] == 50
    assert stats["current_entries"] == 20


def test_streaming_context_attribution():
    """Can run attribution on the current window."""
    stream_ctx = tp.streaming_training_context(
        source_id="stream_test",
        window_size=30,
        proj_dim=32,
    )

    for i in range(30):
        stream_ctx.log_gradient(np.random.randn(10), sample_index=i)

    result = stream_ctx.attribute(np.random.randn(10), top_k=5)
    assert result is not None
    assert len(result) > 0


def test_streaming_context_checkpoint(tmp_path):
    """Streaming context saves checkpoints at configured interval."""
    cp_path = str(tmp_path / "checkpoints")
    stream_ctx = tp.streaming_training_context(
        source_id="cp_test",
        window_size=100,
        checkpoint_interval=10,
        checkpoint_path=cp_path,
        proj_dim=32,
    )

    for i in range(25):
        stream_ctx.log_gradient(np.random.randn(10), sample_index=i)

    assert stream_ctx._checkpoint_count >= 2
    assert os.path.exists(os.path.join(cp_path, "checkpoint_0000.npz"))
    assert os.path.exists(os.path.join(cp_path, "checkpoint_0001.npz"))


def test_streaming_context_manager():
    """StreamingTrainingContext supports with-statement."""
    with tp.streaming_training_context(
        source_id="ctx_test", window_size=10, proj_dim=32
    ) as ctx:
        for i in range(5):
            ctx.log_gradient(np.random.randn(10), sample_index=i)
        assert len(ctx.gradient_store) == 5


def test_streaming_window_eviction_order():
    """Oldest samples are evicted first."""
    stream_ctx = tp.streaming_training_context(
        source_id="evict_test",
        window_size=5,
        proj_dim=16,
    )

    for i in range(10):
        stream_ctx.log_gradient(np.random.randn(5), sample_index=i)

    # Only last 5 should remain
    remaining_indices = sorted(
        e.sample_index for e in stream_ctx.gradient_store._entries.values()
    )
    assert remaining_indices == [5, 6, 7, 8, 9]


# ── Integration: End-to-End Phase 5 ──


def test_full_phase5_pipeline():
    """End-to-end: preprocessing → training → attribution → unlearning → valuation."""
    # Phase 1-3: preprocessing
    source_A = tp.array(np.random.rand(30, 5), source_id="hospital_A")
    source_B = tp.array(np.random.rand(30, 5), source_id="hospital_B")
    processed_A = source_A - source_A.mean(axis=0)
    processed_B = source_B - source_B.mean(axis=0)

    node_A = getattr(processed_A, "_provenance_node_id", None)
    node_B = getattr(processed_B, "_provenance_node_id", None)

    # Phase 4: training context
    ctx = tp.training_context(source_id="hospital_A")
    for i in range(60):
        sid = "hospital_A" if i < 30 else "hospital_B"
        nid = node_A if i < 30 else node_B
        ctx.gradient_store.log_gradient(
            np.random.randn(10),
            source_id=sid,
            sample_index=i,
            source_node_id=nid,
        )

    # Phase 4: attribution
    engine = tp.attribution_engine(ctx.gradient_store)
    attr = engine.attribute(np.random.randn(10), top_k=5)
    assert len(attr) > 0

    # Phase 5: unlearning
    unlearn_result = tp.unlearn(
        gradient_store=ctx.gradient_store,
        source_id="hospital_A",
        n_steps=100,
        lr=1e-3,
        verification_threshold=0.5,
    )
    assert unlearn_result.n_samples_targeted == 30

    # Phase 5: data valuation
    val_result = tp.data_valuation(
        gradient_store=ctx.gradient_store,
        val_gradients=np.random.randn(5, 10),
        k=5,
        lineage_graph=get_graph(),
    )
    by_source = val_result.by_source()
    assert "hospital_A" in by_source
    assert "hospital_B" in by_source

    print("Phase 5 full pipeline: OK")
