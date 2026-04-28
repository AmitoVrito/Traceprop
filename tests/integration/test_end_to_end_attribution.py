"""
Full end-to-end test: preprocessing → training → attribution → source trace.
Uses NumPy only (no PyTorch required) to test the core attribution pipeline.
"""
import numpy as np

import traceprop as tp
from traceprop.graph import reset_graph
from traceprop.attribution.gradient_store import GradientStore


def setup_function():
    reset_graph()


def test_full_attribution_pipeline():
    """CSV → preprocessing → simulated training → attribution → source trace."""
    source_A = tp.array(np.random.rand(50, 5), source_id="cohort_A")
    source_B = tp.array(np.random.rand(50, 5), source_id="cohort_B")

    norm_A = source_A - source_A.mean(axis=0)
    norm_B = source_B - source_B.mean(axis=0)

    ctx = tp.training_context(source_id="cohort_A")

    node_id_A = getattr(norm_A, "_provenance_node_id", None)
    node_id_B = getattr(norm_B, "_provenance_node_id", None)

    for i in range(100):
        fake_grad = np.random.randn(10)
        nid = node_id_A if i < 50 else node_id_B
        ctx.log_gradient(
            gradient=fake_grad,
            sample_index=i,
            source_node_id=nid,
            loss_value=np.random.rand(),
        )

    test_grad = np.random.randn(10)
    engine = tp.attribution_engine(ctx.gradient_store)
    result = engine.attribute(test_grad, top_k=5)

    assert result is not None
    assert len(result) > 0
    top = result.top(3)
    assert len(top) <= 3
    assert "influence_score" in top[0]
    assert "source_id" in top[0]
    assert len(result.sources()) > 0

    # Trace top result back through lineage graph to source
    trace = result.trace_to_file(rank=0)
    assert "influence_score" in trace
    assert "sources" in trace
    assert "ops" in trace


def test_gradient_store_save_load(tmp_path):
    store = GradientStore(proj_dim=64)
    for i in range(10):
        store.log_gradient(
            gradient=np.random.randn(20),
            source_node_id=i,
            source_id="test_source",
            sample_index=i,
        )
    path = str(tmp_path / "grads.npz")
    store.save(path)

    loaded = GradientStore.load(path, proj_dim=64)
    assert len(loaded) == 10
    assert loaded.stats()["entries"] == 10


def test_attribution_result_sources():
    reset_graph()
    ctx = tp.training_context(source_id="medical_records")
    for i in range(10):
        ctx.log_gradient(
            gradient=np.random.randn(5),
            sample_index=i,
            source_node_id=i,
        )

    engine = tp.attribution_engine(ctx.gradient_store)
    result = engine.attribute(np.random.randn(5), top_k=3)

    assert isinstance(result.sources(), list)
    assert len(result) > 0


def test_attribution_result_trace_to_file():
    """Verify trace resolves back through lineage to source."""
    reset_graph()
    raw = tp.array(np.random.rand(10, 3), source_id="medical_records")
    processed = raw - raw.mean(axis=0)
    node_id = getattr(processed, "_provenance_node_id", None)
    assert node_id is not None

    ctx = tp.training_context(source_id="medical_records")
    for i in range(10):
        ctx.log_gradient(
            gradient=np.random.randn(5),
            sample_index=i,
            source_node_id=node_id,
        )

    engine = tp.attribution_engine(ctx.gradient_store)
    result = engine.attribute(np.random.randn(5), top_k=3)

    trace = result.trace_to_file(rank=0)
    assert isinstance(trace, dict)
    assert "influence_score" in trace
    # Should have sources (upstream computation nodes)
    assert "sources" in trace
    assert isinstance(trace["sources"], list)
    # Should have ops (connected edges)
    assert "ops" in trace


def test_per_result_provenance():
    """Each attribution result entry should have a provenance object."""
    reset_graph()
    raw = tp.array(np.random.rand(10, 3), source_id="test_source")
    processed = raw * 2.0
    node_id = getattr(processed, "_provenance_node_id", None)

    ctx = tp.training_context(source_id="test_source")
    for i in range(10):
        ctx.log_gradient(np.random.randn(5), sample_index=i, source_node_id=node_id)

    engine = tp.attribution_engine(ctx.gradient_store)
    result = engine.attribute(np.random.randn(5), top_k=3)
    top = result.top(3)

    assert "provenance" in top[0]
    prov = top[0]["provenance"]
    assert prov is not None
    # Can query upstream through the provenance
    assert prov.node_id == node_id
    sources = prov.sources()
    assert isinstance(sources, list)


def test_context_manager_protocol():
    """TrainingContext supports with-statement."""
    reset_graph()
    with tp.training_context(source_id="ctx_test") as ctx:
        for i in range(5):
            ctx.log_gradient(np.random.randn(5), sample_index=i)

    assert len(ctx.gradient_store) == 5


def test_context_manager_with_X_y_train():
    """training_context accepts X_train and y_train."""
    reset_graph()
    X = tp.array(np.random.rand(10, 3))
    y = tp.array(np.random.rand(10))

    with tp.training_context(X_train=X, y_train=y, source_id="xy_test") as ctx:
        assert ctx.X_train is X
        assert ctx.y_train is y
        ctx.log_gradient(np.random.randn(5), sample_index=0)

    assert len(ctx.gradient_store) == 1


def test_training_attribution_on_provenance_view():
    """ProvenanceView.training_attribution() returns attribution results."""
    reset_graph()
    data = tp.array(np.random.rand(10, 3), source_id="train_attr_test")
    processed = data * 2.0
    node_id = getattr(processed, "_provenance_node_id", None)

    ctx = tp.training_context(source_id="train_attr_test")
    for i in range(10):
        ctx.log_gradient(np.random.randn(5), sample_index=i, source_node_id=node_id)

    view = tp.provenance(processed)
    result = view.training_attribution(
        gradient_store=ctx.gradient_store,
        test_gradient=np.random.randn(5),
        top_k=3,
    )

    assert result is not None
    assert len(result) > 0


def test_training_attribution_auto_store():
    """training_attribution() can use the last active store automatically."""
    reset_graph()
    data = tp.array(np.random.rand(5, 3))
    processed = data + 1.0

    ctx = tp.training_context(source_id="auto_store_test")
    for i in range(5):
        ctx.log_gradient(np.random.randn(5), sample_index=i)

    view = tp.provenance(processed)
    result = view.training_attribution(test_gradient=np.random.randn(5), top_k=2)

    assert result is not None
    assert len(result) > 0


def test_influence_graph():
    """InfluenceGraph unifies computation and gradient lineage."""
    reset_graph()
    from traceprop.attribution.influence_graph import InfluenceGraph
    from traceprop.graph import get_graph

    data = tp.array(np.random.rand(5, 3), source_id="ig_test")
    processed = data * 3.0
    node_id = getattr(processed, "_provenance_node_id", None)

    ig = InfluenceGraph(get_graph())

    # Add gradient edges
    ig.add_gradient_edge(
        source_node_id=0, target_node_id=node_id,
        influence_score=0.85, source_id="ig_test", sample_index=0,
    )
    ig.add_gradient_edge(
        source_node_id=1, target_node_id=node_id,
        influence_score=0.42, source_id="ig_test", sample_index=1,
    )

    stats = ig.stats()
    assert stats["gradient_edges"] == 2
    assert stats["nodes"] > 0

    ancestors = ig.gradient_ancestors(node_id)
    assert len(ancestors) == 2
    assert ancestors[0].influence_score == 0.85

    trace = ig.full_trace(node_id)
    assert "computation_ancestors" in trace
    assert "gradient_influences" in trace
    assert len(trace["gradient_influences"]) == 2


def test_random_projection_consistency():
    """Same seed produces identical projections."""
    from traceprop.attribution.gradient_store import RandomProjection
    p1 = RandomProjection(input_dim=100, proj_dim=32, seed=42)
    p2 = RandomProjection(input_dim=100, proj_dim=32, seed=42)
    g = np.random.randn(100).astype(np.float32)
    assert np.allclose(p1.project(g), p2.project(g))


def test_influence_scores_shape():
    from traceprop.attribution.influence import compute_influence_scores
    store = GradientStore(proj_dim=32)
    for i in range(20):
        store.log_gradient(np.random.randn(10), sample_index=i)

    scores = compute_influence_scores(
        test_gradient=np.random.randn(10),
        train_store=store,
        projection=store._projection,
    )
    assert scores.shape == (20,)


def test_empty_store_attribution():
    engine = tp.attribution_engine(GradientStore(proj_dim=32))
    result = engine.attribute(np.random.randn(10), top_k=5)
    assert result is not None
    assert len(result) == 0
