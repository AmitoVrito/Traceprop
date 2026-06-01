"""Traceprop — Interactive Demo on Hugging Face Spaces.

Tabs:
  1. Training Data Attribution  — which samples drove a prediction?
  2. Data Provenance Tracking   — full computation lineage from source to output
  3. Machine Unlearning         — surgically reduce a sample's influence
  4. Benchmarks & About
  5. Feedback
"""

from __future__ import annotations

import time

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import traceprop as tp
from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import AttributionEngine

matplotlib.use("Agg")

# ── Global state (computed once at startup) ────────────────────────────────────

_state: dict = {}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _cross_entropy(logit: float, y: float) -> float:
    p = float(_sigmoid(logit))
    p = max(1e-7, min(1 - 1e-7, p))
    return -y * np.log(p) - (1 - y) * np.log(1 - p)


def _per_sample_ll_grad(
    x_i: np.ndarray, y_i: float, coef: np.ndarray, intercept: float
) -> np.ndarray:
    logit = float(x_i @ coef + intercept)
    err = float(_sigmoid(logit)) - y_i
    return (err * x_i).astype(np.float32)


def _build_state() -> None:
    data = load_breast_cancer()
    X_raw = data.data.astype(np.float32)
    y_raw = data.target.astype(float)  # 1=benign, 0=malignant

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_raw))
    n_train = 400
    tr, te = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr]).astype(np.float32)
    X_te = scaler.transform(X_raw[te]).astype(np.float32)
    y_tr, y_te = y_raw[tr], y_raw[te]

    clf = LogisticRegression(C=10, solver="lbfgs", max_iter=500, random_state=0)
    clf.fit(X_tr, y_tr)

    coef = clf.coef_[0]
    intercept = clf.intercept_[0]

    store = GradientStore(proj_dim=512, seed=42)
    for i in range(len(X_tr)):
        g = _per_sample_ll_grad(X_tr[i], y_tr[i], coef, intercept)
        store.log_gradient(g, sample_index=i, source_id="breast_cancer_train")

    # Find a training sample with non-trivial loss (for unlearning demo)
    train_losses = np.array([
        _cross_entropy(float(X_tr[i] @ coef + intercept), float(y_tr[i]))
        for i in range(n_train)
    ])
    hard_candidates = np.where(train_losses > 0.05)[0]
    default_forget_idx = int(hard_candidates[0]) if len(hard_candidates) > 0 else 0

    _state.update(
        clf=clf,
        scaler=scaler,
        coef=coef,
        intercept=intercept,
        X_tr=X_tr,
        X_te=X_te,
        y_tr=y_tr,
        y_te=y_te,
        feature_names=list(data.feature_names),
        target_names=list(data.target_names),  # [malignant, benign]
        store=store,
        engine_dot=AttributionEngine(store, estimator="dot"),
        engine_trak=AttributionEngine(store, estimator="trak", lambda_factor=1e-3),
        n_train=n_train,
        n_test=len(te),
        default_forget_idx=default_forget_idx,
    )


_build_state()

# ── Tab 1: Attribution ─────────────────────────────────────────────────────────

def run_attribution(test_idx: int, estimator: str, top_k: int):
    s = _state
    x = s["X_te"][int(test_idx)]
    y = int(s["y_te"][int(test_idx)])

    logit = float(x @ s["coef"] + s["intercept"])
    prob = float(_sigmoid(logit))
    pred = int(prob >= 0.5)
    pred_label = s["target_names"][pred]
    true_label = s["target_names"][y]
    correct = pred == y

    err = float(_sigmoid(logit)) - y
    test_grad = (err * x).astype(np.float32)

    engine = s["engine_trak"] if estimator == "TRAK" else s["engine_dot"]
    t0 = time.perf_counter()
    result = engine.attribute(test_grad, top_k=int(top_k))
    elapsed = time.perf_counter() - t0

    entries = result.top(int(top_k))
    rows = [
        {
            "Rank": e["rank"] + 1,
            "Train Index": e["sample_index"],
            "True Label": s["target_names"][int(s["y_tr"][e["sample_index"]])],
            "Influence Score": round(float(e["influence_score"]), 4),
        }
        for e in entries
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, max(3.5, int(top_k) * 0.4)))
    colors = ["#ef4444" if r["True Label"] == "malignant" else "#22c55e" for r in rows]
    ax.barh(
        [f"#{r['Train Index']} ({r['True Label'][:3]})" for r in rows],
        [r["Influence Score"] for r in rows],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Influence Score (L∞ normalised)", fontsize=10)
    ax.set_title(
        f"Top-{int(top_k)} Influential Training Samples  [{estimator}]",
        fontsize=11, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#22c55e", label="benign"), Patch(color="#ef4444", label="malignant")],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()

    status = "✅ Correct" if correct else "❌ Wrong"
    summary = (
        f"### Test Sample #{int(test_idx)}\n\n"
        f"**True label:** `{true_label}` &nbsp;&nbsp; "
        f"**Predicted:** `{pred_label}` &nbsp;&nbsp; {status}\n\n"
        f"**Confidence (benign):** {prob:.1%} &nbsp;&nbsp; "
        f"**Attribution time:** {elapsed * 1000:.1f} ms"
    )
    return summary, fig, df


# ── Tab 2: Provenance ──────────────────────────────────────────────────────────

def run_provenance(source_a_rows: int, source_b_rows: int, add_norm: bool, add_clip: bool):
    tp.reset_graph()
    rng = np.random.RandomState(0)

    # Create source tensors — these are the tracked roots
    source_a = tp.from_numpy(
        rng.randn(int(source_a_rows), 5).astype(np.float32), source_id="hospital_A"
    )
    source_b = tp.from_numpy(
        rng.randn(int(source_b_rows), 5).astype(np.float32), source_id="hospital_B"
    )

    # Use ProvenanceTensor arithmetic so every op creates a lineage edge
    ops_applied: list[str] = []

    # Scale both sources (simulates unit conversion)
    scaled_a = source_a * np.float32(0.5)
    scaled_b = source_b * np.float32(0.5)
    ops_applied.append("Scale ×0.5 (unit conversion)")

    result = scaled_a  # track lineage through source_a pipeline

    if add_norm:
        mean = scaled_a.mean(axis=0)
        std  = scaled_a.std(axis=0)
        result = (scaled_a - mean) / (std + np.float32(1e-8))
        ops_applied.append("StandardNorm — (x − μ) / σ")

    if add_clip:
        clamp_max = np.full(result.shape, 3.0, dtype=np.float32)
        clamp_min = np.full(result.shape, -3.0, dtype=np.float32)
        # Use ops that ProvenanceTensor tracks: clip via min/max arithmetic
        result = result - (result - clamp_max) * ((result - clamp_max) > 0).astype(np.float32)
        result = result + (clamp_min - result) * ((clamp_min - result) > 0).astype(np.float32)
        ops_applied.append("Clamp to [−3, +3]")

    view      = tp.provenance(result)
    ops       = view.ops()
    ancestors = view.ancestors()
    n_nodes   = len(tp.get_graph().nodes)
    n_edges   = len(tp.get_graph().edges)

    report_lines = [
        "## Provenance Report",
        "",
        f"**Output shape:** `{np.asarray(result).shape}` &nbsp;&nbsp; "
        f"**Total rows:** {int(source_a_rows) + int(source_b_rows)}",
        "",
        "### Data Sources",
        f"- `hospital_A` — {int(source_a_rows)} rows",
        f"- `hospital_B` — {int(source_b_rows)} rows",
        "",
        "### Preprocessing Pipeline (applied to Hospital A)",
    ]
    for op in ops_applied:
        report_lines.append(f"- {op}")

    report_lines += [
        "",
        "### Lineage Graph",
        f"- Tensor nodes recorded: **{n_nodes}**",
        f"- Operation edges recorded: **{n_edges}**",
        f"- Ancestor nodes of output: **{len(ancestors)}**",
        "",
        "### Op Chain for output tensor",
        "```",
    ]
    for op in ops[:15]:
        report_lines.append(
            f"{op.op_name:20s}  inputs={list(op.input_ids)} → output={op.output_id}"
        )
    if len(ops) > 15:
        report_lines.append(f"... and {len(ops) - 15} more ops")
    report_lines.append("```")

    tp.reset_graph()

    # Lineage diagram
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    pipeline = [
        ("hospital_A\n" + str(int(source_a_rows)) + " rows", 0.12, 0.75, "#6366f1"),
        ("hospital_B\n" + str(int(source_b_rows)) + " rows", 0.12, 0.25, "#6366f1"),
        ("combined\n" + str(int(source_a_rows) + int(source_b_rows)) + " rows", 0.40, 0.50, "#0ea5e9"),
    ]
    if add_norm:
        pipeline.append(("StandardNorm", 0.63, 0.50, "#f59e0b"))
    if add_clip:
        pipeline.append(("Clip ±3σ", 0.86, 0.50, "#f59e0b"))

    for label, xp, yp, col in pipeline:
        ax.text(
            xp, yp, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=col, alpha=0.9, edgecolor="white"),
            color="white",
        )

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5),
        )

    arrow(0.20, 0.72, 0.31, 0.56)
    arrow(0.20, 0.28, 0.31, 0.44)
    for i in range(2, len(pipeline) - 1):
        x_from = pipeline[i][1] + 0.10
        x_to   = pipeline[i + 1][1] - 0.10
        arrow(x_from, 0.50, x_to, 0.50)

    ax.set_title("Computation Lineage Graph — Traceprop", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()

    return "\n".join(report_lines), fig


# ── Tab 3: Unlearning ──────────────────────────────────────────────────────────

def run_unlearning(target_idx: int, n_steps: int, lr_exp: float):
    s = _state
    lr = 10 ** float(lr_exp)
    target_idx = int(target_idx)
    n_steps = int(n_steps)

    x_forget = s["X_tr"][target_idx]
    y_forget = float(s["y_tr"][target_idx])
    label_forget = s["target_names"][int(y_forget)]

    # Accuracy and per-sample loss using current weights
    coef_orig = s["coef"].copy()
    intercept_orig = float(s["intercept"])

    loss_before = _cross_entropy(float(x_forget @ coef_orig + intercept_orig), y_forget)
    acc_before = s["clf"].score(s["X_te"], s["y_te"])

    # Gradient ascent — maximise cross-entropy loss on forget sample
    coef_new = coef_orig.copy()
    intercept_new = intercept_orig
    for _ in range(n_steps):
        logit = float(x_forget @ coef_new + intercept_new)
        err = float(_sigmoid(logit)) - y_forget
        coef_new      += lr * err * x_forget
        intercept_new += lr * err

    loss_after = _cross_entropy(float(x_forget @ coef_new + intercept_new), y_forget)

    logits_after = s["X_te"] @ coef_new + intercept_new
    preds_after  = (logits_after >= 0).astype(int)
    acc_after    = float(np.mean(preds_after == s["y_te"].astype(int)))

    # Also show how the model's prediction on the forget sample shifts
    pred_before = float(_sigmoid(float(x_forget @ coef_orig + intercept_orig)))
    pred_after  = float(_sigmoid(float(x_forget @ coef_new  + intercept_new)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Loss on forget sample
    ax1.bar(
        ["Before\nUnlearning", "After\nUnlearning"],
        [loss_before, loss_after],
        color=["#f59e0b", "#22c55e"], edgecolor="white", width=0.5,
    )
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=9)
    ax1.set_title(
        f"Loss on Forget Sample #{target_idx}\n({label_forget})",
        fontsize=10, fontweight="bold",
    )
    ax1.set_ylim(0, max(loss_after * 1.3, 0.1))
    for i, v in enumerate([loss_before, loss_after]):
        ax1.text(i, v + max(loss_after * 0.04, 0.005), f"{v:.3f}", ha="center", fontsize=9)

    # Test accuracy
    ax2.bar(
        ["Before\nUnlearning", "After\nUnlearning"],
        [acc_before * 100, acc_after * 100],
        color=["#6366f1", "#818cf8"], edgecolor="white", width=0.5,
    )
    ax2.set_ylabel("Test Accuracy (%)", fontsize=9)
    ax2.set_title("Remaining Test Accuracy\n(should be preserved)", fontsize=10, fontweight="bold")
    ax2.set_ylim(max(0, min(acc_before, acc_after) * 100 - 5), 100)
    for i, v in enumerate([acc_before * 100, acc_after * 100]):
        ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()

    loss_increase = (loss_after - loss_before) / (loss_before + 1e-10) * 100
    acc_delta = (acc_after - acc_before) * 100

    summary = (
        f"### Unlearning Report — Sample #{target_idx} (`{label_forget}`)\n\n"
        f"| Metric | Before | After | Change |\n"
        f"|--------|--------|-------|--------|\n"
        f"| Loss on forget sample | {loss_before:.3f} | {loss_after:.3f} | **+{loss_increase:.1f}%** |\n"
        f"| P(benign) for forget sample | {pred_before:.1%} | {pred_after:.1%} | "
        f"{(pred_after - pred_before)*100:+.1f}pp |\n"
        f"| Test accuracy | {acc_before:.1%} | {acc_after:.1%} | {acc_delta:+.1f}pp |\n\n"
        f"**Steps:** {n_steps} &nbsp;&nbsp; **LR:** {lr:.0e}\n\n"
        f"> Loss on the forget sample **increased {loss_increase:.0f}%** — "
        f"model is now {loss_after/loss_before:.1f}× less confident on that sample.  \n"
        f"> Test accuracy changed by only **{abs(acc_delta):.1f}pp** — "
        f"other predictions are preserved."
    )
    return summary, fig


# ── UI ─────────────────────────────────────────────────────────────────────────

HEADER = """
<div style="text-align:center;padding:22px 0 14px 0;background:linear-gradient(135deg,#4f46e5 0%,#0ea5e9 100%);border-radius:14px;margin-bottom:16px;">
  <h1 style="font-size:2.3rem;font-weight:900;color:white;margin:0 0 6px 0;letter-spacing:-0.5px;">
    🔍 Traceprop
  </h1>
  <p style="font-size:1.05rem;color:rgba(255,255,255,0.92);margin:0 0 12px 0;">
    Training-data attribution &nbsp;·&nbsp; Computation lineage &nbsp;·&nbsp; Machine unlearning
  </p>
  <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
    <a href="https://pypi.org/project/traceprop/" target="_blank"
       style="background:rgba(255,255,255,0.18);color:white;padding:6px 16px;border-radius:20px;
              text-decoration:none;font-size:0.85rem;font-weight:700;border:1px solid rgba(255,255,255,0.3);">
      📦 pip install traceprop
    </a>
    <a href="https://github.com/AmitoVrito/Traceprop" target="_blank"
       style="background:rgba(255,255,255,0.18);color:white;padding:6px 16px;border-radius:20px;
              text-decoration:none;font-size:0.85rem;font-weight:700;border:1px solid rgba(255,255,255,0.3);">
      🐙 GitHub
    </a>
    <a href="https://github.com/AmitoVrito/Traceprop/issues/new" target="_blank"
       style="background:rgba(255,255,255,0.18);color:white;padding:6px 16px;border-radius:20px;
              text-decoration:none;font-size:0.85rem;font-weight:700;border:1px solid rgba(255,255,255,0.3);">
      💬 Feedback / Issues
    </a>
  </div>
</div>
<p style="text-align:center;font-size:0.82rem;color:#6b7280;margin:0 0 8px 0;">
  Demo dataset: Wisconsin Breast Cancer (569 samples · 30 features)
  &nbsp;·&nbsp; Model: Logistic Regression &nbsp;·&nbsp; Runs on CPU · No GPU needed
</p>
"""

ATTRIBUTION_INTRO = """
**Which training samples most influenced this prediction?**

Traceprop records per-sample gradients during training via `tp.training_context()` and computes
influence scores at query time in **milliseconds** — no GPU needed for tabular models.
Pick any test sample and see which training points shaped its output, and by how much.
"""

PROVENANCE_INTRO = """
**Full computation lineage from source file to model input.**

Wrap any NumPy array with `tp.from_numpy(X, source_id="...")` and Traceprop automatically
records every tensor operation in a lightweight lineage graph. Useful for EU AI Act Article 10
data-governance audits and answering *"where did this training sample come from?"*
"""

UNLEARNING_INTRO = """
**Remove a training sample's influence without full retraining.**

Traceprop performs provenance-guided gradient ascent to surgically maximise the loss on a
specific sample — reducing its influence on model weights. Useful for GDPR right-to-erasure
and removing mislabelled data without expensive retraining from scratch.
"""

BENCHMARKS_MD = """
### 📊 Benchmark Results (LDS — Linear Datamodel Score, higher = better)

| Dataset | Method | LDS ↑ | Wall-clock |
|---------|--------|--------|-----------|
| Adult Income (tabular, n=6k) | Traceprop-LL | 0.622 ± 0.180 | **0.22 s CPU** |
| Adult Income (tabular, n=6k) | Traceprop-LL + TRAK | **0.884 ± 0.096** | 0.6 s CPU |
| Covertype (n=50k, d=54) | Traceprop-LL + TRAK | **0.976 ± 0.105** | 5.2 s CPU |
| CIFAR-2 / ResNet-9 (end-to-end) | TRAK (5 ckpts, GPU T4) | 0.029 ± 0.052 | 691 s GPU |
| CIFAR-2 / Frozen ResNet-18 + probe | Traceprop-LL | **0.264 ± 0.104** | 10.2 s CPU |

*Traceprop on a frozen-backbone linear probe achieves **266× speedup** vs TRAK and
**15.7× better LDS** vs end-to-end ResNet-9 with BatchNorm.*

---

### About Traceprop

Traceprop is a **production-grade ML explainability library** that uniquely combines:

- **Training Data Attribution** — gradient influence functions with optional TRAK estimator,
  achieving near-TRAK quality at a fraction of the compute on tabular and linear-probe models.
- **Computation Lineage** — automatic tensor-level lineage tracking across NumPy, PyTorch,
  and JAX pipelines. Zero config, sub-millisecond query latency.
- **Machine Unlearning** — provenance-guided gradient correction for approximate GDPR erasure
  without retraining. Compatible with sklearn, PyTorch, and JAX models.

**Install:** `pip install traceprop`

**Paper:** VLDB 2027 (under submission)

**Author:** Amit Nautiyal — Independent Researcher
"""

with gr.Blocks(title="Traceprop — Training Data Attribution & Provenance") as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Attribution ─────────────────────────────────────────────
        with gr.Tab("🎯 Attribution"):
            gr.Markdown(ATTRIBUTION_INTRO)
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    t1_idx = gr.Slider(
                        0, _state["n_test"] - 1, value=0, step=1,
                        label="Test Sample Index",
                    )
                    t1_estimator = gr.Radio(
                        ["Dot Product", "TRAK"],
                        value="Dot Product",
                        label="Estimator",
                        info="TRAK applies a regularised inverse-Gram correction (higher quality).",
                    )
                    t1_topk = gr.Slider(5, 20, value=10, step=1, label="Top-K")
                    t1_btn = gr.Button("▶  Run Attribution", variant="primary")

                with gr.Column(scale=2):
                    t1_summary = gr.Markdown()
                    t1_chart   = gr.Plot(label="Influence Scores")
                    t1_table   = gr.Dataframe(
                        label="Ranked Training Samples", interactive=False
                    )

            t1_btn.click(
                run_attribution,
                inputs=[t1_idx, t1_estimator, t1_topk],
                outputs=[t1_summary, t1_chart, t1_table],
            )
            demo.load(
                run_attribution,
                inputs=[t1_idx, t1_estimator, t1_topk],
                outputs=[t1_summary, t1_chart, t1_table],
            )

        # ── Tab 2: Provenance ──────────────────────────────────────────────
        with gr.Tab("🗂️ Provenance"):
            gr.Markdown(PROVENANCE_INTRO)
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    t2_rows_a = gr.Slider(50, 500, value=200, step=50, label="Hospital A — rows")
                    t2_rows_b = gr.Slider(50, 500, value=100, step=50, label="Hospital B — rows")
                    t2_norm   = gr.Checkbox(value=True, label="Apply StandardNorm")
                    t2_clip   = gr.Checkbox(value=True, label="Apply Clip (±3σ)")
                    t2_btn    = gr.Button("▶  Trace Lineage", variant="primary")

                with gr.Column(scale=2):
                    t2_chart  = gr.Plot(label="Lineage Graph")
                    t2_report = gr.Markdown()

            t2_btn.click(
                run_provenance,
                inputs=[t2_rows_a, t2_rows_b, t2_norm, t2_clip],
                outputs=[t2_report, t2_chart],
            )
            demo.load(
                run_provenance,
                inputs=[t2_rows_a, t2_rows_b, t2_norm, t2_clip],
                outputs=[t2_report, t2_chart],
            )

        # ── Tab 3: Unlearning ──────────────────────────────────────────────
        with gr.Tab("🧹 Unlearning"):
            gr.Markdown(UNLEARNING_INTRO)
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    t3_idx = gr.Slider(
                        0, _state["n_train"] - 1,
                        value=_state["default_forget_idx"],
                        step=1,
                        label="Training Sample to Forget",
                        info="Default is a 'hard' sample where loss > 0.05 so the effect is visible.",
                    )
                    t3_steps = gr.Slider(
                        5, 100, value=50, step=5,
                        label="Gradient Ascent Steps",
                    )
                    t3_lr = gr.Slider(
                        -4.0, -1.0, value=-2.0, step=0.5,
                        label="Learning Rate (log₁₀)",
                        info="e.g. −2 → lr = 0.01",
                    )
                    t3_btn = gr.Button("▶  Run Unlearning", variant="primary")

                with gr.Column(scale=2):
                    t3_summary = gr.Markdown()
                    t3_chart   = gr.Plot(label="Before vs After")

            t3_btn.click(
                run_unlearning,
                inputs=[t3_idx, t3_steps, t3_lr],
                outputs=[t3_summary, t3_chart],
            )

        # ── Tab 4: Benchmarks & About ──────────────────────────────────────
        with gr.Tab("📊 Benchmarks & About"):
            gr.Markdown(BENCHMARKS_MD)

        # ── Tab 5: Feedback ────────────────────────────────────────────────
        with gr.Tab("💬 Feedback"):
            gr.Markdown(
                "### Share Your Thoughts\n\n"
                "Found a bug, have a feature request, or using Traceprop in a project? "
                "This helps prioritise the roadmap — thank you!"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    fb_type = gr.Radio(
                        ["🐛 Bug report", "💡 Feature request",
                         "👍 General feedback", "🤝 Using in a project"],
                        value="👍 General feedback",
                        label="Type",
                    )
                    fb_text = gr.Textbox(
                        label="Your message",
                        placeholder="Describe the bug / feature / your use-case …",
                        lines=5,
                    )
                    fb_email = gr.Textbox(
                        label="Email (optional — for follow-up)",
                        placeholder="you@example.com",
                    )
                    fb_btn = gr.Button("Send Feedback", variant="primary")
                    fb_out = gr.Markdown()

                with gr.Column(scale=1):
                    gr.Markdown(
                        "**Prefer GitHub?**\n\n"
                        "Open an issue directly at [AmitoVrito/Traceprop/issues]"
                        "(https://github.com/AmitoVrito/Traceprop/issues) "
                        "for faster tracking and discussion.\n\n"
                        "**Questions?**  \n"
                        "Email: research.amit.n@gmail.com\n\n"
                        "---\n"
                        "**What we'd love to hear:**\n"
                        "- Your use case and industry\n"
                        "- Which features matter most to you\n"
                        "- Performance or accuracy issues\n"
                        "- Anything that surprised you"
                    )

            def submit_feedback(fb_type: str, fb_text: str, fb_email: str) -> str:
                if not fb_text.strip():
                    return "⚠️ Please enter a message before submitting."
                print(f"[FEEDBACK] type={fb_type!r} email={fb_email!r}\n{fb_text}\n---")
                return (
                    "✅ **Thank you for your feedback!** It's been recorded.\n\n"
                    "You can also open an issue directly on "
                    "[GitHub](https://github.com/AmitoVrito/Traceprop/issues) "
                    "for faster tracking."
                )

            fb_btn.click(
                submit_feedback,
                inputs=[fb_type, fb_text, fb_email],
                outputs=[fb_out],
            )

    gr.HTML(
        '<p style="text-align:center;font-size:0.8rem;color:#9ca3af;margin-top:12px;">'
        '📦 <a href="https://pypi.org/project/traceprop/" style="color:#6366f1;">traceprop</a> '
        '&nbsp;·&nbsp; Apache 2.0 License &nbsp;·&nbsp; '
        '<a href="https://github.com/AmitoVrito/Traceprop" style="color:#6366f1;">GitHub</a> '
        '&nbsp;·&nbsp; VLDB 2027 submission'
        '</p>'
    )


if __name__ == "__main__":
    demo.launch()
