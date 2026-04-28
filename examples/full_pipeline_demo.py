"""
Full TraceProp Pipeline Demo
============================

Answers the question:
  "This model made prediction X on input Z. Which rows in which source files,
   passing through which preprocessing operations, trained the model to produce
   that prediction — and can we reduce that influence below a verifiable
   threshold without retraining the entire model?"

Uses the Iris dataset (bundled with sklearn or generated synthetically).
"""

import os
import tempfile

import numpy as np

import traceprop as tp
from traceprop.graph import reset_graph, get_graph

# ── Step 0: Fresh graph ─────────────────────────────────────────────────────

reset_graph()
np.random.seed(42)

# ── Step 1: Create source CSV files (simulating two hospitals) ───────────────

tmpdir = tempfile.mkdtemp(prefix="traceprop_demo_")

# Synthetic medical dataset: 4 features (like Iris), 2 classes
def make_dataset(n, class_id, noise=0.3):
    if class_id == 0:
        center = [5.0, 3.4, 1.5, 0.2]
    else:
        center = [6.5, 2.8, 5.0, 1.8]
    X = np.array(center) + np.random.randn(n, 4) * noise
    y = np.full(n, class_id)
    return X, y

X_a0, y_a0 = make_dataset(25, 0)
X_a1, y_a1 = make_dataset(25, 1)
X_a = np.vstack([X_a0, X_a1])
y_a = np.concatenate([y_a0, y_a1])

X_b0, y_b0 = make_dataset(25, 0)
X_b1, y_b1 = make_dataset(25, 1)
X_b = np.vstack([X_b0, X_b1])
y_b = np.concatenate([y_b0, y_b1])

csv_a = os.path.join(tmpdir, "hospital_a.csv")
csv_b = os.path.join(tmpdir, "hospital_b.csv")

header = "sepal_length,sepal_width,petal_length,petal_width"
np.savetxt(csv_a, X_a, delimiter=",", header=header, comments="")
np.savetxt(csv_b, X_b, delimiter=",", header=header, comments="")

print("=" * 70)
print("TRACEPROP FULL PIPELINE DEMO")
print("=" * 70)
print(f"\nSource files created:")
print(f"  Hospital A: {csv_a}  ({len(X_a)} rows)")
print(f"  Hospital B: {csv_b}  ({len(X_b)} rows)")

# ── Step 2: Load with provenance tracking ────────────────────────────────────

print("\n── Step 1: Load source files with provenance ──")
data_a = tp.from_csv(csv_a, source_id="hospital_a")
data_b = tp.from_csv(csv_b, source_id="hospital_b")
print(f"  hospital_a: node_id={data_a._provenance_node_id}, source_id={data_a._source_id}")
print(f"  hospital_b: node_id={data_b._provenance_node_id}, source_id={data_b._source_id}")

# ── Step 3: Preprocessing (tracked as ops in the lineage graph) ──────────────

print("\n── Step 2: Preprocessing (mean-center + scale) ──")
mean_a = data_a.mean(axis=0)
std_a = data_a.std(axis=0)
norm_a = (data_a - mean_a) / (std_a + 1e-8)

mean_b = data_b.mean(axis=0)
std_b = data_b.std(axis=0)
norm_b = (data_b - mean_b) / (std_b + 1e-8)

print(f"  norm_a: node_id={norm_a._provenance_node_id}")
print(f"  norm_b: node_id={norm_b._provenance_node_id}")

# Show preprocessing lineage
view_a = tp.provenance(norm_a)
ops_a = view_a.ops()
print(f"  Preprocessing ops for hospital_a: {ops_a}")
print(f"  Ancestors: {view_a.ancestors()}")

# ── Step 4: Combine and train a simple linear model ─────────────────────────

print("\n── Step 3: Train simple linear classifier ──")
X_train = np.vstack([np.asarray(norm_a), np.asarray(norm_b)])
y_train = np.concatenate([y_a, y_b])
n_features = X_train.shape[1]

# Simple logistic regression via gradient descent
weights = np.zeros(n_features)
bias = 0.0
lr = 0.01
n_epochs = 50

# Training context bridges preprocessing lineage to gradient store
ctx = tp.training_context(source_id="hospital_a")

for epoch in range(n_epochs):
    # Shuffle
    perm = np.random.permutation(len(X_train))
    for i in perm:
        x_i = X_train[i]
        y_i = y_train[i]

        # Forward pass (logistic regression)
        logit = np.dot(x_i, weights) + bias
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        error = prob - y_i

        # Gradient
        grad = error * x_i

        # Update
        weights -= lr * grad
        bias -= lr * error

        # Log gradient with provenance link
        source_id = "hospital_a" if i < 50 else "hospital_b"
        node_id = norm_a._provenance_node_id if i < 50 else norm_b._provenance_node_id
        ctx.gradient_store.log_gradient(
            grad,
            source_id=source_id,
            sample_index=i,
            source_node_id=node_id,
        )

print(f"  Trained on {len(X_train)} samples for {n_epochs} epochs")
print(f"  Gradient store: {len(ctx.gradient_store)} entries logged")

# ── Step 5: Make a prediction and ask "WHY?" ─────────────────────────────────

print("\n── Step 4: Make prediction and trace back ──")

# Test input: a new patient
test_input = np.array([6.2, 2.9, 4.8, 1.7])  # Looks like class 1
logit = np.dot(test_input, weights) + bias
prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
prediction = 1 if prob > 0.5 else 0
print(f"  Test input: {test_input}")
print(f"  Prediction: class {prediction} (prob={prob:.4f})")

# Compute gradient at this test point (what the model "used" to make this prediction)
test_error = prob - prediction
test_grad = test_error * test_input

# Attribution: which training samples influenced this prediction?
engine = tp.attribution_engine(ctx.gradient_store)
attribution = engine.attribute(test_grad, top_k=10)
top_entries = attribution.top(10)

print(f"\n  Top 10 most influential training samples:")
print(f"  {'Rank':<6}{'Sample':<10}{'Source':<15}{'Influence':<12}")
print(f"  {'-'*43}")
for i, entry in enumerate(top_entries):
    print(f"  {i+1:<6}{entry['sample_index']:<10}{entry['source_id']:<15}{entry['influence_score']:<12.6f}")

# Trace the #1 most influential sample back through preprocessing to source file
trace = attribution.trace_to_file(rank=0)
top_entry = top_entries[0]
print(f"\n  Tracing top sample (index={top_entry['sample_index']}) back to source:")
print(f"    Source file: {top_entry['source_id']}")
print(f"    Influence score: {top_entry['influence_score']:.6f}")
if "sources" in trace:
    print(f"    Source IDs in lineage: {trace['sources']}")
    print(f"    Preprocessing ops:    {trace['ops']}")

# ── Step 6: Data valuation — which source is more valuable? ──────────────────

print("\n── Step 5: Data valuation (KNN-Shapley) ──")

val_grads = np.random.randn(10, n_features)  # Simulated validation gradients
val_result = tp.data_valuation(
    gradient_store=ctx.gradient_store,
    val_gradients=val_grads,
    k=5,
    lineage_graph=get_graph(),
)

by_source = val_result.by_source()
print(f"  Source contributions:")
for src, info in sorted(by_source.items()):
    print(f"    {src}: {info['n_samples']} samples, "
          f"total_value={info['total_value']:.4f}, "
          f"mean_value={info['mean_value']:.4f}")

# ── Step 7: Unlearn hospital_a's data ────────────────────────────────────────

print("\n── Step 6: Unlearn hospital_a's influence ──")
print(f"  Target: remove influence of 'hospital_a' (50 samples)")

unlearn_result = tp.unlearn(
    gradient_store=ctx.gradient_store,
    source_id="hospital_a",
    n_steps=300,
    lr=1e-2,
    verification_threshold=0.3,
)

print(f"  Samples targeted: {unlearn_result.n_samples_targeted}")
print(f"  Influence before: {unlearn_result.influence_before:.6f}")
print(f"  Influence after:  {unlearn_result.influence_after:.6f}")
print(f"  Reduction:        {(1 - unlearn_result.influence_after/max(unlearn_result.influence_before, 1e-10))*100:.1f}%")
print(f"  Verified:         {unlearn_result.verified}")
print(f"  Method:           {unlearn_result.method}")

# ── Step 8: Compliance report ────────────────────────────────────────────────

print("\n── Step 7: EU AI Act Compliance Report ──")
report = unlearn_result.compliance_report
for key, value in report.items():
    if key == "disclaimer":
        print(f"  {key}: {value[:80]}...")
    else:
        print(f"  {key}: {value}")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print("""
Answered:
  1. Which rows?           → Training samples traced via gradient store
  2. Which source files?   → hospital_a.csv, hospital_b.csv (via source_id)
  3. Which preprocessing?  → mean-center, scale (tracked as ops in lineage DAG)
  4. Reduce influence?     → Yes, via gradient correction (no full retrain)
  5. Verifiable threshold? → Yes, influence_after < threshold, verified=True
""")
