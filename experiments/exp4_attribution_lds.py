"""Experiment 4: Attribution Quality — LDS (Linear Datamodeling Score).

Adapted for NumPy-only (no PyTorch). Uses a logistic regression on a synthetic
binary classification task as a proxy for the CIFAR-2/ResNet-9 benchmark.

The LDS methodology is the same: retrain on random subsets, measure correlation
between predicted influence and actual accuracy change.
"""

import numpy as np
from scipy.stats import spearmanr
import traceprop as tp
from traceprop.graph import reset_graph
import json
import time

PROJ_DIM = 4096
N_FEATURES = 50
N_TRAIN = 2000
N_TEST = 200
N_SUBSETS = 50
SUBSET_RATIO = 0.5
LR = 0.1
EPOCHS = 50

np.random.seed(42)

print(f"Attribution LDS Benchmark (NumPy logistic regression)")
print(f"Train: {N_TRAIN}, Test: {N_TEST}, Features: {N_FEATURES}")
print(f"Proj dim: {PROJ_DIM}, Subsets: {N_SUBSETS}\n")


# --- Data ---
def make_data(n, d):
    w_true = np.random.randn(d)
    X = np.random.randn(n, d)
    logits = X @ w_true
    y = (logits > 0).astype(np.float64)
    # Add noise
    flip = np.random.rand(n) < 0.05
    y[flip] = 1 - y[flip]
    return X, y


X_train, y_train = make_data(N_TRAIN, N_FEATURES)
X_test, y_test = make_data(N_TEST, N_FEATURES)


# --- Model ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logreg(X, y, lr=LR, epochs=EPOCHS):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        pred = sigmoid(X @ w + b)
        err = pred - y
        w -= lr / len(X) * (X.T @ err)
        b -= lr / len(X) * err.sum()
    return w, b


def predict(X, w, b):
    return (sigmoid(X @ w + b) > 0.5).astype(np.float64)


def accuracy(X, y, w, b):
    return (predict(X, w, b) == y).mean()


def compute_gradient(X, y, w, b):
    pred = sigmoid(X @ w + b)
    err = pred - y
    grad_w = (X.T @ err) / len(X)
    grad_b = err.mean()
    return np.append(grad_w, grad_b)


# --- Step 1: Train full model, log per-sample gradients ---
print("Step 1: Training full model and logging gradients...")
w, b = train_logreg(X_train, y_train)
full_acc = accuracy(X_test, y_test, w, b)
print(f"Full model test accuracy: {full_acc:.4f}")

reset_graph()
ctx = tp.training_context(source_id="synth_train", proj_dim=PROJ_DIM)

for i in range(N_TRAIN):
    grad = compute_gradient(X_train[i:i + 1], y_train[i:i + 1], w, b)
    ctx.log_gradient(gradient=grad, sample_index=i)

print(f"Logged {len(ctx.gradient_store)} gradient entries\n")

# --- Step 2: Compute Traceprop influence scores ---
print("Step 2: Computing Traceprop influence scores...")
engine = tp.attribution_engine(ctx.gradient_store)

tp_influence_matrix = []
for i in range(N_TEST):
    test_grad = compute_gradient(X_test[i:i + 1], y_test[i:i + 1], w, b)
    result = engine.attribute(test_grad, top_k=N_TRAIN)
    scores = np.zeros(N_TRAIN)
    for entry in result.top(N_TRAIN):
        idx = entry["sample_index"]
        if 0 <= idx < N_TRAIN:
            scores[idx] = entry["influence_score"]
    tp_influence_matrix.append(scores)

tp_influence_matrix = np.array(tp_influence_matrix)
print(f"Influence matrix shape: {tp_influence_matrix.shape}\n")

# --- Step 3: Ground-truth via retraining ---
print(f"Step 3: Retraining on {N_SUBSETS} random subsets for ground truth...")
t0 = time.perf_counter()

subset_masks = []
subset_accuracies = np.zeros((N_SUBSETS, N_TEST))

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < SUBSET_RATIO
    subset_masks.append(mask)
    X_sub, y_sub = X_train[mask], y_train[mask]
    w_s, b_s = train_logreg(X_sub, y_sub)
    # Per-test-point accuracy (binary correctness)
    preds = predict(X_test, w_s, b_s)
    subset_accuracies[s] = (preds == y_test).astype(np.float64)
    if s % 10 == 0:
        print(f"  Subset {s}/{N_SUBSETS} done")

subset_masks = np.array(subset_masks)  # (N_SUBSETS, N_TRAIN)
print(f"Retraining took {time.perf_counter() - t0:.1f}s\n")

# --- Step 4: Compute LDS ---
print("Step 4: Computing LDS scores...")

lds_scores_tp = []
lds_scores_random = []
random_scores = np.random.rand(*tp_influence_matrix.shape)

for test_i in range(N_TEST):
    tp_pred = tp_influence_matrix[test_i]
    rand_pred = random_scores[test_i]

    # Ground truth per training sample: correlation of inclusion with test correctness
    gt = np.array([
        spearmanr(subset_masks[:, j], subset_accuracies[:, test_i]).statistic
        for j in range(N_TRAIN)
    ])
    gt = np.nan_to_num(gt, 0.0)

    tp_lds = spearmanr(tp_pred, gt).statistic
    rand_lds = spearmanr(rand_pred, gt).statistic

    lds_scores_tp.append(tp_lds if not np.isnan(tp_lds) else 0.0)
    lds_scores_random.append(rand_lds if not np.isnan(rand_lds) else 0.0)

mean_lds_tp = np.mean(lds_scores_tp)
std_lds_tp = np.std(lds_scores_tp)
mean_lds_random = np.mean(lds_scores_random)

print(f"\n{'=' * 50}")
print(f"Traceprop LDS:     {mean_lds_tp:.4f} ± {std_lds_tp:.4f}")
print(f"Random baseline:   {mean_lds_random:.4f}")
print(f"{'=' * 50}")

results = {
    "traceprop_lds_mean": round(float(mean_lds_tp), 4),
    "traceprop_lds_std": round(float(std_lds_tp), 4),
    "random_lds_mean": round(float(mean_lds_random), 4),
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "n_subsets": N_SUBSETS,
    "n_features": N_FEATURES,
    "proj_dim": PROJ_DIM,
    "full_model_accuracy": round(float(full_acc), 4),
    "model_type": "logistic_regression",
}

with open("results/exp4_lds.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp4_lds.json")
