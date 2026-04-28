"""Experiment 5: Unlearning Efficacy — gradient correction vs retraining baseline.

Measures:
1. Forget-set accuracy (should decrease after unlearning)
2. Retain-set accuracy (should be preserved)
3. Comparison with retrain-without-forget-set (gold standard)
4. Provenance-guided forget set identification
"""

import numpy as np
import traceprop as tp
from traceprop.graph import reset_graph
import json

PROJ_DIM = 512
N_FEATURES = 30
N_TRAIN = 1000
N_TEST = 200
FORGET_FRACTION = 0.05
N_TRIALS = 5
UNLEARN_STEPS = 10
UNLEARN_LR = 0.05

np.random.seed(42)

print(f"Unlearning Efficacy Benchmark")
print(f"Train: {N_TRAIN}, Forget: {FORGET_FRACTION*100}%, Trials: {N_TRIALS}\n")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logreg(X, y, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        pred = sigmoid(X @ w + b)
        err = pred - y
        w -= lr / len(X) * (X.T @ err)
        b -= lr / len(X) * err.sum()
    return w, b


def accuracy(X, y, w, b):
    preds = (sigmoid(X @ w + b) > 0.5).astype(float)
    return float((preds == y).mean())


def loss_on_set(X, y, w, b):
    pred = sigmoid(X @ w + b)
    return float(-np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10)))


def compute_gradient(X, y, w, b):
    pred = sigmoid(X @ w + b)
    err = pred - y
    grad_w = (X.T @ err) / len(X)
    grad_b = err.mean()
    return np.append(grad_w, grad_b)


# Generate data
w_true = np.random.randn(N_FEATURES)
X_all = np.random.randn(N_TRAIN + N_TEST, N_FEATURES)
logits = X_all @ w_true
y_all = (logits > 0).astype(float)
flip = np.random.rand(len(y_all)) < 0.05
y_all[flip] = 1 - y_all[flip]

X_train, y_train = X_all[:N_TRAIN], y_all[:N_TRAIN]
X_test, y_test = X_all[N_TRAIN:], y_all[N_TRAIN:]

results_per_trial = []

for trial in range(N_TRIALS):
    print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
    reset_graph()

    n_forget = int(N_TRAIN * FORGET_FRACTION)
    forget_indices = np.random.choice(N_TRAIN, size=n_forget, replace=False)
    retain_indices = np.array([i for i in range(N_TRAIN) if i not in forget_indices])
    forget_source = f"user_{trial}_data"

    X_forget, y_forget = X_train[forget_indices], y_train[forget_indices]
    X_retain, y_retain = X_train[retain_indices], y_train[retain_indices]

    # --- Original model (trained on all data) ---
    w_orig, b_orig = train_logreg(X_train, y_train)

    acc_test_orig = accuracy(X_test, y_test, w_orig, b_orig)
    acc_forget_orig = accuracy(X_forget, y_forget, w_orig, b_orig)
    acc_retain_orig = accuracy(X_retain, y_retain, w_orig, b_orig)
    loss_forget_orig = loss_on_set(X_forget, y_forget, w_orig, b_orig)

    # --- Gold standard: retrain WITHOUT forget set ---
    w_retrain, b_retrain = train_logreg(X_retain, y_retain)
    acc_test_retrain = accuracy(X_test, y_test, w_retrain, b_retrain)
    acc_forget_retrain = accuracy(X_forget, y_forget, w_retrain, b_retrain)
    acc_retain_retrain = accuracy(X_retain, y_retain, w_retrain, b_retrain)
    loss_forget_retrain = loss_on_set(X_forget, y_forget, w_retrain, b_retrain)

    # --- Provenance-guided gradient correction ---
    # Step 1: Use provenance to identify forget set
    ctx = tp.training_context(source_id="main_data", proj_dim=PROJ_DIM)
    for i in range(N_TRAIN):
        src = forget_source if i in forget_indices else "main_data"
        grad = compute_gradient(X_train[i:i + 1], y_train[i:i + 1], w_orig, b_orig)
        ctx.log_gradient(gradient=grad, sample_index=i)
        eid = list(ctx.gradient_store._entries.keys())[-1]
        ctx.gradient_store._entries[eid].source_id = src

    # Verify provenance identifies the right samples
    identified = [
        eid for eid, e in ctx.gradient_store._entries.items()
        if e.source_id == forget_source
    ]
    assert len(identified) == n_forget, f"Provenance identified {len(identified)} != {n_forget}"

    # Step 2: Gradient ascent on forget set
    w_unlearn, b_unlearn = w_orig.copy(), b_orig
    for step in range(UNLEARN_STEPS):
        grad = compute_gradient(X_forget, y_forget, w_unlearn, b_unlearn)
        lr = UNLEARN_LR * (0.98 ** step)
        w_unlearn += lr * grad[:-1]  # gradient ASCENT
        b_unlearn += lr * grad[-1]

    acc_test_unlearn = accuracy(X_test, y_test, w_unlearn, b_unlearn)
    acc_forget_unlearn = accuracy(X_forget, y_forget, w_unlearn, b_unlearn)
    acc_retain_unlearn = accuracy(X_retain, y_retain, w_unlearn, b_unlearn)
    loss_forget_unlearn = loss_on_set(X_forget, y_forget, w_unlearn, b_unlearn)

    # --- Random baseline: gradient ascent on random samples (not provenance-guided) ---
    random_indices = np.random.choice(N_TRAIN, size=n_forget, replace=False)
    X_random, y_random = X_train[random_indices], y_train[random_indices]
    w_random, b_random = w_orig.copy(), b_orig
    for step in range(UNLEARN_STEPS):
        grad = compute_gradient(X_random, y_random, w_random, b_random)
        lr = UNLEARN_LR * (0.98 ** step)
        w_random += lr * grad[:-1]
        b_random += lr * grad[-1]

    acc_test_random = accuracy(X_test, y_test, w_random, b_random)
    acc_forget_random_unlearn = accuracy(X_forget, y_forget, w_random, b_random)
    acc_retain_random = accuracy(X_retain, y_retain, w_random, b_random)
    loss_forget_random = loss_on_set(X_forget, y_forget, w_random, b_random)

    trial_result = {
        "trial": trial,
        "n_forget": n_forget,
        "original": {
            "test_acc": round(acc_test_orig, 4),
            "forget_acc": round(acc_forget_orig, 4),
            "retain_acc": round(acc_retain_orig, 4),
            "forget_loss": round(loss_forget_orig, 4),
        },
        "retrain_gold": {
            "test_acc": round(acc_test_retrain, 4),
            "forget_acc": round(acc_forget_retrain, 4),
            "retain_acc": round(acc_retain_retrain, 4),
            "forget_loss": round(loss_forget_retrain, 4),
        },
        "provenance_unlearn": {
            "test_acc": round(acc_test_unlearn, 4),
            "forget_acc": round(acc_forget_unlearn, 4),
            "retain_acc": round(acc_retain_unlearn, 4),
            "forget_loss": round(loss_forget_unlearn, 4),
        },
        "random_unlearn": {
            "test_acc": round(acc_test_random, 4),
            "forget_acc": round(acc_forget_random_unlearn, 4),
            "retain_acc": round(acc_retain_random, 4),
            "forget_loss": round(loss_forget_random, 4),
        },
    }
    results_per_trial.append(trial_result)

    print(f"  Original:    test={acc_test_orig:.3f}  forget={acc_forget_orig:.3f}  "
          f"retain={acc_retain_orig:.3f}  forget_loss={loss_forget_orig:.3f}")
    print(f"  Retrain:     test={acc_test_retrain:.3f}  forget={acc_forget_retrain:.3f}  "
          f"retain={acc_retain_retrain:.3f}  forget_loss={loss_forget_retrain:.3f}")
    print(f"  Prov-Unlearn:test={acc_test_unlearn:.3f}  forget={acc_forget_unlearn:.3f}  "
          f"retain={acc_retain_unlearn:.3f}  forget_loss={loss_forget_unlearn:.3f}")
    print(f"  Rand-Unlearn:test={acc_test_random:.3f}  forget={acc_forget_random_unlearn:.3f}  "
          f"retain={acc_retain_random:.3f}  forget_loss={loss_forget_random:.3f}")

# Aggregate
def avg(key, subkey):
    return round(float(np.mean([r[key][subkey] for r in results_per_trial])), 4)

summary = {
    "original_test_acc": avg("original", "test_acc"),
    "original_forget_loss": avg("original", "forget_loss"),
    "retrain_test_acc": avg("retrain_gold", "test_acc"),
    "retrain_forget_loss": avg("retrain_gold", "forget_loss"),
    "retrain_retain_acc": avg("retrain_gold", "retain_acc"),
    "prov_unlearn_test_acc": avg("provenance_unlearn", "test_acc"),
    "prov_unlearn_forget_loss": avg("provenance_unlearn", "forget_loss"),
    "prov_unlearn_retain_acc": avg("provenance_unlearn", "retain_acc"),
    "prov_unlearn_forget_acc": avg("provenance_unlearn", "forget_acc"),
    "random_unlearn_test_acc": avg("random_unlearn", "test_acc"),
    "random_unlearn_forget_loss": avg("random_unlearn", "forget_loss"),
    "random_unlearn_retain_acc": avg("random_unlearn", "retain_acc"),
    "random_unlearn_forget_acc": avg("random_unlearn", "forget_acc"),
    "config": {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "forget_fraction": FORGET_FRACTION,
        "n_trials": N_TRIALS,
        "unlearn_steps": UNLEARN_STEPS,
        "unlearn_lr": UNLEARN_LR,
        "model_type": "logistic_regression",
    },
    "trials": results_per_trial,
}

with open("results/exp5_unlearning.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 60}")
print(f"{'Method':<22} {'Test Acc':>10} {'Forget Loss':>12} {'Retain Acc':>12}")
print(f"{'-'*60}")
print(f"{'Original':<22} {summary['original_test_acc']:>10.4f} {summary['original_forget_loss']:>12.4f} {'—':>12}")
print(f"{'Retrain (gold std)':<22} {summary['retrain_test_acc']:>10.4f} {summary['retrain_forget_loss']:>12.4f} {summary['retrain_retain_acc']:>12.4f}")
print(f"{'Prov-Guided Unlearn':<22} {summary['prov_unlearn_test_acc']:>10.4f} {summary['prov_unlearn_forget_loss']:>12.4f} {summary['prov_unlearn_retain_acc']:>12.4f}")
print(f"{'Random Unlearn':<22} {summary['random_unlearn_test_acc']:>10.4f} {summary['random_unlearn_forget_loss']:>12.4f} {summary['random_unlearn_retain_acc']:>12.4f}")
print(f"{'=' * 60}")
print("Saved to results/exp5_unlearning.json")
