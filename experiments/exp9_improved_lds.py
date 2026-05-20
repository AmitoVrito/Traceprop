"""Experiment 9: Improved Attribution — Multi-Checkpoint + TRAK Estimator + Frozen BN.

Tests all 4 improvements vs baseline Traceprop-LL on Adult Income tabular data.
CPU-only. Expected to push LDS well above the baseline 0.622.

Variants:
  A) TP-LL baseline          (dot product, 1 checkpoint)
  B) TP-LL + TRAK estimator  (dot → (ΦᵀΦ)⁻¹ estimator, 1 checkpoint)
  C) TP-LL + 5 checkpoints   (averaged dot product, 5 checkpoints)
  D) TP-LL + 5 ckpts + TRAK  (averaged TRAK estimator, 5 checkpoints) ← target
  E) Random baseline
"""

import json
import os
import time

# Pin BLAS to all available cores before importing numpy
_N_CORES = str(os.cpu_count())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_CORES)

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import (
    AttributionEngine,
    MultiCheckpointAttributionEngine,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN      = 6000
N_TEST       = 500
N_SUBSETS    = 500
SUBSET_RATIO = 0.5
PROJ_DIM     = 4096
N_CHECKPOINTS = 5
LR           = 1.0
MAX_ITER     = 200         # total iterations (converged)
CKPT_ITERS   = [40, 80, 120, 160, 200]  # checkpoint at these iteration counts
LAMBDA       = 1e-3
np.random.seed(42)

print("=" * 65)
print("Exp 9: Improved LDS — Multi-Checkpoint + TRAK Estimator")
print(f"  N_train={N_TRAIN}, N_test={N_TEST}, N_subsets={N_SUBSETS}")
print(f"  proj_dim={PROJ_DIM}, n_checkpoints={N_CHECKPOINTS}, lambda={LAMBDA}")
print("=" * 65)

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading Adult Income dataset...")
adult = fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
X_raw, y_raw = adult.data, (adult.target == ">50K").astype(float)

# Keep complete rows
mask = ~np.isnan(X_raw).any(axis=1)
X_raw, y_raw = X_raw[mask], y_raw[mask]

rng = np.random.RandomState(42)
idx = rng.permutation(len(X_raw))
tr, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST]
X_tr, y_tr = X_raw[tr], y_raw[tr]
X_te, y_te = X_raw[te], y_raw[te]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def per_sample_ll_grad(x_i, y_i, coef, intercept):
    """Per-sample last-layer gradient = (σ(w·x+b) - y) * x  (logistic reg)."""
    logit = float(x_i @ coef + intercept)
    err = sigmoid(logit) - y_i
    return (err * x_i).astype(np.float32)

def train_lr(X, y, max_iter=30, seed=0):
    clf = LogisticRegression(
        C=10.0, solver="lbfgs", max_iter=max_iter,
        random_state=seed, warm_start=False,
    )
    clf.fit(X, y)
    return clf

def lr_margin(clf, X, y):
    """Signed margin: decision_function * (2y-1), positive when correctly confident."""
    return (clf.decision_function(X) * (2 * y - 1)).astype(np.float32)

def log_gradients(X, y, coef, intercept, proj_dim, seed):
    """Log per-sample LL gradients into a new GradientStore."""
    store = GradientStore(proj_dim=proj_dim, seed=seed)
    for i in range(len(X)):
        g = per_sample_ll_grad(X[i], y[i], coef, intercept)
        store.log_gradient(g, sample_index=i, source_id="adult")
    return store

# ── Step 1: Train at multiple checkpoints ────────────────────────────────────
print("\nStep 1: Training model at 5 checkpoints...")
checkpoint_stores = []
for ckpt_idx, n_iter in enumerate(CKPT_ITERS):
    clf_k = train_lr(X_tr, y_tr, max_iter=n_iter, seed=0)
    coef_k = clf_k.coef_[0]
    intercept_k = clf_k.intercept_[0]
    seed_k = 42 + ckpt_idx   # different JL seed per checkpoint
    store_k = log_gradients(X_tr, y_tr, coef_k, intercept_k, PROJ_DIM, seed_k)
    checkpoint_stores.append(store_k)
    print(f"  Checkpoint {ckpt_idx+1}/{N_CHECKPOINTS} (iter={n_iter}): "
          f"{len(store_k)} entries, seed={seed_k}")

# Final checkpoint = baseline model
clf_final = train_lr(X_tr, y_tr, max_iter=MAX_ITER, seed=0)
baseline_store = checkpoint_stores[-1]

# ── Step 2: Ground truth — 500 subset retrains ───────────────────────────────
print(f"\nStep 2: Retraining on {N_SUBSETS} subsets...")
t0 = time.perf_counter()
subset_masks   = np.zeros((N_SUBSETS, N_TRAIN), dtype=bool)
subset_margins = np.zeros((N_SUBSETS, N_TEST),  dtype=np.float32)

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < SUBSET_RATIO
    subset_masks[s] = mask
    clf_s = train_lr(X_tr[mask], y_tr[mask], max_iter=MAX_ITER, seed=s + 1)
    subset_margins[s] = lr_margin(clf_s, X_te, y_te)
    if s % 100 == 0:
        print(f"  {s}/{N_SUBSETS}  ({time.perf_counter()-t0:.0f}s)")

t_retrain = time.perf_counter() - t0
print(f"  Done in {t_retrain:.1f}s")

# ── Step 3: Build attribution engines ─────────────────────────────────────────
print("\nStep 3: Building attribution engines...")

engine_A = AttributionEngine(baseline_store, estimator="dot")
engine_B = AttributionEngine(baseline_store, estimator="trak", lambda_factor=LAMBDA)
engine_C = MultiCheckpointAttributionEngine(checkpoint_stores, use_trak=False)
engine_D = MultiCheckpointAttributionEngine(checkpoint_stores, use_trak=True, lambda_factor=LAMBDA)
print("  All engines ready.")

# ── Step 4: Compute attribution matrices (batched — all CPU cores via BLAS) ───
print(f"\nStep 4: Computing attribution matrices (batch mode, {os.cpu_count()} cores)...")

# All test gradients at once: (N_TEST, N_FEATURES)
coef_f, intercept_f = clf_final.coef_[0], clf_final.intercept_[0]
err_te = sigmoid(X_te @ coef_f + intercept_f) - y_te   # (N_TEST,)
G_te = (err_te[:, None] * X_te).astype(np.float32)      # (N_TEST, N_FEATURES)

def build_inf_batch(engine, G_te, name):
    t = time.perf_counter()
    print(f"  [{name}] batching {len(G_te)} test samples...")
    inf = engine.attribute_scores_batch(G_te)            # one call, all cores
    print(f"  [{name}] done in {time.perf_counter()-t:.1f}s")
    return inf, time.perf_counter() - t

inf_A, t_A = build_inf_batch(engine_A, G_te, "A:baseline")
inf_B, t_B = build_inf_batch(engine_B, G_te, "B:trak")
inf_C, t_C = build_inf_batch(engine_C, G_te, "C:5ckpt")
inf_D, t_D = build_inf_batch(engine_D, G_te, "D:5ckpt+trak")

rand_inf = np.random.randn(N_TEST, N_TRAIN).astype(np.float32)
rand_inf /= np.abs(rand_inf).max(axis=1, keepdims=True) + 1e-8

# ── Step 5: LDS ───────────────────────────────────────────────────────────────
print("\nStep 5: Computing LDS...")

def compute_lds(inf_matrix):
    predicted = inf_matrix @ subset_masks.T  # (N_TEST, N_SUBSETS)
    lds = []
    for i in range(len(predicted)):
        r = spearmanr(predicted[i], subset_margins[:, i]).statistic
        lds.append(0.0 if np.isnan(r) else r)
    return np.array(lds)

lds_A    = compute_lds(inf_A)
lds_B    = compute_lds(inf_B)
lds_C    = compute_lds(inf_C)
lds_D    = compute_lds(inf_D)
lds_rand = compute_lds(rand_inf)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"{'Variant':<30} {'LDS mean':>10} {'± std':>10} {'Time':>8}")
print(f"{'-' * 65}")
print(f"{'A: TP-LL baseline':<30} {lds_A.mean():>10.4f} {lds_A.std():>10.4f} {t_A:>7.1f}s")
print(f"{'B: TP-LL + TRAK estimator':<30} {lds_B.mean():>10.4f} {lds_B.std():>10.4f} {t_B:>7.1f}s")
print(f"{'C: TP-LL + 5 checkpoints':<30} {lds_C.mean():>10.4f} {lds_C.std():>10.4f} {t_C:>7.1f}s")
print(f"{'D: TP-LL + 5 ckpts + TRAK':<30} {lds_D.mean():>10.4f} {lds_D.std():>10.4f} {t_D:>7.1f}s")
print(f"{'E: Random baseline':<30} {lds_rand.mean():>10.4f} {lds_rand.std():>10.4f} {'---':>8}")
print(f"{'=' * 65}")

best = max(lds_A.mean(), lds_B.mean(), lds_C.mean(), lds_D.mean())
print(f"\nBest variant LDS: {best:.4f}  (baseline was 0.6222)")
print(f"Improvement over baseline: {(best - lds_A.mean()) / lds_A.mean() * 100:+.1f}%")

results = {
    "experiment": "exp9_improved_lds",
    "dataset": "adult_income",
    "n_train": N_TRAIN, "n_test": N_TEST, "n_subsets": N_SUBSETS,
    "proj_dim": PROJ_DIM, "n_checkpoints": N_CHECKPOINTS,
    "lambda_factor": LAMBDA,
    "A_baseline":       {"lds_mean": round(float(lds_A.mean()), 4),
                         "lds_std":  round(float(lds_A.std()), 4), "time_s": round(t_A, 2)},
    "B_trak":           {"lds_mean": round(float(lds_B.mean()), 4),
                         "lds_std":  round(float(lds_B.std()), 4), "time_s": round(t_B, 2)},
    "C_5ckpt":          {"lds_mean": round(float(lds_C.mean()), 4),
                         "lds_std":  round(float(lds_C.std()), 4), "time_s": round(t_C, 2)},
    "D_5ckpt_trak":     {"lds_mean": round(float(lds_D.mean()), 4),
                         "lds_std":  round(float(lds_D.std()), 4), "time_s": round(t_D, 2)},
    "random":           {"lds_mean": round(float(lds_rand.mean()), 4),
                         "lds_std":  round(float(lds_rand.std()), 4)},
    "retrain_time_s": round(t_retrain, 1),
}

with open("results/exp9_improved_lds.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp9_improved_lds.json")
