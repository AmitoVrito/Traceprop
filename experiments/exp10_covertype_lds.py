"""Experiment 10: LDS on large tabular dataset — Covertype 50K.

Demonstrates Traceprop-LL + TRAK estimator scales beyond UCI Adult Income.
Covertype: 581K samples, 54 features, 7 classes → binary (class 2 vs rest).
We subsample 50K train + 500 test for feasibility with 500 subset retrains.

Uses batched matrix multiply (all CPU cores) from exp9.
"""

import json
import os
import time

_N_CORES = str(os.cpu_count())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_CORES)

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.attribution_engine import (
    AttributionEngine, MultiCheckpointAttributionEngine
)
from traceprop.attribution.gradient_store import GradientStore

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN    = 50_000
N_TEST     = 500
N_SUBSETS  = 500
PROJ_DIM   = 4096
LAMBDA     = 1e-3
C_LR       = 10.0
MAX_ITER   = 200
SEED       = 42
np.random.seed(SEED)

print("=" * 65)
print("Exp 10: LDS — Covertype 50K (large tabular benchmark)")
print(f"  N_train={N_TRAIN:,}, N_test={N_TEST}, N_subsets={N_SUBSETS}")
print(f"  proj_dim={PROJ_DIM}, C={C_LR}, max_iter={MAX_ITER}, {os.cpu_count()} cores")
print("=" * 65)

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading Covertype dataset...")
t0 = time.perf_counter()
data = fetch_covtype()
X_all = data.data.astype(np.float32)
y_all = (data.target == 2).astype(np.float32)   # binary: class 2 vs rest
print(f"  Full dataset: {X_all.shape}, class balance: {y_all.mean():.3f}")

# Stratified subsample
idx0 = np.where(y_all == 0)[0]
idx1 = np.where(y_all == 1)[0]
np.random.shuffle(idx0); np.random.shuffle(idx1)
n1 = N_TRAIN // 2; n0 = N_TRAIN - n1
tr_idx = np.concatenate([idx0[:n0], idx1[:n1]])
np.random.shuffle(tr_idx)
te_idx = np.concatenate([idx0[n0:n0+N_TEST//2], idx1[n1:n1+N_TEST//2]])

X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}  [{time.perf_counter()-t0:.1f}s]")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_lr(X, y, max_iter=MAX_ITER, seed=SEED):
    clf = LogisticRegression(C=C_LR, solver="lbfgs", max_iter=max_iter,
                             random_state=seed, warm_start=False)
    clf.fit(X, y)
    return clf

def lr_margin(clf, X, y):
    return (clf.decision_function(X) * (2 * y - 1)).astype(np.float32)

def build_store(X, y, clf, proj_dim, seed):
    store = GradientStore(proj_dim=proj_dim, seed=seed)
    coef, intercept = clf.coef_[0], clf.intercept_[0]
    err = sigmoid(X @ coef + intercept) - y
    G = (err[:, None] * X).astype(np.float32)
    for i in range(len(X)):
        store.log_gradient(G[i], sample_index=i, source_id="covertype")
    return store

def compute_lds(inf_matrix, subset_masks, subset_margins):
    predicted = inf_matrix @ subset_masks.T
    scores = [
        spearmanr(predicted[i], subset_margins[:, i]).statistic
        for i in range(len(predicted))
    ]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)), float(np.std(scores))

# ── Step 1: Train full model + build store ────────────────────────────────────
print("\nStep 1: Training full model...")
t0 = time.perf_counter()
clf = train_lr(X_tr, y_tr)
print(f"  Accuracy: {clf.score(X_te, y_te):.4f}  [{time.perf_counter()-t0:.1f}s]")

print("  Building gradient store...")
t0 = time.perf_counter()
store = build_store(X_tr, y_tr, clf, PROJ_DIM, seed=SEED)
print(f"  Store: {len(store)} entries  [{time.perf_counter()-t0:.1f}s]")

# ── Step 2: Retrain on 500 subsets ───────────────────────────────────────────
print(f"\nStep 2: Retraining on {N_SUBSETS} subsets...")
t0 = time.perf_counter()
subset_masks   = np.zeros((N_SUBSETS, N_TRAIN), dtype=np.float32)
subset_margins = np.zeros((N_SUBSETS, N_TEST),  dtype=np.float32)

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < 0.7
    subset_masks[s] = mask.astype(np.float32)
    clf_s = train_lr(X_tr[mask], y_tr[mask])
    subset_margins[s] = lr_margin(clf_s, X_te, y_te)
    if s % 100 == 0:
        print(f"  {s}/{N_SUBSETS}  ({time.perf_counter()-t0:.0f}s)")

retrain_time = time.perf_counter() - t0
print(f"  Done in {retrain_time:.1f}s")

# ── Step 3: Attribution engines ───────────────────────────────────────────────
print("\nStep 3: Building engines...")
engine_dot  = AttributionEngine(store, estimator="dot")
engine_trak = AttributionEngine(store, estimator="trak", lambda_factor=LAMBDA)
print("  Ready.")

# ── Step 4: Batch attribution ─────────────────────────────────────────────────
print(f"\nStep 4: Batch attribution ({os.cpu_count()} cores)...")
coef, intercept = clf.coef_[0], clf.intercept_[0]
err_te = sigmoid(X_te @ coef + intercept) - y_te
G_te   = (err_te[:, None] * X_te).astype(np.float32)

t0 = time.perf_counter()
inf_dot = engine_dot.attribute_scores_batch(G_te)
t_dot = time.perf_counter() - t0
print(f"  [dot]  {t_dot:.1f}s")

t0 = time.perf_counter()
inf_trak = engine_trak.attribute_scores_batch(G_te)
t_trak = time.perf_counter() - t0
print(f"  [trak] {t_trak:.1f}s")

rand_inf = np.random.randn(N_TEST, N_TRAIN).astype(np.float32)

# ── Step 5: LDS ───────────────────────────────────────────────────────────────
print("\nStep 5: Computing LDS...")
lds_dot_mean,  lds_dot_std  = compute_lds(inf_dot,  subset_masks, subset_margins)
lds_trak_mean, lds_trak_std = compute_lds(inf_trak, subset_masks, subset_margins)
lds_rand_mean, lds_rand_std = compute_lds(rand_inf, subset_masks, subset_margins)

print()
print("=" * 65)
print(f"{'Variant':<35} {'LDS mean':>9} {'± std':>8} {'Time':>7}")
print("-" * 65)
print(f"{'TP-LL (dot product)':<35} {lds_dot_mean:>9.4f} {lds_dot_std:>8.4f} {t_dot:>6.1f}s")
print(f"{'TP-LL + TRAK estimator':<35} {lds_trak_mean:>9.4f} {lds_trak_std:>8.4f} {t_trak:>6.1f}s")
print(f"{'Random baseline':<35} {lds_rand_mean:>9.4f} {lds_rand_std:>8.4f}    ---")
print("=" * 65)

results = {
    "experiment": "exp10_covertype_lds",
    "dataset": "covertype_binary_class2_vs_rest",
    "n_train": N_TRAIN, "n_test": N_TEST, "n_subsets": N_SUBSETS,
    "n_features": X_tr.shape[1], "proj_dim": PROJ_DIM,
    "C": C_LR, "max_iter": MAX_ITER,
    "tp_ll_dot":  {"lds_mean": round(lds_dot_mean, 4),  "lds_std": round(lds_dot_std, 4),  "time_s": round(t_dot, 2)},
    "tp_ll_trak": {"lds_mean": round(lds_trak_mean, 4), "lds_std": round(lds_trak_std, 4), "time_s": round(t_trak, 2)},
    "random":     {"lds_mean": round(lds_rand_mean, 4), "lds_std": round(lds_rand_std, 4)},
    "retrain_time_s": round(retrain_time, 1),
    "test_accuracy": round(float(clf.score(X_te, y_te)), 4),
}

with open("results/exp10_covertype_lds.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp10_covertype_lds.json")
