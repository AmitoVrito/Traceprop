"""Experiment 16: Unlearning on Covertype 50K — scale validation.

Addresses the SIGMOD reviewer concern that unlearning was only evaluated at
n=6,000. Runs the identical Traceprop gradient-correction pipeline on
Covertype (n=50,000, d=54), demonstrating that provenance-guided forget set
selection scales to production-size tabular datasets.

Methodology mirrors exp15 (Adult Income):
  - C=100 logistic regression (less L2 → more per-sample memorization)
  - Forget set = top-200 highest-influence training samples (0.4% of data)
  - 5-step multi-step gradient correction, ETA=0.1
  - Gap-closed metric: (forget_loss_method - orig) / (gold - orig) * 100
  - Random baseline: gradient ascent on 200 randomly selected samples

Expected runtime: ~5 min on CPU (sklearn lbfgs, 500 retraining subsets not needed).
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

_N_CORES = str(os.cpu_count())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _N_CORES)

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.attribution_engine import AttributionEngine
from traceprop.attribution.gradient_store import GradientStore

# ── Config ─────────────────────────────────────────────────────────────────────
N_TRAIN   = 50_000
N_TEST    = 5_000
FORGET_K  = 500      # 1.0% of training data (proportional scaling from exp15's 0.83%)
PROJ_DIM  = 4096
C_LR      = 100.0
MAX_ITER  = 500
ETA       = 0.05     # smaller step to avoid over-correction at larger n
N_STEPS   = 5
SEED      = 42
np.random.seed(SEED)

print("=" * 65)
print("Exp 16: Unlearning on Covertype 50K (scale validation)")
print(f"  N_train={N_TRAIN:,}, forget_k={FORGET_K}, C={C_LR}, steps={N_STEPS}")
print("=" * 65)

# ── Data ───────────────────────────────────────────────────────────────────────
print("\nLoading Covertype...")
t0 = time.perf_counter()
cov = fetch_covtype()
X_raw = cov.data.astype(np.float32)
# Binary: class 2 vs rest (same as exp10 LDS benchmark for consistency)
y_raw = (cov.target == 2).astype(np.float32)

idx = np.random.permutation(len(X_raw))
tr_idx = idx[:N_TRAIN]; te_idx = idx[N_TRAIN:N_TRAIN + N_TEST]
X_tr, y_tr = X_raw[tr_idx], y_raw[tr_idx]
X_te, y_te = X_raw[te_idx], y_raw[te_idx]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}  [{time.perf_counter()-t0:.1f}s]")
print(f"  Class balance: {y_tr.mean():.3f} positive")

# ── Helpers ────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_lr(X, y, C=C_LR):
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                             random_state=SEED, n_jobs=-1)
    clf.fit(X, y)
    return clf

def cross_entropy_loss(coef, intercept, X, y):
    p = sigmoid(X @ coef + intercept)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def clf_accuracy(coef, intercept, X, y):
    return ((sigmoid(X @ coef + intercept) > 0.5) == y).mean()

def build_store(X, y, coef, intercept):
    store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
    err = sigmoid(X @ coef + intercept) - y
    G = (err[:, None] * X).astype(np.float32)
    for i in range(len(X)):
        store.log_gradient(G[i], sample_index=i, source_id="covertype")
    return store

# ── Step 1: Train original model ───────────────────────────────────────────────
print("\nStep 1: Training original model (C=100, n=50K)...")
t0 = time.perf_counter()
clf_orig  = train_lr(X_tr, y_tr)
coef_orig = clf_orig.coef_[0].copy()
int_orig  = clf_orig.intercept_[0]
orig_loss = cross_entropy_loss(coef_orig, int_orig, X_tr, y_tr).mean()
orig_acc  = clf_accuracy(coef_orig, int_orig, X_te, y_te)
print(f"  Original — train loss: {orig_loss:.4f}, test acc: {orig_acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 2: Attribution → identify forget set ──────────────────────────────────
print(f"\nStep 2: Attribution → top-{FORGET_K} most influential samples...")
t0 = time.perf_counter()
store  = build_store(X_tr, y_tr, coef_orig, int_orig)
engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)

# Use a subset of test samples for attribution (500 is sufficient)
N_TEST_ATTR = 500
err_te  = sigmoid(X_te[:N_TEST_ATTR] @ coef_orig + int_orig) - y_te[:N_TEST_ATTR]
G_te    = (err_te[:, None] * X_te[:N_TEST_ATTR]).astype(np.float32)
inf_mat = engine.attribute_scores_batch(G_te)   # (N_TEST_ATTR, N_TRAIN)

mean_inf       = np.abs(inf_mat).mean(axis=0)
forget_indices = np.argsort(mean_inf)[-FORGET_K:]
retain_mask    = np.ones(N_TRAIN, dtype=bool)
retain_mask[forget_indices] = False
random_indices = np.random.choice(np.where(retain_mask)[0], FORGET_K, replace=False)

X_forget, y_forget = X_tr[forget_indices], y_tr[forget_indices]
X_retain, y_retain = X_tr[retain_mask],    y_tr[retain_mask]
orig_forget_loss = cross_entropy_loss(coef_orig, int_orig, X_forget, y_forget).mean()
print(f"  Forget: {len(X_forget)} samples  Retain: {len(X_retain)} samples")
print(f"  Original forget-set loss: {orig_forget_loss:.4f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 3: Gold model ─────────────────────────────────────────────────────────
print("\nStep 3: Training gold model (retrain on retain set, n=49,800)...")
t0 = time.perf_counter()
clf_gold  = train_lr(X_retain, y_retain)
coef_gold = clf_gold.coef_[0]; int_gold = clf_gold.intercept_[0]
gold_loss = cross_entropy_loss(coef_gold, int_gold, X_forget, y_forget).mean()
gold_acc  = clf_accuracy(coef_gold, int_gold, X_te, y_te)
print(f"  Gold — forget loss: {gold_loss:.4f}, test acc: {gold_acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: Traceprop unlearning ───────────────────────────────────────────────
print(f"\nStep 4: Traceprop gradient correction ({N_STEPS} steps, ETA={ETA})...")
t0 = time.perf_counter()
coef_ul = coef_orig.copy()
int_ul  = int_orig

for step in range(N_STEPS):
    err_f  = sigmoid(X_forget @ coef_ul + int_ul) - y_forget
    grad_w = (X_forget.T @ err_f) / len(X_forget)
    grad_b = err_f.mean()
    coef_ul += ETA * grad_w
    int_ul  += ETA * grad_b

ul_forget_loss = cross_entropy_loss(coef_ul, int_ul, X_forget, y_forget).mean()
ul_acc         = clf_accuracy(coef_ul, int_ul, X_te, y_te)
ul_time        = time.perf_counter() - t0
gap_closed_tp  = (ul_forget_loss - orig_forget_loss) / (gold_loss - orig_forget_loss) * 100
print(f"  Traceprop — forget loss: {ul_forget_loss:.4f}, test acc: {ul_acc:.4f}, "
      f"gap closed: {gap_closed_tp:.1f}%  [{ul_time:.3f}s]")

# ── Step 5: Random unlearning baseline ────────────────────────────────────────
print("\nStep 5: Random baseline...")
coef_rand = coef_orig.copy()
int_rand  = int_orig
X_rand    = X_tr[random_indices]; y_rand = y_tr[random_indices]

for step in range(N_STEPS):
    err_r  = sigmoid(X_rand @ coef_rand + int_rand) - y_rand
    grad_w = (X_rand.T @ err_r) / len(X_rand)
    grad_b = err_r.mean()
    coef_rand += ETA * grad_w
    int_rand  += ETA * grad_b

rand_forget_loss = cross_entropy_loss(coef_rand, int_rand, X_forget, y_forget).mean()
rand_acc         = clf_accuracy(coef_rand, int_rand, X_te, y_te)
gap_closed_rand  = (rand_forget_loss - orig_forget_loss) / (gold_loss - orig_forget_loss) * 100
print(f"  Random — forget loss: {rand_forget_loss:.4f}, test acc: {rand_acc:.4f}, "
      f"gap closed: {gap_closed_rand:.1f}%")

# ── Results ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Method':<20} {'Fgt Loss':>9} {'Test Acc':>9} {'Gap Closed':>11}")
print("-" * 65)
print(f"{'Original':<20} {orig_forget_loss:>9.4f} {orig_acc:>9.4f} {'—':>11}")
print(f"{'Gold (retrain)':<20} {gold_loss:>9.4f} {gold_acc:>9.4f} {'100%':>11}")
print(f"{'Traceprop':<20} {ul_forget_loss:>9.4f} {ul_acc:>9.4f} {gap_closed_tp:>10.1f}%")
print(f"{'Random':<20} {rand_forget_loss:>9.4f} {rand_acc:>9.4f} {gap_closed_rand:>10.1f}%")
print("=" * 65)

results = {
    "experiment": "exp16_covertype_unlearning",
    "dataset": "covertype",
    "n_train": N_TRAIN, "n_test": N_TEST, "forget_k": FORGET_K,
    "C": C_LR, "eta": ETA, "n_steps": N_STEPS,
    "original":   {"forget_loss": round(float(orig_forget_loss), 4), "test_acc": round(float(orig_acc), 4)},
    "gold":       {"forget_loss": round(float(gold_loss), 4),       "test_acc": round(float(gold_acc), 4)},
    "traceprop":  {"forget_loss": round(float(ul_forget_loss), 4),  "test_acc": round(float(ul_acc), 4),
                   "gap_closed_pct": round(float(gap_closed_tp), 1), "time_s": round(ul_time, 3)},
    "random":     {"forget_loss": round(float(rand_forget_loss), 4), "test_acc": round(float(rand_acc), 4),
                   "gap_closed_pct": round(float(gap_closed_rand), 1)},
}

os.makedirs("results", exist_ok=True)
with open("results/exp16_covertype_unlearning.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp16_covertype_unlearning.json")
