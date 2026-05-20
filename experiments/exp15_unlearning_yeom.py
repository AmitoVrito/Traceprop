"""Experiment 15: Unlearning on Adult Income with Yeom MIA + less regularization.

Fixes two weaknesses from exp11:
  1. C=100 (vs C=10): less L2 regularization → more per-sample memorization
     → gradient correction has more impact → higher gap closed.
  2. Yeom et al. threshold MIA instead of shadow classifier:
     - Predict 'member' if loss < median training loss
     - Before unlearning: forget set (high-influence, memorized) has low loss → member
     - After unlearning: forget set loss increases → non-member (quality → 1.0)
     - Gold (retrain without forget): forget set has high loss → non-member
     This gives a clear, non-trivial signal with meaningful before/after difference.
  3. Multi-step correction (5 steps, ETA=0.1 per step): more complete unlearning.
  4. Random baseline added for comparison.
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
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.attribution_engine import AttributionEngine
from traceprop.attribution.gradient_store import GradientStore

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 6000
N_TEST    = 500
FORGET_K  = 50
PROJ_DIM  = 4096
C_LR      = 100.0     # less regularization → more memorization (vs C=10 in exp11)
MAX_ITER  = 500        # more iterations for weaker-regularized model
ETA       = 0.1        # gradient correction step size
N_STEPS   = 5          # multi-step correction (vs 1 step in exp11)
SEED      = 42
np.random.seed(SEED)

print("=" * 60)
print("Exp 15: Unlearning (C=100, 5-step) + Yeom MIA")
print(f"  N_train={N_TRAIN}, forget_k={FORGET_K}, C={C_LR}, steps={N_STEPS}")
print("=" * 60)

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading Adult Income...")
adult = fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
X_raw = adult.data.astype(np.float32)
y_raw = (adult.target == ">50K").astype(np.float32)

valid = ~np.isnan(X_raw).any(axis=1)
X_raw, y_raw = X_raw[valid], y_raw[valid]

idx = np.random.permutation(len(X_raw))
tr_idx = idx[:N_TRAIN]; te_idx = idx[N_TRAIN:N_TRAIN+N_TEST]
X_tr, y_tr = X_raw[tr_idx], y_raw[tr_idx]
X_te, y_te = X_raw[te_idx], y_raw[te_idx]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_lr(X, y, C=C_LR):
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                             random_state=SEED)
    clf.fit(X, y); return clf

def cross_entropy_loss(coef, intercept, X, y):
    """Per-sample cross-entropy loss from raw parameters."""
    p = sigmoid(X @ coef + intercept)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def build_store(X, y, coef, intercept):
    store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
    err = sigmoid(X @ coef + intercept) - y
    G = (err[:, None] * X).astype(np.float32)
    for i in range(len(X)):
        store.log_gradient(G[i], sample_index=i, source_id="adult")
    return store

def clf_accuracy(coef, intercept, X, y):
    return ((sigmoid(X @ coef + intercept) > 0.5) == y).mean()

def yeom_mia_quality(coef, intercept, X_train, y_train, X_forget, y_forget):
    """
    Yeom et al. 2018 threshold MIA.
    Predict 'member' if loss < median training loss.
    Forget quality = P(forget set predicted as non-member).
    1.0 = perfect forgetting (all predicted non-member).
    0.0 = no forgetting (all predicted member).
    """
    train_losses  = cross_entropy_loss(coef, intercept, X_train, y_train)
    forget_losses = cross_entropy_loss(coef, intercept, X_forget, y_forget)
    threshold = np.median(train_losses)
    is_member = (forget_losses < threshold)
    quality = 1.0 - is_member.mean()
    return float(quality), float(is_member.mean()), float(forget_losses.mean())

# ── Step 1: Train original model ──────────────────────────────────────────────
print("\nStep 1: Training original model (C=100)...")
clf_orig   = train_lr(X_tr, y_tr)
coef_orig  = clf_orig.coef_[0].copy()
int_orig   = clf_orig.intercept_[0]
orig_loss  = cross_entropy_loss(coef_orig, int_orig, X_tr, y_tr).mean()
orig_acc   = clf_accuracy(coef_orig, int_orig, X_te, y_te)
print(f"  Original — train loss: {orig_loss:.4f}, test acc: {orig_acc:.4f}")

# ── Step 2: Attribution → identify forget set ─────────────────────────────────
print(f"\nStep 2: Attribution → top-{FORGET_K} most influential samples...")
store  = build_store(X_tr, y_tr, coef_orig, int_orig)
engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)

err_te  = sigmoid(X_te @ coef_orig + int_orig) - y_te
G_te    = (err_te[:, None] * X_te).astype(np.float32)
inf_mat = engine.attribute_scores_batch(G_te)   # (N_TEST, N_TRAIN)

mean_inf       = np.abs(inf_mat).mean(axis=0)
forget_indices = np.argsort(mean_inf)[-FORGET_K:]
retain_mask    = np.ones(N_TRAIN, dtype=bool)
retain_mask[forget_indices] = False
random_indices = np.random.choice(np.where(retain_mask)[0], FORGET_K, replace=False)

X_forget, y_forget = X_tr[forget_indices], y_tr[forget_indices]
X_retain, y_retain = X_tr[retain_mask],    y_tr[retain_mask]
print(f"  Forget: {len(X_forget)} samples  Retain: {len(X_retain)} samples")

orig_forget_loss = cross_entropy_loss(coef_orig, int_orig, X_forget, y_forget).mean()
mia_orig, mem_orig, fl_orig = yeom_mia_quality(coef_orig, int_orig, X_tr, y_tr,
                                                X_forget, y_forget)
print(f"  Original forget-set loss: {orig_forget_loss:.4f}  MIA quality: {mia_orig:.3f} "
      f"(member_rate={mem_orig:.3f})")

# ── Step 3: Gold model ────────────────────────────────────────────────────────
print("\nStep 3: Training gold model (retrain on retain set)...")
t0 = time.perf_counter()
clf_gold  = train_lr(X_retain, y_retain)
coef_gold = clf_gold.coef_[0]; int_gold = clf_gold.intercept_[0]
gold_loss = cross_entropy_loss(coef_gold, int_gold, X_forget, y_forget).mean()
gold_acc  = clf_accuracy(coef_gold, int_gold, X_te, y_te)
mia_gold, mem_gold, _ = yeom_mia_quality(coef_gold, int_gold,
                                          X_retain, y_retain, X_forget, y_forget)
print(f"  Gold — forget loss: {gold_loss:.4f}, test acc: {gold_acc:.4f}, "
      f"MIA quality: {mia_gold:.3f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: Traceprop unlearning (multi-step gradient correction) ─────────────
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
mia_ul, mem_ul, _ = yeom_mia_quality(coef_ul, int_ul, X_tr, y_tr, X_forget, y_forget)
print(f"  Traceprop — forget loss: {ul_forget_loss:.4f}, test acc: {ul_acc:.4f}, "
      f"MIA quality: {mia_ul:.3f}, gap closed: {gap_closed_tp:.1f}%  [{ul_time:.3f}s]")

# ── Step 5: Random unlearning baseline ───────────────────────────────────────
print("\nStep 5: Random unlearning baseline (ascent on random samples)...")
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
mia_rand, mem_rand, _ = yeom_mia_quality(coef_rand, int_rand, X_tr, y_tr, X_forget, y_forget)
print(f"  Random — forget loss: {rand_forget_loss:.4f}, test acc: {rand_acc:.4f}, "
      f"MIA quality: {mia_rand:.3f}, gap closed: {gap_closed_rand:.1f}%")

# ── Results ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"{'Method':<20} {'Fgt Loss':>9} {'Test Acc':>9} {'MIA Qual':>9} {'Gap Closed':>11}")
print("-" * 70)
print(f"{'Original':<20} {orig_forget_loss:>9.4f} {orig_acc:>9.4f} {mia_orig:>9.3f} {'—':>11}")
print(f"{'Gold (retrain)':<20} {gold_loss:>9.4f} {gold_acc:>9.4f} {mia_gold:>9.3f} {'100%':>11}")
print(f"{'Traceprop':<20} {ul_forget_loss:>9.4f} {ul_acc:>9.4f} {mia_ul:>9.3f} {gap_closed_tp:>10.1f}%")
print(f"{'Random':<20} {rand_forget_loss:>9.4f} {rand_acc:>9.4f} {mia_rand:>9.3f} {gap_closed_rand:>10.1f}%")
print("=" * 70)

results = {
    "experiment": "exp15_unlearning_yeom",
    "dataset": "adult_income",
    "n_train": N_TRAIN, "n_test": N_TEST, "forget_k": FORGET_K,
    "C": C_LR, "eta": ETA, "n_steps": N_STEPS, "max_iter": MAX_ITER,
    "original": {
        "forget_loss": round(float(orig_forget_loss), 4),
        "test_acc":    round(float(orig_acc), 4),
        "mia_quality": round(mia_orig, 4),
        "member_rate": round(mem_orig, 4),
    },
    "gold": {
        "forget_loss": round(float(gold_loss), 4),
        "test_acc":    round(float(gold_acc), 4),
        "mia_quality": round(mia_gold, 4),
        "member_rate": round(mem_gold, 4),
    },
    "traceprop": {
        "forget_loss":   round(float(ul_forget_loss), 4),
        "test_acc":      round(float(ul_acc), 4),
        "mia_quality":   round(mia_ul, 4),
        "member_rate":   round(mem_ul, 4),
        "gap_closed_pct": round(float(gap_closed_tp), 1),
        "time_s":        round(ul_time, 4),
    },
    "random": {
        "forget_loss":   round(float(rand_forget_loss), 4),
        "test_acc":      round(float(rand_acc), 4),
        "mia_quality":   round(mia_rand, 4),
        "member_rate":   round(mem_rand, 4),
        "gap_closed_pct": round(float(gap_closed_rand), 1),
    },
    "notes": ("C=100 (less L2 reg → more memorization). "
              "5-step multi-step gradient correction. "
              "Yeom et al. threshold MIA: member if loss < median train loss."),
}

with open("results/exp15_unlearning_yeom.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp15_unlearning_yeom.json")
