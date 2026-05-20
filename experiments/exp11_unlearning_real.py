"""Experiment 11: Unlearning on real data (Adult Income) + MIA forget quality.

Replaces the 1K synthetic unlearning benchmark with:
  1. Real Adult Income data (6K samples)
  2. Membership Inference Attack (MIA) forget quality score
     — standard metric from Guo et al. 2020 / Carlini et al. 2022

MIA forget quality: train a binary classifier to distinguish
"seen" (training) vs "unseen" (test) samples using their loss values.
After unlearning, the forget set should look "unseen" — the MIA
attack should fail on them. Forget quality = 1 - MIA attack accuracy
on forget set (1.0 = perfect forgetting, 0.5 = random = perfect).
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
from sklearn.model_selection import train_test_split

from traceprop.attribution.attribution_engine import AttributionEngine
from traceprop.attribution.gradient_store import GradientStore

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 6000
N_TEST    = 500
FORGET_K  = 50        # top-K most influential samples to forget
PROJ_DIM  = 4096
C_LR      = 10.0
MAX_ITER  = 200
ETA       = 0.1       # gradient correction step size
SEED      = 42
np.random.seed(SEED)

print("=" * 60)
print("Exp 11: Unlearning on Adult Income + MIA Forget Quality")
print(f"  N_train={N_TRAIN}, N_test={N_TEST}, forget_k={FORGET_K}")
print("=" * 60)

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading Adult Income...")
adult = fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
X_raw = adult.data.astype(np.float32)
y_raw = (adult.target == ">50K").astype(np.float32)

# Remove NaN rows
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

def train_lr(X, y):
    clf = LogisticRegression(C=C_LR, solver="lbfgs", max_iter=MAX_ITER,
                             random_state=SEED)
    clf.fit(X, y); return clf

def cross_entropy_loss(clf, X, y):
    """Per-sample cross-entropy loss."""
    p = sigmoid(clf.decision_function(X))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def build_store(X, y, clf, seed=SEED):
    store = GradientStore(proj_dim=PROJ_DIM, seed=seed)
    coef, intercept = clf.coef_[0], clf.intercept_[0]
    err = sigmoid(X @ coef + intercept) - y
    G = (err[:, None] * X).astype(np.float32)
    for i in range(len(X)):
        store.log_gradient(G[i], sample_index=i, source_id="adult")
    return store

def mia_forget_quality(clf_eval, X_forget, y_forget, X_retain_sample, y_retain_sample,
                       X_unseen, y_unseen):
    """
    MIA-based forget quality score.
    Train a shadow classifier on (loss, confidence) features to distinguish
    'seen' (retain) vs 'unseen' (held-out) samples.
    Apply to forget set — if unlearning worked, forget set looks 'unseen'.
    Returns: forget_quality in [0,1] where 1.0 = perfect forgetting.
    """
    # Features: [loss, |decision_function|]
    def features(clf, X, y):
        loss = cross_entropy_loss(clf, X, y)
        conf = np.abs(clf.decision_function(X))
        return np.stack([loss, conf], axis=1)

    # Shadow training data: seen=retain (label 1), unseen=held-out (label 0)
    n_shadow = min(len(X_retain_sample), len(X_unseen))
    F_seen   = features(clf_eval, X_retain_sample[:n_shadow], y_retain_sample[:n_shadow])
    F_unseen = features(clf_eval, X_unseen[:n_shadow],        y_unseen[:n_shadow])
    F_shadow = np.vstack([F_seen, F_unseen])
    y_shadow = np.concatenate([np.ones(n_shadow), np.zeros(n_shadow)])

    mia_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=SEED)
    mia_clf.fit(F_shadow, y_shadow)

    # Evaluate on forget set — predict probability of being "seen"
    F_forget = features(clf_eval, X_forget, y_forget)
    seen_prob = mia_clf.predict_proba(F_forget)[:, 1].mean()

    # forget_quality: 0.5 = looks unseen (perfect), 1.0 = still looks seen (bad)
    # Normalise: quality = 1 - (seen_prob - 0.5) / 0.5  (1.0=perfect, 0.0=not forgotten)
    forget_quality = float(1.0 - (seen_prob - 0.5) / 0.5)
    forget_quality = max(0.0, min(1.0, forget_quality))
    return forget_quality, float(seen_prob)

# ── Step 1: Train original model ──────────────────────────────────────────────
print("\nStep 1: Training original model...")
clf_orig = train_lr(X_tr, y_tr)
orig_loss = cross_entropy_loss(clf_orig, X_tr, y_tr).mean()
orig_acc  = clf_orig.score(X_te, y_te)
print(f"  Original — loss: {orig_loss:.4f}, test acc: {orig_acc:.4f}")

# ── Step 2: Identify forget set via attribution ───────────────────────────────
print(f"\nStep 2: Attribution → identify top-{FORGET_K} most influential samples...")
store = build_store(X_tr, y_tr, clf_orig)
engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)

coef, intercept = clf_orig.coef_[0], clf_orig.intercept_[0]
err_te = sigmoid(X_te @ coef + intercept) - y_te
G_te = (err_te[:, None] * X_te).astype(np.float32)
inf_matrix = engine.attribute_scores_batch(G_te)   # (N_TEST, N_TRAIN)

# Aggregate influence per training sample (mean absolute influence across test)
mean_influence = np.abs(inf_matrix).mean(axis=0)    # (N_TRAIN,)
forget_indices = np.argsort(mean_influence)[-FORGET_K:]
retain_mask = np.ones(N_TRAIN, dtype=bool)
retain_mask[forget_indices] = False

X_forget, y_forget = X_tr[forget_indices], y_tr[forget_indices]
X_retain, y_retain = X_tr[retain_mask],    y_tr[retain_mask]
print(f"  Forget set: {len(X_forget)} samples, retain: {len(X_retain)} samples")

# ── Step 3: Gold model (retrain without forget set) ───────────────────────────
print("\nStep 3: Training gold model (retrain from scratch on retain set)...")
t0 = time.perf_counter()
clf_gold = train_lr(X_retain, y_retain)
gold_loss = cross_entropy_loss(clf_gold, X_forget, y_forget).mean()
gold_acc  = clf_gold.score(X_te, y_te)
print(f"  Gold    — forget loss: {gold_loss:.4f}, test acc: {gold_acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: Traceprop unlearning (gradient correction) ───────────────────────
print("\nStep 4: Traceprop gradient correction unlearning...")
t0 = time.perf_counter()

coef_ul      = clf_orig.coef_[0].copy()
intercept_ul = clf_orig.intercept_[0].copy()

# Gradient ascent on forget set (reverse training signal)
err_f  = sigmoid(X_forget @ coef_ul + intercept_ul) - y_forget
grad_w = (X_forget.T @ err_f) / len(X_forget)
grad_b = err_f.mean()
coef_ul      += ETA * grad_w    # ascend = move away from forget set
intercept_ul += ETA * grad_b

# Package as a sklearn clf-like object
class UnlearnedModel:
    def __init__(self, coef, intercept):
        self.coef_      = coef[None, :]
        self.intercept_ = np.array([intercept])
    def decision_function(self, X):
        return X @ self.coef_[0] + self.intercept_[0]
    def score(self, X, y):
        return ((sigmoid(self.decision_function(X)) > 0.5) == y).mean()

clf_ul = UnlearnedModel(coef_ul, intercept_ul)
ul_forget_loss = cross_entropy_loss(clf_ul, X_forget, y_forget).mean()
ul_acc         = clf_ul.score(X_te, y_te)
ul_time        = time.perf_counter() - t0

orig_forget_loss = cross_entropy_loss(clf_orig, X_forget, y_forget).mean()
gap_closed = (ul_forget_loss - orig_forget_loss) / (gold_loss - orig_forget_loss) * 100

print(f"  Traceprop — forget loss: {ul_forget_loss:.4f}, test acc: {ul_acc:.4f}  [{ul_time:.2f}s]")
print(f"  Gap closed: {gap_closed:.1f}%")

# ── Step 5: MIA forget quality ────────────────────────────────────────────────
print("\nStep 5: MIA forget quality score...")
# Use held-out data as "unseen" for MIA shadow model
X_unseen = X_te[:200]; y_unseen = y_te[:200]
X_retain_s = X_retain[:200]; y_retain_s = y_retain[:200]

fq_orig, sp_orig = mia_forget_quality(clf_orig, X_forget, y_forget,
                                       X_retain_s, y_retain_s, X_unseen, y_unseen)
fq_gold, sp_gold = mia_forget_quality(clf_gold, X_forget, y_forget,
                                       X_retain_s, y_retain_s, X_unseen, y_unseen)
fq_ul,   sp_ul   = mia_forget_quality(clf_ul,   X_forget, y_forget,
                                       X_retain_s, y_retain_s, X_unseen, y_unseen)

print(f"  MIA forget quality (higher=better forgetting):")
print(f"    Original:  {fq_orig:.3f} (seen_prob={sp_orig:.3f})")
print(f"    Gold:      {fq_gold:.3f} (seen_prob={sp_gold:.3f})")
print(f"    Traceprop: {fq_ul:.3f}   (seen_prob={sp_ul:.3f})")

# ── Results ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Method':<18} {'Forget Loss':>11} {'Test Acc':>9} {'MIA Quality':>12} {'Gap Closed':>11}")
print("-" * 60)
print(f"{'Original':<18} {orig_forget_loss:>11.4f} {orig_acc:>9.4f} {fq_orig:>12.3f} {'—':>11}")
print(f"{'Gold (retrain)':<18} {gold_loss:>11.4f} {gold_acc:>9.4f} {fq_gold:>12.3f} {'100%':>11}")
print(f"{'Traceprop':<18} {ul_forget_loss:>11.4f} {ul_acc:>9.4f} {fq_ul:>12.3f} {gap_closed:>10.1f}%")
print("=" * 60)

results = {
    "experiment": "exp11_unlearning_real",
    "dataset": "adult_income",
    "n_train": N_TRAIN, "n_test": N_TEST, "forget_k": FORGET_K,
    "eta": ETA, "C": C_LR, "max_iter": MAX_ITER,
    "original": {
        "forget_loss": round(float(orig_forget_loss), 4),
        "test_acc": round(float(orig_acc), 4),
        "mia_forget_quality": round(fq_orig, 4),
    },
    "gold": {
        "forget_loss": round(float(gold_loss), 4),
        "test_acc": round(float(gold_acc), 4),
        "mia_forget_quality": round(fq_gold, 4),
    },
    "traceprop": {
        "forget_loss": round(float(ul_forget_loss), 4),
        "test_acc": round(float(ul_acc), 4),
        "mia_forget_quality": round(fq_ul, 4),
        "gap_closed_pct": round(float(gap_closed), 1),
        "time_s": round(ul_time, 3),
    },
}

with open("results/exp11_unlearning_real.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp11_unlearning_real.json")
