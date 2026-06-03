"""Experiment 19: Tuned Unlearning — Step-Count Sweep.

Addresses reviewer concern: "why not tune to exactly 100% gap-closed?"
Shows that Traceprop's over-correction is a tunable parameter, not a flaw.
By sweeping N_STEPS in [1, 2, 3, 4, 5], we can land near 100% gap-closed.

Datasets:
  - Adult Income: n=6000, forget_k=50, C=100, ETA=0.1 (same as exp15)
  - Covertype 50K: n=50000, forget_k=500, C=100, ETA=0.05 (same as exp16)

For each dataset:
  1. Build original model + attribution pipeline (same pipeline as exp15/exp16)
  2. Identify forget set (same top-k selection)
  3. Train gold model (retrain on retain set)
  4. Run gradient correction for steps 1..5, record forget loss + gap + test acc
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
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.attribution_engine import AttributionEngine
from traceprop.attribution.gradient_store import GradientStore

SEED = 42
np.random.seed(SEED)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_lr(X, y, C, max_iter, n_jobs=1):
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter,
                             random_state=SEED, n_jobs=n_jobs)
    clf.fit(X, y)
    return clf


def cross_entropy_loss(coef, intercept, X, y):
    p = sigmoid(X @ coef + intercept)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clf_accuracy(coef, intercept, X, y):
    return ((sigmoid(X @ coef + intercept) > 0.5) == y).mean()


def build_store(X, y, coef, intercept, proj_dim, source_id):
    store = GradientStore(proj_dim=proj_dim, seed=SEED)
    err = sigmoid(X @ coef + intercept) - y
    G = (err[:, None] * X).astype(np.float32)
    for i in range(len(X)):
        store.log_gradient(G[i], sample_index=i, source_id=source_id)
    return store


def run_sweep(dataset_name, X_tr, y_tr, X_te, y_te,
              forget_k, C_lr, max_iter, eta, proj_dim, n_test_attr,
              source_id, n_jobs=1):
    """Full sweep over N_STEPS for one dataset. Returns list of per-step dicts."""

    print(f"\n{'=' * 65}")
    print(f"Dataset: {dataset_name}")
    print(f"  n_train={len(X_tr):,}, forget_k={forget_k}, C={C_lr}, ETA={eta}")
    print('=' * 65)

    # ── Original model ──────────────────────────────────────────────────────────
    print("\nTraining original model...")
    t0 = time.perf_counter()
    clf_orig  = train_lr(X_tr, y_tr, C=C_lr, max_iter=max_iter, n_jobs=n_jobs)
    coef_orig = clf_orig.coef_[0].copy()
    int_orig  = clf_orig.intercept_[0]
    orig_acc  = clf_accuracy(coef_orig, int_orig, X_te, y_te)
    print(f"  Train loss: {cross_entropy_loss(coef_orig, int_orig, X_tr, y_tr).mean():.4f}, "
          f"test acc: {orig_acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

    # ── Attribution → forget set ────────────────────────────────────────────────
    print(f"\nBuilding attribution store, identifying top-{forget_k} forget set...")
    t0 = time.perf_counter()
    store  = build_store(X_tr, y_tr, coef_orig, int_orig, proj_dim, source_id)
    engine = AttributionEngine(store, estimator="trak", lambda_factor=1e-3)

    err_te  = sigmoid(X_te[:n_test_attr] @ coef_orig + int_orig) - y_te[:n_test_attr]
    G_te    = (err_te[:, None] * X_te[:n_test_attr]).astype(np.float32)
    inf_mat = engine.attribute_scores_batch(G_te)

    mean_inf       = np.abs(inf_mat).mean(axis=0)
    forget_indices = np.argsort(mean_inf)[-forget_k:]
    retain_mask    = np.ones(len(X_tr), dtype=bool)
    retain_mask[forget_indices] = False

    X_forget, y_forget = X_tr[forget_indices], y_tr[forget_indices]
    X_retain, y_retain = X_tr[retain_mask],    y_tr[retain_mask]
    orig_forget_loss   = cross_entropy_loss(coef_orig, int_orig, X_forget, y_forget).mean()
    print(f"  Forget: {len(X_forget)} samples, Retain: {len(X_retain)} samples")
    print(f"  Original forget-set loss: {orig_forget_loss:.4f}  [{time.perf_counter()-t0:.1f}s]")

    # ── Gold model ──────────────────────────────────────────────────────────────
    print("\nTraining gold model (retrain on retain set)...")
    t0 = time.perf_counter()
    clf_gold  = train_lr(X_retain, y_retain, C=C_lr, max_iter=max_iter, n_jobs=n_jobs)
    coef_gold = clf_gold.coef_[0]
    int_gold  = clf_gold.intercept_[0]
    gold_loss = cross_entropy_loss(coef_gold, int_gold, X_forget, y_forget).mean()
    gold_acc  = clf_accuracy(coef_gold, int_gold, X_te, y_te)
    print(f"  Gold — forget loss: {gold_loss:.4f}, test acc: {gold_acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

    gap_denom = gold_loss - orig_forget_loss

    # ── Step sweep ──────────────────────────────────────────────────────────────
    print(f"\nStep sweep (ETA={eta}, steps 1..5)...")
    step_results = []
    coef_ul = coef_orig.copy()
    int_ul  = int_orig

    for step in range(1, 6):
        # Apply one more correction step (cumulative from previous)
        err_f  = sigmoid(X_forget @ coef_ul + int_ul) - y_forget
        grad_w = (X_forget.T @ err_f) / len(X_forget)
        grad_b = err_f.mean()
        coef_ul = coef_ul + eta * grad_w
        int_ul  = int_ul  + eta * grad_b

        fgt_loss   = cross_entropy_loss(coef_ul, int_ul, X_forget, y_forget).mean()
        test_acc   = clf_accuracy(coef_ul, int_ul, X_te, y_te)
        gap_closed = (fgt_loss - orig_forget_loss) / gap_denom * 100

        step_results.append({
            "steps": step,
            "forget_loss": round(float(fgt_loss), 4),
            "gap_closed_pct": round(float(gap_closed), 1),
            "test_acc": round(float(test_acc), 4),
        })
        print(f"  Step {step}: fgt_loss={fgt_loss:.4f}  gap_closed={gap_closed:.1f}%  "
              f"test_acc={test_acc:.4f}")

    # Find step closest to 100%
    best = min(step_results, key=lambda r: abs(r["gap_closed_pct"] - 100.0))
    print(f"\n  Best step (closest to 100%): step={best['steps']} "
          f"→ gap_closed={best['gap_closed_pct']}%, test_acc={best['test_acc']}")

    return {
        "dataset": dataset_name,
        "n_train": len(X_tr),
        "forget_k": forget_k,
        "C": C_lr,
        "eta": eta,
        "original_forget_loss": round(float(orig_forget_loss), 4),
        "gold_forget_loss": round(float(gold_loss), 4),
        "gold_test_acc": round(float(gold_acc), 4),
        "steps": step_results,
        "best_step": best,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Adult Income
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading Adult Income...")
adult = fetch_openml("adult", version=2, as_frame=False, parser="liac-arff")
X_raw = adult.data.astype(np.float32)
y_raw = (adult.target == ">50K").astype(np.float32)
valid = ~np.isnan(X_raw).any(axis=1)
X_raw, y_raw = X_raw[valid], y_raw[valid]

rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X_raw))
X_tr_a = X_raw[idx[:6000]];  y_tr_a = y_raw[idx[:6000]]
X_te_a = X_raw[idx[6000:6500]]; y_te_a = y_raw[idx[6000:6500]]

sc_a = StandardScaler()
X_tr_a = sc_a.fit_transform(X_tr_a).astype(np.float32)
X_te_a = sc_a.transform(X_te_a).astype(np.float32)

adult_results = run_sweep(
    dataset_name="Adult Income",
    X_tr=X_tr_a, y_tr=y_tr_a, X_te=X_te_a, y_te=y_te_a,
    forget_k=50, C_lr=100.0, max_iter=500, eta=0.1,
    proj_dim=4096, n_test_attr=500, source_id="adult",
    n_jobs=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Covertype 50K
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\nLoading Covertype...")
cov    = fetch_covtype()
X_raw2 = cov.data.astype(np.float32)
y_raw2 = (cov.target == 2).astype(np.float32)

rng2 = np.random.RandomState(SEED)
idx2 = rng2.permutation(len(X_raw2))
X_tr_c = X_raw2[idx2[:50000]];       y_tr_c = y_raw2[idx2[:50000]]
X_te_c = X_raw2[idx2[50000:55000]];  y_te_c = y_raw2[idx2[50000:55000]]

sc_c = StandardScaler()
X_tr_c = sc_c.fit_transform(X_tr_c).astype(np.float32)
X_te_c = sc_c.transform(X_te_c).astype(np.float32)

cov_results = run_sweep(
    dataset_name="Covertype",
    X_tr=X_tr_c, y_tr=y_tr_c, X_te=X_te_c, y_te=y_te_c,
    forget_k=500, C_lr=100.0, max_iter=500, eta=0.05,
    proj_dim=4096, n_test_attr=500, source_id="covertype",
    n_jobs=-1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Combined output table
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print(f"{'Dataset':<14} {'Steps':>5}  {'Fgt Loss':>8}  {'Gap Closed':>10}  {'Test Acc':>8}")
print("-" * 65)
for r in adult_results["steps"]:
    marker = " <-- tuned" if r["steps"] == adult_results["best_step"]["steps"] else ""
    print(f"{'Adult Income':<14} {r['steps']:>5}  {r['forget_loss']:>8.4f}  "
          f"{r['gap_closed_pct']:>9.1f}%  {r['test_acc']:>8.4f}{marker}")
print()
for r in cov_results["steps"]:
    marker = " <-- tuned" if r["steps"] == cov_results["best_step"]["steps"] else ""
    print(f"{'Covertype':<14} {r['steps']:>5}  {r['forget_loss']:>8.4f}  "
          f"{r['gap_closed_pct']:>9.1f}%  {r['test_acc']:>8.4f}{marker}")
print("=" * 65)

print(f"\nAdult Income: best step = {adult_results['best_step']['steps']} "
      f"→ gap_closed = {adult_results['best_step']['gap_closed_pct']}%")
print(f"Covertype:    best step = {cov_results['best_step']['steps']} "
      f"→ gap_closed = {cov_results['best_step']['gap_closed_pct']}%")

# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "experiment": "exp19_tuned_unlearning",
    "description": (
        "Step-count sweep [1..5] to show over-correction is tunable. "
        "Same forget sets and pipelines as exp15 (Adult Income) and exp16 (Covertype)."
    ),
    "adult_income": adult_results,
    "covertype": cov_results,
}

os.makedirs(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
    exist_ok=True
)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "exp19_tuned_unlearning.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
