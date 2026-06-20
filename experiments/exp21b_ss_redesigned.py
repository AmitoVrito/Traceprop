"""Experiment 21b: Redesigned Traceprop-SS validation.

Fixes the two leakage paths that made exp21 a tautology:

  (1) Zero-padding sparsity signature — each segment's non-owned feature columns
      were exactly zero, so (p-y)*x inherited a deterministic source-identifying
      sparsity pattern. The Gram-solve picked this up as source identity, not
      attribution.

  (2) Source identity ↔ feature support bijection — each segment's label was
      derived from exactly the columns it "owned," so source classification was
      equivalent to feature-support classification.

Redesign:
  - ALL 8 columns are non-zero for every sample (drawn from a shared joint
    distribution + segment-specific perturbations).
  - Each "regime" (bureau/prev_app/app) generates y from a different INTERACTION
    of two columns. Source identity now lives in WHICH gradient direction the
    sample's signal points along — exactly what attribution should discriminate.

Baselines (mandatory):
  - random (0.333 over 3 sources)
  - per-source feature magnitude: argmax over column-block ||x|| per source
  - per-source test-gradient magnitude: argmax over column-block ||g_te|| per source
  - Traceprop-SS (Gram-solve, mean influence per source)

Pre-registered pass thresholds (set BEFORE seeing the result):
  - SS > random by ≥20 pp (P@1 ≥ 0.533), AND
  - SS > best gradient-magnitude baseline by ≥10 pp.

If either fails, Traceprop-SS is no better than trivial baselines and the
contribution must be demoted to "ergonomic plumbing" (W1).
"""

import json
import os
import time

import numpy as np
import scipy.linalg
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

SEED = 42
rng = np.random.default_rng(SEED)

# Segment sizes — same as exp21
N_BUR, N_PREV, N_APP = 5_000, 3_000, 2_000
N_TOTAL = N_BUR + N_PREV + N_APP
N_TEST  = 200
D_FEAT  = 8

# Column ownership (informational only; NOT used by Traceprop-SS):
#   bureau   -> cols 0, 1
#   prev_app -> cols 3, 4
#   app      -> cols 6, 7
# Cols 2 and 5 are "shared noise" — every regime has some signal there.
COLS_BUR  = (0, 1)
COLS_PREV = (3, 4)
COLS_APP  = (6, 7)

print("=" * 65)
print("Exp 21b: Redesigned Traceprop-SS (no zero-padding leak)")
print("=" * 65)

# ── Generate shared 8-D feature distribution ──────────────────────────────────
# Every sample gets a non-zero value in every column. Column means/scales differ
# slightly per segment (realistic ETL: bureau-history samples have different
# typical aggregations than no-history samples), but no column is structurally
# zero for any segment.
print(f"\nGenerating shared 8-D feature space "
      f"(bureau={N_BUR}, prev_app={N_PREV}, app={N_APP})...")
t0 = time.perf_counter()

def make_segment(n, mean_shift):
    """Draw n samples from a shared 8-D Gaussian with small per-segment shift."""
    base = rng.standard_normal((n, D_FEAT)).astype(np.float64)
    return base + mean_shift[None, :]

# Per-segment mean shifts (small — segments overlap heavily in feature space)
shift_bur  = np.array([0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
shift_prev = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0])
shift_app  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2])

X_bur  = make_segment(N_BUR,  shift_bur)
X_prev = make_segment(N_PREV, shift_prev)
X_app  = make_segment(N_APP,  shift_app)

# ── Labels from CROSS-COLUMN INTERACTIONS specific to each regime ────────────
# bureau regime:  y depends on bur_col0 * bur_col1
# prev regime:    y depends on prev_col3 * prev_col4
# app regime:     y depends on app_col6 * app_col7
# Noise is meaningful (~0.5σ) so the task isn't trivial.

def label_from_interaction(X, c1, c2, noise_std=0.5):
    score = X[:, c1] * X[:, c2] + rng.normal(0, noise_std, len(X))
    return (score > np.median(score)).astype(np.float32)

y_bur  = label_from_interaction(X_bur,  *COLS_BUR)
y_prev = label_from_interaction(X_prev, *COLS_PREV)
y_app  = label_from_interaction(X_app,  *COLS_APP)

X_all = np.vstack([X_bur, X_prev, X_app]).astype(np.float64)
y_all = np.concatenate([y_bur, y_prev, y_app])
source_all = (["bureau"] * N_BUR +
              ["previous_application"] * N_PREV +
              ["application"] * N_APP)

# Sanity check: no structural zeros
nz = (X_all != 0).mean(axis=0)
print(f"  Column non-zero rate: {nz.round(3).tolist()}  (should be ~1.0 everywhere)")
assert (nz > 0.99).all(), "redesign goal failed: zero-padding still present"

# Shuffle and split
perm   = rng.permutation(N_TOTAL)
X_all  = X_all[perm]; y_all = y_all[perm]
source_all = [source_all[i] for i in perm]

te_idx = np.arange(N_TEST)
tr_idx = np.arange(N_TEST, N_TOTAL)
X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]
src_tr = [source_all[i] for i in tr_idx]
src_te = [source_all[i] for i in te_idx]

print(f"  Train {len(X_tr):,}  Test {N_TEST}")
print(f"  Class balance: {y_all.mean():.3f}  [{time.perf_counter()-t0:.2f}s]")

# Standardise — important now that columns are non-degenerate
X_mean = X_tr.mean(axis=0); X_std = X_tr.std(axis=0) + 1e-8
X_tr = (X_tr - X_mean) / X_std
X_te = (X_te - X_mean) / X_std

# ── Train logistic regression ─────────────────────────────────────────────────
print("\nTraining logistic regression...")
t0 = time.perf_counter()
clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500,
                         random_state=SEED, n_jobs=-1)
clf.fit(X_tr, y_tr)
coef = clf.coef_[0]; intercept = clf.intercept_[0]
acc = clf.score(X_te, y_te)
print(f"  Test accuracy: {acc:.4f}  [{time.perf_counter()-t0:.2f}s]")

# ── Helper: per-source column blocks ──────────────────────────────────────────
SOURCE_COLS = {
    "bureau":               list(COLS_BUR),
    "previous_application": list(COLS_PREV),
    "application":          list(COLS_APP),
}
SOURCES = list(SOURCE_COLS.keys())

# ── Gradients ─────────────────────────────────────────────────────────────────
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

err_tr = sigmoid(X_tr @ coef + intercept) - y_tr
G_tr   = err_tr[:, None] * X_tr                  # (n_train, D)
err_te = sigmoid(X_te @ coef + intercept) - y_te
G_te   = err_te[:, None] * X_te                  # (N_TEST, D)

# ── Baseline 1: random ────────────────────────────────────────────────────────
def p1_correct(pred_sources, true_sources):
    return sum(p == t for p, t in zip(pred_sources, true_sources)) / len(true_sources)

rng2 = np.random.default_rng(SEED + 1)
random_preds = [SOURCES[i] for i in rng2.integers(0, 3, N_TEST)]
p1_random = p1_correct(random_preds, src_te)

# ── Baseline 2: per-source FEATURE-magnitude argmax ──────────────────────────
# Assign source by which 2-column block has the largest ||x|| in the test sample.
feat_preds = []
for j in range(N_TEST):
    block_mag = {s: np.linalg.norm(X_te[j, cols]) for s, cols in SOURCE_COLS.items()}
    feat_preds.append(max(block_mag, key=block_mag.get))
p1_featmag = p1_correct(feat_preds, src_te)

# ── Baseline 3: per-source TEST-GRADIENT-magnitude argmax ────────────────────
# Assign source by which 2-column block has the largest |g_te| in the test sample.
grad_preds = []
for j in range(N_TEST):
    block_mag = {s: np.linalg.norm(G_te[j, cols]) for s, cols in SOURCE_COLS.items()}
    grad_preds.append(max(block_mag, key=block_mag.get))
p1_gradmag = p1_correct(grad_preds, src_te)

# ── Method: Traceprop-SS (TRAK Gram-solve, mean influence per source) ────────
gram = G_tr.T @ G_tr
lam  = 1e-3 * np.trace(gram) / D_FEAT
gram += lam * np.eye(D_FEAT)
gram_factor = scipy.linalg.cho_factor(gram)

t_ss = time.perf_counter()
V = scipy.linalg.cho_solve(gram_factor, G_te.T)        # (D, N_TEST)
scores_all = (G_tr @ V).astype(np.float32)             # (n_train, N_TEST)
ss_time = time.perf_counter() - t_ss

source_masks = {s: np.array([i for i, x in enumerate(src_tr) if x == s])
                for s in SOURCES}

mean_inf_matrix = {s: scores_all[idx, :].mean(axis=0) for s, idx in source_masks.items()}

ss_preds = []
for j in range(N_TEST):
    scores_j = {s: abs(mean_inf_matrix[s][j]) for s in SOURCES}
    ss_preds.append(max(scores_j, key=scores_j.get))
p1_ss = p1_correct(ss_preds, src_te)

# Per-segment breakdown for Traceprop-SS
per_segment = {}
for seg in SOURCES:
    te_segs = [j for j, s in enumerate(src_te) if s == seg]
    if te_segs:
        c = sum(1 for j in te_segs if ss_preds[j] == seg)
        per_segment[seg] = {"n_test": len(te_segs), "correct": c,
                            "p1": round(c / len(te_segs), 3)}

# Cross-prediction Spearman ρ
ranking_vecs = np.array([[mean_inf_matrix[s][j] for s in SOURCES]
                         for j in range(N_TEST)])
pairwise_rhos = []
for a in range(min(50, N_TEST)):
    for b in range(a + 1, min(50, N_TEST)):
        rho, _ = spearmanr(ranking_vecs[a], ranking_vecs[b])
        if not np.isnan(rho):
            pairwise_rhos.append(rho)
consistency_rho = float(np.mean(pairwise_rhos)) if pairwise_rhos else 0.0

mean_inf = {s: float(mean_inf_matrix[s].mean()) for s in SOURCES}
std_inf  = {s: float(mean_inf_matrix[s].std())  for s in SOURCES}

# ── Pre-registered pass thresholds ───────────────────────────────────────────
THRESH_RANDOM_LIFT  = 0.20    # SS - random ≥ 0.20
THRESH_GRADMAG_LIFT = 0.10    # SS - argmax_gradmag ≥ 0.10
best_baseline = max(p1_random, p1_featmag, p1_gradmag)

pass_random   = (p1_ss - p1_random)  >= THRESH_RANDOM_LIFT
pass_gradmag  = (p1_ss - p1_gradmag) >= THRESH_GRADMAG_LIFT
overall_pass  = pass_random and pass_gradmag

# ── Report ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Traceprop-SS Redesigned (shared columns, interaction labels)")
print("-" * 65)
print(f"  Model test accuracy:                  {acc:.4f}")
print(f"  Latency Traceprop-SS:                 {1000*ss_time/N_TEST:.3f} ms/query")
print()
print("  Source                  Mean Inf      Std")
print("  " + "-" * 50)
for s in SOURCES:
    print(f"  {s:22s} {mean_inf[s]:+9.4f}  {std_inf[s]:7.4f}")
print()
print("  Correct-source P@1 (baseline 0.333):")
print(f"    {'random':30s}  {p1_random:.3f}")
print(f"    {'argmax |x| per source':30s}  {p1_featmag:.3f}")
print(f"    {'argmax |g_te| per source':30s}  {p1_gradmag:.3f}")
print(f"    {'Traceprop-SS':30s}  {p1_ss:.3f}  <-- method")
print()
print("  Per-segment Traceprop-SS P@1:")
for seg, v in per_segment.items():
    print(f"    {seg:22s}  {v['correct']}/{v['n_test']} = {v['p1']:.3f}")
print()
print("  Cross-prediction Spearman ρ:           {:.3f}".format(consistency_rho))
print()
print("  Pre-registered pass thresholds:")
print(f"    SS - random   = {p1_ss - p1_random:+.3f}  (needs ≥ +{THRESH_RANDOM_LIFT:.2f})  "
      f"{'PASS' if pass_random else 'FAIL'}")
print(f"    SS - gradmag  = {p1_ss - p1_gradmag:+.3f}  (needs ≥ +{THRESH_GRADMAG_LIFT:.2f})  "
      f"{'PASS' if pass_gradmag else 'FAIL'}")
print(f"    OVERALL:                         {'PASS' if overall_pass else 'FAIL'}")
print("=" * 65)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "experiment": "exp21b_ss_redesigned",
    "description": ("Redesigned Traceprop-SS validation. Shared 8-D feature "
                    "space (no zero-padding), labels from regime-specific "
                    "column interactions, pre-registered pass thresholds."),
    "n_train": int(N_TOTAL - N_TEST),
    "n_test":  N_TEST,
    "d_feat":  D_FEAT,
    "model_accuracy": float(acc),
    "source_cols":   {s: list(c) for s, c in SOURCE_COLS.items()},
    "source_counts_train": {s: int(len(source_masks[s])) for s in SOURCES},
    "baselines": {
        "random":          round(p1_random,  4),
        "feature_argmax":  round(p1_featmag, 4),
        "gradmag_argmax":  round(p1_gradmag, 4),
    },
    "traceprop_ss": {
        "correct_source_p1": round(p1_ss, 4),
        "per_segment_p1":    per_segment,
        "cross_prediction_rho": round(consistency_rho, 4),
        "latency_ms_per_query": round(1000 * ss_time / N_TEST, 4),
        "mean_influence":   {s: round(mean_inf[s], 6) for s in SOURCES},
        "std_influence":    {s: round(std_inf[s],  6) for s in SOURCES},
    },
    "pre_registered_thresholds": {
        "lift_over_random":   THRESH_RANDOM_LIFT,
        "lift_over_gradmag":  THRESH_GRADMAG_LIFT,
        "pass_random":        bool(pass_random),
        "pass_gradmag":       bool(pass_gradmag),
        "overall_pass":       bool(overall_pass),
    },
}

os.makedirs("results", exist_ok=True)
with open("results/exp21b_ss_redesigned.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to results/exp21b_ss_redesigned.json")
