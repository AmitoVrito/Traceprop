"""Experiment 21c: Easy-mode probe for Traceprop-SS.

Same shared 8-D no-zero-padding feature space as exp21b, but:
  - Labels are LINEAR in regime-specific columns (no multiplicative interaction
    that logistic regression can't represent).
  - Low noise so the model fits well.

This isolates the question: does Traceprop-SS work AT ALL when the model is
well-specified and the signal is clean? If it still fails here, the method
fails fundamentally. If it works here but fails on exp21b, the method only
works on linear-in-features models with low noise — a real but bounded scope.

Same baselines and pre-registered thresholds as exp21b.
"""

import json
import os
import time

import numpy as np
import scipy.linalg
from sklearn.linear_model import LogisticRegression

SEED = 42
rng  = np.random.default_rng(SEED)

N_BUR, N_PREV, N_APP = 5_000, 3_000, 2_000
N_TOTAL = N_BUR + N_PREV + N_APP
N_TEST  = 200
D_FEAT  = 8

COLS_BUR, COLS_PREV, COLS_APP = (0, 1), (3, 4), (6, 7)
SOURCE_COLS = {
    "bureau":               list(COLS_BUR),
    "previous_application": list(COLS_PREV),
    "application":          list(COLS_APP),
}
SOURCES = list(SOURCE_COLS.keys())

print("=" * 65)
print("Exp 21c: Easy-mode Traceprop-SS probe (linear labels, low noise)")
print("=" * 65)

# Shared 8-D feature space, all non-zero
def make_segment(n, mean_shift):
    return rng.standard_normal((n, D_FEAT)).astype(np.float64) + mean_shift[None, :]

shift_bur  = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
shift_prev = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
shift_app  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])

X_bur  = make_segment(N_BUR,  shift_bur)
X_prev = make_segment(N_PREV, shift_prev)
X_app  = make_segment(N_APP,  shift_app)

# LINEAR labels — well-specified for logistic regression. Low noise.
def label_linear(X, c1, c2, noise=0.1):
    score = X[:, c1] + X[:, c2] + rng.normal(0, noise, len(X))
    return (score > np.median(score)).astype(np.float32)

y_bur  = label_linear(X_bur,  *COLS_BUR)
y_prev = label_linear(X_prev, *COLS_PREV)
y_app  = label_linear(X_app,  *COLS_APP)

X_all = np.vstack([X_bur, X_prev, X_app])
y_all = np.concatenate([y_bur, y_prev, y_app])
source_all = (["bureau"]*N_BUR + ["previous_application"]*N_PREV +
              ["application"]*N_APP)

perm = rng.permutation(N_TOTAL)
X_all, y_all = X_all[perm], y_all[perm]
source_all = [source_all[i] for i in perm]

te_idx = np.arange(N_TEST)
tr_idx = np.arange(N_TEST, N_TOTAL)
X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]
src_tr = [source_all[i] for i in tr_idx]
src_te = [source_all[i] for i in te_idx]

X_mean = X_tr.mean(axis=0); X_std = X_tr.std(axis=0) + 1e-8
X_tr = (X_tr - X_mean) / X_std
X_te = (X_te - X_mean) / X_std

clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500,
                         random_state=SEED).fit(X_tr, y_tr)
coef, intercept = clf.coef_[0], clf.intercept_[0]
acc = clf.score(X_te, y_te)
print(f"\n  Model test accuracy: {acc:.4f}  (should be high if well-specified)")

# Gradients
def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
err_tr = sig(X_tr @ coef + intercept) - y_tr
G_tr   = err_tr[:, None] * X_tr
err_te = sig(X_te @ coef + intercept) - y_te
G_te   = err_te[:, None] * X_te

# Baselines
def p1(pred, true): return sum(p == t for p, t in zip(pred, true)) / len(true)

rng2 = np.random.default_rng(SEED + 1)
random_preds  = [SOURCES[i] for i in rng2.integers(0, 3, N_TEST)]
feat_preds    = [max({s: np.linalg.norm(X_te[j, c]) for s, c in SOURCE_COLS.items()},
                     key=lambda s: np.linalg.norm(X_te[j, SOURCE_COLS[s]]))
                 for j in range(N_TEST)]
grad_preds    = [max({s: np.linalg.norm(G_te[j, c]) for s, c in SOURCE_COLS.items()},
                     key=lambda s: np.linalg.norm(G_te[j, SOURCE_COLS[s]]))
                 for j in range(N_TEST)]
p1_random, p1_feat, p1_grad = p1(random_preds, src_te), p1(feat_preds, src_te), p1(grad_preds, src_te)

# Traceprop-SS
gram = G_tr.T @ G_tr
lam  = 1e-3 * np.trace(gram) / D_FEAT
gram += lam * np.eye(D_FEAT)
gf   = scipy.linalg.cho_factor(gram)

t0 = time.perf_counter()
V          = scipy.linalg.cho_solve(gf, G_te.T)
scores_all = (G_tr @ V).astype(np.float32)
ss_time    = time.perf_counter() - t0

masks    = {s: np.array([i for i, x in enumerate(src_tr) if x == s]) for s in SOURCES}
mean_inf = {s: scores_all[m, :].mean(axis=0) for s, m in masks.items()}
ss_preds = [max({s: abs(mean_inf[s][j]) for s in SOURCES},
                key=lambda s: abs(mean_inf[s][j]))
            for j in range(N_TEST)]
p1_ss = p1(ss_preds, src_te)

per_seg = {}
for seg in SOURCES:
    js = [j for j, s in enumerate(src_te) if s == seg]
    if js:
        c = sum(1 for j in js if ss_preds[j] == seg)
        per_seg[seg] = f"{c}/{len(js)} = {c/len(js):.3f}"

THRESH_R, THRESH_G = 0.20, 0.10
pass_r = (p1_ss - p1_random) >= THRESH_R
pass_g = (p1_ss - p1_grad)   >= THRESH_G

print()
print("  Correct-source P@1 (baseline 0.333):")
print(f"    random                    {p1_random:.3f}")
print(f"    argmax |x| per source     {p1_feat:.3f}")
print(f"    argmax |g_te| per source  {p1_grad:.3f}")
print(f"    Traceprop-SS              {p1_ss:.3f}  <-- method")
print()
print("  Per-segment Traceprop-SS P@1:")
for seg, v in per_seg.items():
    print(f"    {seg:22s}  {v}")
print()
print("  Pre-registered thresholds:")
print(f"    SS - random  = {p1_ss - p1_random:+.3f}  needs ≥ +0.20  -> "
      f"{'PASS' if pass_r else 'FAIL'}")
print(f"    SS - gradmag = {p1_ss - p1_grad:+.3f}  needs ≥ +0.10  -> "
      f"{'PASS' if pass_g else 'FAIL'}")
print(f"    OVERALL: {'PASS' if (pass_r and pass_g) else 'FAIL'}")
print("=" * 65)

out = {
    "experiment": "exp21c_ss_easy_mode",
    "description": "Easy-mode probe: linear labels, low noise, no zero-padding.",
    "model_accuracy": float(acc),
    "baselines": {
        "random":         round(p1_random, 4),
        "feature_argmax": round(p1_feat,   4),
        "gradmag_argmax": round(p1_grad,   4),
    },
    "traceprop_ss": {
        "correct_source_p1": round(p1_ss, 4),
        "per_segment_p1":    per_seg,
        "latency_ms_per_query": round(1000 * ss_time / N_TEST, 4),
    },
    "pre_registered": {
        "lift_over_random":   round(p1_ss - p1_random, 4),
        "lift_over_gradmag":  round(p1_ss - p1_grad,   4),
        "pass_random":        bool(pass_r),
        "pass_gradmag":       bool(pass_g),
        "overall_pass":       bool(pass_r and pass_g),
    },
}
os.makedirs("results", exist_ok=True)
with open("results/exp21c_ss_easy_mode.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp21c_ss_easy_mode.json")
