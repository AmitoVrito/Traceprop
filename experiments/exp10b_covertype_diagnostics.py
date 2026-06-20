"""Experiment 10b: Diagnostic re-run of Covertype LDS to defend the 0.9763 number.

Reviewer concern (W4): LDS ≈ 0.98 on real n=50K tabular data is unusually high.
Possible artifacts:
  (a) Subset-mask overlap (we used 70% inclusion; Park et al. 2023 use 50%) —
      with high inclusion, subset margins are highly correlated and inflate
      Spearman ρ.
  (b) Margin variance too low across subsets — any reasonable score correlates.
  (c) Attribution overfits training-time gradients.

Diagnostic: redo exp10's LDS pipeline at p={0.5, 0.7} subset inclusion and
report (i) pairwise subset Jaccard, (ii) margin variance per test sample,
(iii) LDS at N_subsets = {100, 250, 500}. Single-seed run; same data subsample
as exp10 (SEED=42).
"""

import json
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.attribution_engine import AttributionEngine
from traceprop.attribution.gradient_store import GradientStore

_N_CORES = str(os.cpu_count())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _N_CORES)

N_TRAIN, N_TEST, N_SUBSETS = 50_000, 500, 500
PROJ_DIM, LAMBDA, C_LR, MAX_ITER, SEED = 4096, 1e-3, 10.0, 200, 42

np.random.seed(SEED)
data = fetch_covtype()
X_all = data.data.astype(np.float32)
y_all = (data.target == 2).astype(np.float32)

idx0 = np.where(y_all == 0)[0]
idx1 = np.where(y_all == 1)[0]
np.random.shuffle(idx0); np.random.shuffle(idx1)
n1 = N_TRAIN // 2; n0 = N_TRAIN - n1
tr_idx = np.concatenate([idx0[:n0], idx1[:n1]]); np.random.shuffle(tr_idx)
te_idx = np.concatenate([idx0[n0:n0+N_TEST//2], idx1[n1:n1+N_TEST//2]])
X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def train_lr(X, y, max_iter=MAX_ITER, seed=SEED):
    return LogisticRegression(C=C_LR, solver="lbfgs", max_iter=max_iter,
                              random_state=seed).fit(X, y)
def lr_margin(clf, X, y):
    return (clf.decision_function(X) * (2 * y - 1)).astype(np.float32)

# Single shared full-model + gradient store
clf = train_lr(X_tr, y_tr)
coef, intercept = clf.coef_[0], clf.intercept_[0]
err = sigmoid(X_tr @ coef + intercept) - y_tr
G = (err[:, None] * X_tr).astype(np.float32)
store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
for i in range(len(X_tr)):
    store.log_gradient(G[i], sample_index=i, source_id="covertype")
engine_trak = AttributionEngine(store, estimator="trak", lambda_factor=LAMBDA)

err_te = sigmoid(X_te @ coef + intercept) - y_te
G_te = (err_te[:, None] * X_te).astype(np.float32)
print("Building TRAK influence matrix..."); t0 = time.perf_counter()
inf_trak = engine_trak.attribute_scores_batch(G_te)
print(f"  done {time.perf_counter()-t0:.1f}s")

def compute_lds(inf_matrix, masks, margins):
    predicted = inf_matrix @ masks.T
    scores = [spearmanr(predicted[i], margins[:, i]).statistic
              for i in range(len(predicted))]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)), float(np.std(scores))

results = {"setup": {"N_TRAIN": N_TRAIN, "N_TEST": N_TEST,
                     "N_SUBSETS": N_SUBSETS, "PROJ_DIM": PROJ_DIM},
           "runs": {}}

# Run with two inclusion rates
for p_incl in (0.5, 0.7):
    label = f"p={p_incl:.1f}"
    print(f"\n=== Inclusion rate {label} (Park et al. uses 0.5) ===")
    rng = np.random.default_rng(SEED + int(p_incl * 10))
    masks = (rng.random((N_SUBSETS, N_TRAIN)) < p_incl).astype(np.float32)

    # Pairwise Jaccard (sample of pairs)
    n_pairs = 100
    pair_a, pair_b = rng.integers(0, N_SUBSETS, n_pairs), rng.integers(0, N_SUBSETS, n_pairs)
    jaccards = []
    for a, b in zip(pair_a, pair_b):
        if a == b: continue
        inter = (masks[a] * masks[b]).sum()
        union = ((masks[a] + masks[b]) > 0).sum()
        jaccards.append(inter / max(union, 1))
    mean_jaccard = float(np.mean(jaccards))
    print(f"  Mean pairwise Jaccard (n=100): {mean_jaccard:.4f}")

    print(f"  Retraining {N_SUBSETS} subsets...")
    t0 = time.perf_counter()
    margins = np.zeros((N_SUBSETS, N_TEST), dtype=np.float32)
    for s in range(N_SUBSETS):
        clf_s = train_lr(X_tr[masks[s].astype(bool)], y_tr[masks[s].astype(bool)])
        margins[s] = lr_margin(clf_s, X_te, y_te)
        if (s + 1) % 100 == 0:
            print(f"    {s+1}/{N_SUBSETS}  [{time.perf_counter()-t0:.0f}s]")
    print(f"  Retrain {time.perf_counter()-t0:.1f}s")

    # Margin variance across subsets per test sample
    margin_std_per_test = margins.std(axis=0)
    mean_margin_std = float(margin_std_per_test.mean())
    print(f"  Mean margin std across subsets: {mean_margin_std:.4f}")

    # LDS at varying subset counts (cumulative)
    lds_by_n = {}
    for n in (100, 250, 500):
        mean, std = compute_lds(inf_trak, masks[:n], margins[:n])
        lds_by_n[n] = {"lds_mean": round(mean, 4), "lds_std": round(std, 4)}
        print(f"  LDS at N_subsets={n}:  {mean:.4f} ± {std:.4f}")

    results["runs"][label] = {
        "p_inclusion":            p_incl,
        "mean_pairwise_jaccard":  round(mean_jaccard, 4),
        "mean_margin_std":        round(mean_margin_std, 4),
        "lds_by_subset_count":    lds_by_n,
    }

os.makedirs("results", exist_ok=True)
with open("results/exp10b_covertype_diagnostics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary ===")
print(json.dumps(results, indent=2))
print("\nSaved to results/exp10b_covertype_diagnostics.json")
