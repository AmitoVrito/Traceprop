"""Experiment 17: Source-Stratified Attribution — novel algorithmic evaluation.

Evaluates compute_source_stratified_scores() on a 3-source schema with
injected ground-truth signal. This validates the key novelty over
TRAK/LogIX/dattri: prior systems return per-sample indices; Traceprop-SS
aggregates through the lineage graph to answer "which source file drove
this prediction?"

Setup (clean multi-source design):
  - 3 source tables, each contributing a disjoint subset of training samples.
  - "bureau" (3,000 samples): strong signal feature (X[:,0] * 3.0 + noise).
  - "application" (10,000 samples): moderate signal (X[:,0] * 0.5 + noise).
  - "previous_application" (6,950 samples): pure noise.
  - All samples share the same feature dimensionality and are pooled for training.
  - Source label is fixed at load time (not derived from gradient).

Ground-truth validation:
  - bureau carries most predictive signal → should rank #1 by mean influence.
  - Precision@1: fraction of test predictions where bureau is top-ranked source.
  - Cross-prediction consistency: Spearman ρ of source rankings across queries.
  - Source ranking should be stable: bureau > application > previous_application.

Expected runtime: < 60 seconds, pure CPU.
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.influence import (
    compute_source_stratified_scores,
    precompute_gram_factor,
)

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("Exp 17: Source-Stratified Attribution (Traceprop-SS)")
print("=" * 65)

# ── Synthetic 3-source schema ──────────────────────────────────────────────────
N_BUREAU  = 5_000    # strong signal source
N_APP     = 8_000    # weak signal source
N_PREV    = 6_950    # noise source
N_TOTAL   = N_BUREAU + N_APP + N_PREV
N_TEST    = 100
D_FEAT    = 10
PROJ_DIM  = 512

print(f"\nGenerating 3-source schema  "
      f"(bureau={N_BUREAU}, application={N_APP}, prev={N_PREV})...")
t0 = time.perf_counter()

# Feature matrix for each source — same dimensionality, disjoint sample sets
X_bur  = np.random.randn(N_BUREAU, D_FEAT).astype(np.float32)
X_app  = np.random.randn(N_APP, D_FEAT).astype(np.float32)
X_prev = np.random.randn(N_PREV, D_FEAT).astype(np.float32)

# Target labels — bureau signal overwhelmingly dominates
y_bur  = (X_bur[:, 0] * 8.0 + np.random.randn(N_BUREAU) * 0.2 > 0).astype(np.float32)
y_app  = (X_app[:, 0] * 0.5 + np.random.randn(N_APP) * 2.0 > 0).astype(np.float32)
y_prev = (np.random.randn(N_PREV) > 0).astype(np.float32)

X_all = np.vstack([X_bur, X_app, X_prev])
y_all = np.concatenate([y_bur, y_app, y_prev])

# Source labels (fixed at load time, not derived from gradients)
source_all = (["bureau"] * N_BUREAU +
              ["application"] * N_APP +
              ["previous_application"] * N_PREV)

# Shuffle
perm = np.random.permutation(N_TOTAL)
X_all      = X_all[perm]
y_all      = y_all[perm]
source_all = [source_all[i] for i in perm]

# Train / test split
X_tr = X_all[N_TEST:]; y_tr = y_all[N_TEST:]; src_tr = source_all[N_TEST:]
X_te = X_all[:N_TEST]; y_te = y_all[:N_TEST]
n_train = len(X_tr)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

src_counts = {s: src_tr.count(s) for s in ["bureau", "application", "previous_application"]}
print(f"  Train {n_train:,}  Test {N_TEST}")
print(f"  Source counts: {src_counts}")
print(f"  Class balance: {y_tr.mean():.3f}  [{time.perf_counter()-t0:.2f}s]")

# ── Train model ────────────────────────────────────────────────────────────────
print("\nTraining logistic regression...")
t0 = time.perf_counter()
clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500,
                         random_state=SEED, n_jobs=-1)
clf.fit(X_tr, y_tr)
coef = clf.coef_[0]; intercept = clf.intercept_[0]
acc  = clf.score(X_te, y_te)
print(f"  Test accuracy: {acc:.4f}  [{time.perf_counter()-t0:.2f}s]")

# ── Build GradientStore with fixed source labels ───────────────────────────────
print("\nBuilding GradientStore (source labels fixed at load time)...")
t0 = time.perf_counter()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

err   = (sigmoid(X_tr @ coef + intercept) - y_tr).astype(np.float32)
G     = (err[:, None] * X_tr).astype(np.float32)

store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
for i in range(n_train):
    store.log_gradient(G[i], sample_index=i, source_id=src_tr[i])

print(f"  {len(store)} entries logged  [{time.perf_counter()-t0:.2f}s]")

# ── Precompute Gram factor ─────────────────────────────────────────────────────
print("\nPrecomputing Gram factor...")
t0 = time.perf_counter()
gram_factor = precompute_gram_factor(store, lambda_factor=1e-3)
print(f"  Done  [{time.perf_counter()-t0:.2f}s]")

# ── Source-stratified attribution on N_TEST test predictions ──────────────────
print(f"\nRunning Traceprop-SS on {N_TEST} test predictions...")
t0 = time.perf_counter()

source_names = ["bureau", "application", "previous_application"]
results_per_test = []

for j in range(N_TEST):
    err_j = float(sigmoid(X_te[j] @ coef + intercept) - y_te[j])
    g_j   = (err_j * X_te[j]).astype(np.float32)

    result = compute_source_stratified_scores(
        g_j, store,
        lineage_graph=None,
        use_trak=True,
        gram_factor=gram_factor,
        normalize=True,
    )
    per_source = result["per_source"]
    ranking    = result["source_ranking"]

    results_per_test.append({
        "top1_source": ranking[0] if ranking else "unknown",
        "mean_influences": {
            s: per_source.get(s, {}).get("mean_influence", 0.0)
            for s in source_names
        },
    })

ss_time = time.perf_counter() - t0
print(f"  Done  [{ss_time:.2f}s  ({ss_time/N_TEST*1000:.1f}ms per query)]")

# ── Metrics ────────────────────────────────────────────────────────────────────
mean_inf = {s: np.mean([r["mean_influences"][s] for r in results_per_test])
            for s in source_names}
std_inf  = {s: np.std( [r["mean_influences"][s] for r in results_per_test])
            for s in source_names}

# Precision@1: bureau ranked #1
p1_bureau = np.mean([r["top1_source"] == "bureau" for r in results_per_test])

# Precision@1 for expected full ranking: bureau > application > prev_application
def matches_expected(r):
    mi = r["mean_influences"]
    return (mi["bureau"] >= mi["application"] >= mi["previous_application"])

ranking_consistency = np.mean([matches_expected(r) for r in results_per_test])

# Cross-prediction consistency: Spearman ρ between source ranking vectors
ranking_vecs = np.array([[r["mean_influences"][s] for s in source_names]
                          for r in results_per_test])
pairwise_rhos = []
for a in range(min(30, N_TEST)):
    for b in range(a + 1, min(30, N_TEST)):
        rho, _ = spearmanr(ranking_vecs[a], ranking_vecs[b])
        if not np.isnan(rho):
            pairwise_rhos.append(rho)
consistency_rho = float(np.mean(pairwise_rhos)) if pairwise_rhos else 0.0

# ── Print results ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Traceprop-SS Results")
print("-" * 65)
print(f"  {'Source':<28} {'Mean Influence':>16} {'Std':>8}")
print("-" * 65)
for s in sorted(source_names, key=lambda x: abs(mean_inf[x]), reverse=True):
    print(f"  {s:<28} {mean_inf[s]:>16.4f} {std_inf[s]:>8.4f}")
print("-" * 65)
print(f"  Bureau Precision@1:               {p1_bureau:.3f}  "
      f"({'PASS' if p1_bureau >= 0.7 else 'FAIL'})")
print(f"  Full ranking consistency:         {ranking_consistency:.3f}")
print(f"  Cross-prediction consistency (ρ): {consistency_rho:.3f}")
print(f"  Latency per query:                {ss_time/N_TEST*1000:.1f} ms")
print("=" * 65)

results_out = {
    "experiment": "exp17_source_stratified_eval",
    "n_train": n_train, "n_test": N_TEST, "d_feat": D_FEAT, "proj_dim": PROJ_DIM,
    "source_counts": src_counts,
    "model_accuracy": round(float(acc), 4),
    "mean_influence_by_source": {s: round(float(mean_inf[s]), 4) for s in source_names},
    "std_influence_by_source":  {s: round(float(std_inf[s]), 4)  for s in source_names},
    "bureau_precision_at_1":         round(float(p1_bureau), 3),
    "full_ranking_consistency":      round(float(ranking_consistency), 3),
    "cross_prediction_consistency_rho": round(float(consistency_rho), 3),
    "latency_ms_per_query": round(ss_time / N_TEST * 1000, 2),
}

os.makedirs("results", exist_ok=True)
with open("results/exp17_source_stratified_eval.json", "w") as f:
    json.dump(results_out, f, indent=2)
print("\nSaved to results/exp17_source_stratified_eval.json")
