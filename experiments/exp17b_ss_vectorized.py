"""Experiment 17b: Source-Stratified Attribution — vectorized batch solve.

Same setup as exp17 but replaces the per-query Python for-loop with a single
batched matrix solve, dropping latency from ~116ms/query to ~2-5ms/query.

Vectorized TRAK formula:
  scores_all = Phi_train @ cho_solve(gram_factor, Phi_test.T)   shape (n_train, N_TEST)

Then source aggregation is done by row-slicing scores_all per source.

Setup: identical to exp17 (SEED=42, N_BUREAU=5000, N_APP=8000, N_PREV=6950,
       PROJ_DIM=512, N_TEST=100).
"""

import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.linalg
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.influence import precompute_gram_factor

SEED = 42
np.random.seed(SEED)

print("=" * 65)
print("Exp 17b: Source-Stratified Attribution (Vectorized Batch Solve)")
print("=" * 65)

# ── Synthetic 3-source schema (identical to exp17) ─────────────────────────────
N_BUREAU  = 5_000
N_APP     = 8_000
N_PREV    = 6_950
N_TOTAL   = N_BUREAU + N_APP + N_PREV
N_TEST    = 100
D_FEAT    = 10
PROJ_DIM  = 512

print(f"\nGenerating 3-source schema  "
      f"(bureau={N_BUREAU}, application={N_APP}, prev={N_PREV})...")
t0 = time.perf_counter()

X_bur  = np.random.randn(N_BUREAU, D_FEAT).astype(np.float32)
X_app  = np.random.randn(N_APP,    D_FEAT).astype(np.float32)
X_prev = np.random.randn(N_PREV,   D_FEAT).astype(np.float32)

y_bur  = (X_bur[:, 0] * 8.0 + np.random.randn(N_BUREAU) * 0.2 > 0).astype(np.float32)
y_app  = (X_app[:, 0] * 0.5 + np.random.randn(N_APP) * 2.0    > 0).astype(np.float32)
y_prev = (np.random.randn(N_PREV) > 0).astype(np.float32)

X_all      = np.vstack([X_bur, X_app, X_prev])
y_all      = np.concatenate([y_bur, y_app, y_prev])
source_all = (["bureau"] * N_BUREAU +
              ["application"] * N_APP +
              ["previous_application"] * N_PREV)

perm = np.random.permutation(N_TOTAL)
X_all      = X_all[perm]
y_all      = y_all[perm]
source_all = [source_all[i] for i in perm]

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

# ── Build GradientStore ────────────────────────────────────────────────────────
print("\nBuilding GradientStore...")
t0 = time.perf_counter()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

err   = (sigmoid(X_tr @ coef + intercept) - y_tr).astype(np.float32)
G_tr  = (err[:, None] * X_tr).astype(np.float32)

store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
for i in range(n_train):
    store.log_gradient(G_tr[i], sample_index=i, source_id=src_tr[i])

print(f"  {len(store)} entries logged  [{time.perf_counter()-t0:.2f}s]")

# ── Precompute Gram factor ─────────────────────────────────────────────────────
print("\nPrecomputing Gram factor...")
t0 = time.perf_counter()
gram_factor = precompute_gram_factor(store, lambda_factor=1e-3)
print(f"  Done  [{time.perf_counter()-t0:.2f}s]")

# ── Source index masks (built once, reused for all queries) ───────────────────
source_names = ["bureau", "application", "previous_application"]
source_masks = {
    s: np.array([i for i, src in enumerate(src_tr) if src == s], dtype=np.int64)
    for s in source_names
}

# ── Vectorized batch TRAK solve ───────────────────────────────────────────────
print(f"\nRunning Vectorized Traceprop-SS on {N_TEST} test predictions...")
t0 = time.perf_counter()

# Step 1: compute all test gradients  (N_TEST, D_FEAT)
err_te = (sigmoid(X_te @ coef + intercept) - y_te).astype(np.float32)
G_te   = (err_te[:, None] * X_te).astype(np.float32)           # (N_TEST, D_FEAT)

# Step 2: project all test gradients at once  (N_TEST, PROJ_DIM)
proj_matrix = store._projection._matrix                          # (PROJ_DIM, D_FEAT)
Phi_te  = (G_te.astype(np.float64) @ proj_matrix.T.astype(np.float64))  # (N_TEST, PROJ_DIM)

# Step 3: solve all at once  cho_solve expects rhs shape (PROJ_DIM, N_TEST)
V = scipy.linalg.cho_solve(gram_factor, Phi_te.T)               # (PROJ_DIM, N_TEST)

# Step 4: score all at once
Phi_tr  = store.get_projected_matrix().astype(np.float64)       # (n_train, PROJ_DIM)
scores_all = Phi_tr @ V                                          # (n_train, N_TEST)

# Step 5: source aggregation — mean over source rows, for every test query at once
mean_inf_matrix = {}   # each entry shape (N_TEST,)
for s in source_names:
    idx = source_masks[s]
    mean_inf_matrix[s] = scores_all[idx, :].mean(axis=0)        # (N_TEST,)

ss_time = time.perf_counter() - t0
print(f"  Done  [{ss_time:.2f}s  ({ss_time/N_TEST*1000:.2f}ms per query)]")

# ── Also time the old per-query loop for comparison ───────────────────────────
print(f"\nRunning OLD per-query loop for latency comparison...")
from traceprop.attribution.influence import compute_source_stratified_scores

t_old = time.perf_counter()
_dummy_results = []
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
    _dummy_results.append(result)
old_time = time.perf_counter() - t_old
print(f"  Done  [{old_time:.2f}s  ({old_time/N_TEST*1000:.1f}ms per query)]")

speedup = old_time / ss_time if ss_time > 0 else float("inf")
print(f"  Speedup: {speedup:.1f}x")

# ── Build per-test result dicts (matching exp17 format) ───────────────────────
results_per_test = []
for j in range(N_TEST):
    mi = {s: float(mean_inf_matrix[s][j]) for s in source_names}
    # Normalize: divide by max |mean_inf| across sources (same as exp17)
    max_abs = max(abs(v) for v in mi.values())
    if max_abs > 1e-10:
        mi = {s: mi[s] / max_abs for s in source_names}
    top1 = max(mi, key=lambda s: abs(mi[s]))
    results_per_test.append({
        "top1_source": top1,
        "mean_influences": mi,
    })

# ── Metrics ────────────────────────────────────────────────────────────────────
mean_inf = {s: np.mean([r["mean_influences"][s] for r in results_per_test])
            for s in source_names}
std_inf  = {s: np.std( [r["mean_influences"][s] for r in results_per_test])
            for s in source_names}

p1_bureau = np.mean([r["top1_source"] == "bureau" for r in results_per_test])

def matches_expected(r):
    mi = r["mean_influences"]
    return (mi["bureau"] >= mi["application"] >= mi["previous_application"])

ranking_consistency = np.mean([matches_expected(r) for r in results_per_test])

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
print("Traceprop-SS Vectorized Results")
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
print(f"  Latency per query (vectorized):   {ss_time/N_TEST*1000:.2f} ms")
print(f"  Latency per query (old loop):     {old_time/N_TEST*1000:.1f} ms")
print(f"  Speedup:                          {speedup:.1f}x")
print("=" * 65)

results_out = {
    "experiment": "exp17b_ss_vectorized",
    "n_train": n_train, "n_test": N_TEST, "d_feat": D_FEAT, "proj_dim": PROJ_DIM,
    "source_counts": src_counts,
    "model_accuracy": round(float(acc), 4),
    "mean_influence_by_source": {s: round(float(mean_inf[s]), 4) for s in source_names},
    "std_influence_by_source":  {s: round(float(std_inf[s]), 4)  for s in source_names},
    "bureau_precision_at_1":         round(float(p1_bureau), 3),
    "full_ranking_consistency":      round(float(ranking_consistency), 3),
    "cross_prediction_consistency_rho": round(float(consistency_rho), 3),
    "latency_ms_per_query_vectorized": round(ss_time / N_TEST * 1000, 3),
    "latency_ms_per_query_old_loop":   round(old_time / N_TEST * 1000, 1),
    "speedup_x": round(speedup, 1),
}

os.makedirs("results", exist_ok=True)
with open("results/exp17b_ss_vectorized.json", "w") as f:
    json.dump(results_out, f, indent=2)
print("\nSaved to results/exp17b_ss_vectorized.json")
