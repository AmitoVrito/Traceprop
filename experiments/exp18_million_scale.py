"""
exp18_million_scale.py — Systems benchmark: Traceprop at n=1,000,000 training samples.

This is a throughput/latency benchmark, NOT an ML quality benchmark.
No LDS retraining. Tests that the GradientStore + attribution pipeline
works at million-scale and measures wall-clock times.

Config: N_TRAIN=1_000_000, D_FEAT=20, PROJ_DIM=512, SEED=42
"""

import warnings
warnings.filterwarnings("ignore")

import json
import sys
import time
from pathlib import Path

import numpy as np

# Make sure we can import traceprop from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from traceprop.attribution.gradient_store import GradientStore, RandomProjection

# ── Config ──────────────────────────────────────────────────────────────────
N_TRAIN  = 1_000_000
D_FEAT   = 20
PROJ_DIM = 512
SEED     = 42
N_TEST   = 10
TOP_K    = 10

rng = np.random.default_rng(SEED)

results = {}

# ── 1. Data generation ───────────────────────────────────────────────────────
t0 = time.perf_counter()
X = rng.standard_normal((N_TRAIN, D_FEAT)).astype(np.float32)
w = rng.standard_normal(D_FEAT).astype(np.float32)
logits = X @ w
y = (logits > 0).astype(np.float32)
t_data = time.perf_counter() - t0
print(f"[1] Data generation:       {t_data:.3f}s  (N={N_TRAIN:,}, d={D_FEAT})")
results["data_generation_s"] = round(t_data, 4)

# ── 2. Gradient computation (vectorized) ─────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))

t0 = time.perf_counter()
err = (sigmoid(X @ w) - y).astype(np.float32)   # (N,)
G   = err[:, None] * X                            # (N, d)  — per-sample gradients
t_grad = time.perf_counter() - t0
print(f"[2] Gradient computation:  {t_grad:.3f}s  (G shape {G.shape})")
results["gradient_computation_s"] = round(t_grad, 4)

# ── 3. GradientStore bulk build ───────────────────────────────────────────────
# (a) Init store with one sample so the RandomProjection object is created
store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
store.log_gradient(G[0], sample_index=0)          # seeds _projection
proj_matrix = store._projection._matrix           # shape (PROJ_DIM, D_FEAT), float32

# (b) Batch-project all N gradients: Phi = G @ proj_matrix.T
#     G: (N, d)  proj_matrix: (PROJ_DIM, d) → Phi: (N, PROJ_DIM)
Phi = np.empty((N_TRAIN, PROJ_DIM), dtype=np.float32)

t0 = time.perf_counter()
np.matmul(G, proj_matrix.T, out=Phi)              # single BLAS call
t_store = time.perf_counter() - t0
print(f"[3] GradientStore build:   {t_store:.3f}s  (Phi shape {Phi.shape})")
results["gradientstore_build_s"] = round(t_store, 4)

# ── 4. Attribution scoring ────────────────────────────────────────────────────
# Generate 10 test samples and compute dot-product attribution against N train rows
X_test = rng.standard_normal((N_TEST, D_FEAT)).astype(np.float32)
err_test = (sigmoid(X_test @ w) - (X_test @ w > 0).astype(np.float32)).astype(np.float32)
G_test   = (err_test[:, None] * X_test).astype(np.float32)  # (N_TEST, d)
Phi_test = (G_test @ proj_matrix.T).astype(np.float32)      # (N_TEST, PROJ_DIM)

# Warm-up
_ = Phi @ Phi_test[0]

query_times = []
for i in range(N_TEST):
    phi_q = Phi_test[i]                            # (PROJ_DIM,)
    t0 = time.perf_counter()
    scores = Phi @ phi_q                           # (N_TRAIN,) — vectorized dot product
    t_q = time.perf_counter() - t0
    query_times.append(t_q)

t_query_mean = np.mean(query_times) * 1000         # ms
print(f"[4] Attribution score:     {t_query_mean:.2f}ms  (mean over {N_TEST} queries)")
results["attribution_score_ms"] = round(t_query_mean, 3)

# Keep last scores for top-k
# ── 5. Memory ─────────────────────────────────────────────────────────────────
mem_gb = Phi.nbytes / 1e9
print(f"[5] Memory (Phi matrix):   {mem_gb:.3f} GB")
results["memory_phi_gb"] = round(mem_gb, 4)

# ── 6. Top-k retrieval ───────────────────────────────────────────────────────
topk_times = []
for i in range(N_TEST):
    phi_q  = Phi_test[i]
    scores = Phi @ phi_q
    t0 = time.perf_counter()
    top_idx = np.argsort(scores)[-TOP_K:]
    t_k = time.perf_counter() - t0
    topk_times.append(t_k)

t_topk_mean = np.mean(topk_times) * 1000          # ms
print(f"[6] Top-{TOP_K} retrieval:      {t_topk_mean:.2f}ms  (mean over {N_TEST} queries)")
results["topk_retrieval_ms"] = round(t_topk_mean, 3)

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f"{'Method':<30} {'Value':>10}  {'Unit'}")
print("-" * 50)
print(f"{'Data generation':<30} {t_data:>10.2f}  s")
print(f"{'Gradient computation':<30} {t_grad:>10.2f}  s")
print(f"{'GradientStore build':<30} {t_store:>10.2f}  s")
print(f"{'Attribution score (1 query)':<30} {t_query_mean:>10.2f}  ms")
print(f"{'Top-10 retrieval':<30} {t_topk_mean:>10.2f}  ms")
print(f"{'Memory (Phi matrix)':<30} {mem_gb:>10.3f}  GB")

# ── Save results ──────────────────────────────────────────────────────────────
results_meta = {
    "experiment": "exp18_million_scale",
    "config": {
        "N_TRAIN": N_TRAIN,
        "D_FEAT": D_FEAT,
        "PROJ_DIM": PROJ_DIM,
        "SEED": SEED,
    },
    **results,
}

out_path = REPO_ROOT / "results" / "exp18_million_scale.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results_meta, f, indent=2)
print(f"\nResults saved to {out_path}")
