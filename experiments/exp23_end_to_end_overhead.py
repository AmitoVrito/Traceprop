"""End-to-end training overhead — addresses VLDB reviewer concern #2.

The microbenchmark in Table 4 shows op-mode overhead of 4x at 1K elements
and 2x at 10K, converging to baseline at 10^6. Real training loops are
dominated by many small ops per step (linear matmul, activation, gradient,
update), not single 10^6-element matmuls. We measure total wall-clock for
a real training task in each regime.

Workload: logistic regression on Adult Income (n=6000), trained by manual
mini-batch SGD for 20 epochs at batch_size=64.
  - per step: 64x100 @ 100 matmul (small-op regime, ~6400 elements)
  - per epoch: 94 steps
  - total ops per training run: 1880 small matmuls

Two configurations:
  baseline:  raw NumPy ops
  traceprop: full GradientStore logging (production attribution config)
"""
import json, os, sys, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traceprop.attribution.gradient_store import GradientStore

np.random.seed(42)

print("Loading Adult Income from OpenML...")
ds = fetch_openml("adult", version=2, as_frame=True, parser="auto")
import pandas as pd
X_df = pd.get_dummies(ds.data, drop_first=True).astype(np.float64)
y = (ds.target == ">50K").astype(np.float32).to_numpy()

rng = np.random.default_rng(0)
idx = rng.choice(len(X_df), 6000, replace=False)
X = X_df.to_numpy()[idx]
y = y[idx]
X = StandardScaler().fit_transform(X).astype(np.float32)
print(f"  shape: {X.shape}, class balance: {y.mean():.3f}")

N = len(X); D = X.shape[1]
N_EPOCHS = 20; BATCH = 64
N_TRIALS = 5

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_baseline():
    w = np.zeros(D, dtype=np.float32); b_ = np.float32(0.0)
    lr = 0.01
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X[idx], y[idx]
            p = sigmoid(xb @ w + b_)
            err = p - yb
            g = (err[:, None] * xb).mean(axis=0)
            gb = err.mean()
            w  -= lr * g; b_ -= lr * gb
    return time.perf_counter() - t0

def train_traceprop_batch():
    """Batched per-step gradient logging (no per-sample) — the cheaper
    Traceprop-BM attribution config from Sec 5.1, suitable for
    inexpensive lineage where per-sample attribution is not required."""
    w = np.zeros(D, dtype=np.float32); b_ = np.float32(0.0)
    store = GradientStore(proj_dim=512, seed=42)
    lr = 0.01
    step = 0
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X[idx], y[idx]
            p = sigmoid(xb @ w + b_)
            err = p - yb
            g = (err[:, None] * xb).mean(axis=0)
            gb = err.mean()
            w  -= lr * g; b_ -= lr * gb
            # One batch-mean gradient per step
            store.log_gradient(g.astype(np.float32), sample_index=step,
                               source_id="adult")
            step += 1
    return time.perf_counter() - t0


def train_traceprop_persample_batched():
    """Per-sample last-layer gradients logged via the vectorised
    log_batch() API — same attribution semantics as Traceprop-LL but the
    Python per-sample loop is replaced by a single BLAS matmul + array
    assignment per step."""
    w = np.zeros(D, dtype=np.float32); b_ = np.float32(0.0)
    store = GradientStore(proj_dim=512, seed=42)
    lr = 0.01
    sample_idx_counter = 0
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X[idx], y[idx]
            p = sigmoid(xb @ w + b_)
            err = p - yb
            g = (err[:, None] * xb).mean(axis=0)
            gb = err.mean()
            w  -= lr * g; b_ -= lr * gb
            # Vectorised: one (batch_size, D) array, one BLAS matmul.
            per_sample_grads = (err[:, None] * xb).astype(np.float32)
            store.log_batch(per_sample_grads, source_id="adult",
                            sample_index_offset=sample_idx_counter)
            sample_idx_counter += len(idx)
    return time.perf_counter() - t0


def train_traceprop_persample():
    """Per-sample last-layer gradients logged into a GradientStore — the
    Traceprop-LL production attribution config used for exp14b LDS=0.193.
    This is the strongest attribution-quality config and the most
    expensive at the per-op level."""
    w = np.zeros(D, dtype=np.float32); b_ = np.float32(0.0)
    store = GradientStore(proj_dim=512, seed=42)
    lr = 0.01
    sample_idx_counter = 0
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = X[idx], y[idx]
            p = sigmoid(xb @ w + b_)
            err = p - yb
            g = (err[:, None] * xb).mean(axis=0)
            gb = err.mean()
            w  -= lr * g; b_ -= lr * gb
            for j in range(len(idx)):
                g_j = (err[j] * xb[j]).astype(np.float32)
                store.log_gradient(g_j, sample_index=sample_idx_counter + j,
                                   source_id="adult")
            sample_idx_counter += len(idx)
    return time.perf_counter() - t0

print(f"\nRunning {N_TRIALS} trials each ({N_EPOCHS} epochs, batch={BATCH}, "
      f"94 steps/epoch, {N_EPOCHS*94} ops/run)...")
baseline_times, bm_times, ll_times, llb_times = [], [], [], []
for trial in range(N_TRIALS):
    bt = train_baseline()
    bm = train_traceprop_batch()
    llb = train_traceprop_persample_batched()
    ll = train_traceprop_persample()
    baseline_times.append(bt); bm_times.append(bm)
    ll_times.append(ll); llb_times.append(llb)
    print(f"  trial {trial+1}: baseline={bt:.2f}s  TP-BM={bm:.2f}s ({bm/bt:.2f}x)  "
          f"TP-LL-batched={llb:.2f}s ({llb/bt:.2f}x)  "
          f"TP-LL-naive={ll:.2f}s ({ll/bt:.2f}x)")

# Replace single traceprop ratio with split BM/LL ratios
traceprop_times = ll_times    # keep LL for backward compat in summary

def stats(times):
    return float(np.mean(times)), float(np.std(times))
b_mean, b_std = stats(baseline_times)
bm_mean, bm_std = stats(bm_times)
llb_mean, llb_std = stats(llb_times)
ll_mean, ll_std = stats(ll_times)
bm_ratios  = [t/b for t, b in zip(bm_times,  baseline_times)]
llb_ratios = [t/b for t, b in zip(llb_times, baseline_times)]
ll_ratios  = [t/b for t, b in zip(ll_times,  baseline_times)]

print()
print("=" * 78)
print(f"End-to-end training overhead (Adult Income, manual minibatch SGD)")
print(f"  baseline              : {b_mean:.3f} ± {b_std:.3f} s")
print(f"  Traceprop-BM          : {bm_mean:.3f} ± {bm_std:.3f} s  "
      f"({np.mean(bm_ratios):.2f}x ± {np.std(bm_ratios):.2f})")
print(f"  Traceprop-LL (batched): {llb_mean:.3f} ± {llb_std:.3f} s  "
      f"({np.mean(llb_ratios):.2f}x ± {np.std(llb_ratios):.2f})")
print(f"  Traceprop-LL (naive)  : {ll_mean:.3f} ± {ll_std:.3f} s  "
      f"({np.mean(ll_ratios):.2f}x ± {np.std(ll_ratios):.2f})")
print("=" * 78)

out = {
    "experiment": "exp23_end_to_end_overhead",
    "workload": ("Adult Income n=6000, logistic regression with manual "
                 "minibatch SGD, 20 epochs, batch_size=64 — 1880 small "
                 "matmul ops per training run, representative of the "
                 "small-op regime real training spends most time in"),
    "n_trials": N_TRIALS,
    "baseline_mean_s":     round(b_mean, 3),
    "baseline_std_s":      round(b_std,  3),
    "traceprop_bm_mean_s": round(bm_mean, 3),
    "traceprop_bm_std_s":  round(bm_std,  3),
    "traceprop_bm_overhead_ratio_mean": round(float(np.mean(bm_ratios)), 3),
    "traceprop_bm_overhead_ratio_std":  round(float(np.std(bm_ratios)),  3),
    "traceprop_ll_batched_mean_s": round(llb_mean, 3),
    "traceprop_ll_batched_std_s":  round(llb_std,  3),
    "traceprop_ll_batched_overhead_ratio_mean": round(float(np.mean(llb_ratios)), 3),
    "traceprop_ll_batched_overhead_ratio_std":  round(float(np.std(llb_ratios)),  3),
    "traceprop_ll_naive_mean_s": round(ll_mean, 3),
    "traceprop_ll_naive_std_s":  round(ll_std,  3),
    "traceprop_ll_naive_overhead_ratio_mean": round(float(np.mean(ll_ratios)), 3),
    "traceprop_ll_naive_overhead_ratio_std":  round(float(np.std(ll_ratios)),  3),
    "note": ("Real end-to-end overhead under two attribution configs. "
             "Traceprop-BM logs one batch-mean gradient per step "
             "(cheap, lower LDS quality). Traceprop-LL logs per-sample "
             "last-layer gradients (production attribution quality used "
             "for exp14b LDS=0.193, exp4c LDS=0.622). The BM ratio is "
             "what a production lineage-only deployment would see; the "
             "LL ratio is what a system providing per-sample attribution "
             "must pay in CPU."),
}
os.makedirs("results", exist_ok=True)
with open("results/exp23_end_to_end_overhead.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to results/exp23_end_to_end_overhead.json")
