"""Experiment 14b: PyTorch MLP on harder MNIST binary (digits 4 vs 9).

Fixes exp14's trivial dataset (0 vs 1 → 100% accuracy, low LDS signal).
Digits 4 vs 9 is the hardest MNIST pair (~95% accuracy), giving meaningful
margin variation across retraining subsets → stronger LDS signal.

Changes vs exp14:
  - Digits: (4, 9) instead of (0, 1)
  - N_SUBSETS: 500 (consistent with all other benchmarks)
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
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import AttributionEngine

print("=" * 60)
print("Exp 14b: PyTorch MLP + Traceprop (MNIST 4 vs 9, 500 subsets)")
print("=" * 60)

SEED      = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_TRAIN   = 6000
N_TEST    = 500
N_SUBSETS = 500
PROJ_DIM  = 4096
LAMBDA    = 1e-3
EPOCHS    = 20
LR        = 1e-3
BATCH     = 128
DIGITS    = (4, 9)   # hardest MNIST pair

# ── Step 1: Load MNIST (digits 4 vs 9) ───────────────────────────────────────
print(f"\nStep 1: Loading MNIST (digits {DIGITS[0]} vs {DIGITS[1]})...")
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = torchvision.datasets.MNIST("/tmp/mnist", train=True,
                                         download=True, transform=transform)
mnist_test  = torchvision.datasets.MNIST("/tmp/mnist", train=False,
                                         download=True, transform=transform)

def filter_binary(dataset, digits, n_max):
    idx = [i for i, (_, l) in enumerate(dataset) if l in digits][:n_max]
    X = torch.stack([dataset[i][0].flatten() for i in idx]).numpy()
    y = np.array([1.0 if dataset[i][1] == digits[1] else 0.0 for i in idx],
                 dtype=np.float32)
    return X.astype(np.float32), y

X_tr, y_tr = filter_binary(mnist_train, DIGITS, N_TRAIN)
X_te, y_te = filter_binary(mnist_test,  DIGITS, N_TEST)
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
print(f"  Class balance train: {y_tr.mean():.3f}  test: {y_te.mean():.3f}")

# ── Step 2: Define MLP (no BatchNorm) ────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_mlp(X, y, epochs=EPOCHS, lr=LR, batch=BATCH, seed=SEED):
    torch.manual_seed(seed)
    model  = MLP()
    opt    = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X); yt = torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model

def mlp_margin(model, X, y):
    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy()
    return (logits * (2 * y - 1)).astype(np.float32)

def mlp_accuracy(model, X, y):
    with torch.no_grad():
        preds = (torch.sigmoid(model(torch.tensor(X))) > 0.5).numpy().astype(float)
    return (preds == y).mean()

# ── Step 3: Train MLP + log per-sample last-layer gradients ──────────────────
print("\nStep 2: Training MLP and logging per-sample gradients...")
t0    = time.perf_counter()
model = train_mlp(X_tr, y_tr)
acc   = mlp_accuracy(model, X_te, y_te)
print(f"  MLP test accuracy: {acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

print("  Computing per-sample gradients and storing provenance...")
t0 = time.perf_counter()
store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)

Xt_all = torch.tensor(X_tr)
yt_all = torch.tensor(y_tr)

with torch.no_grad():
    hidden = torch.relu(model.net[0](Xt_all))
    logits = model.net[2](hidden).squeeze(1)
    probs  = torch.sigmoid(logits)
    errors = probs - yt_all
    G_tr = (errors[:, None] * hidden).numpy().astype(np.float32)

for i in range(len(X_tr)):
    store.log_gradient(
        G_tr[i],
        sample_index=i,
        source_id="mnist_train",
        metadata={"pixel_source": f"mnist_row_{i}", "label": int(y_tr[i]),
                  "digit": DIGITS[1] if y_tr[i] == 1 else DIGITS[0]}
    )
print(f"  Stored {len(store)} gradient entries  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: Build attribution engine ─────────────────────────────────────────
engine = AttributionEngine(store, estimator="trak", lambda_factor=LAMBDA)

# ── Step 5: Retrain on subsets ────────────────────────────────────────────────
print(f"\nStep 3: Retraining MLP on {N_SUBSETS} subsets...")
t0 = time.perf_counter()
subset_masks   = np.zeros((N_SUBSETS, N_TRAIN), dtype=np.float32)
subset_margins = np.zeros((N_SUBSETS, N_TEST),  dtype=np.float32)

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < 0.7
    subset_masks[s] = mask.astype(np.float32)
    m_s = train_mlp(X_tr[mask], y_tr[mask], seed=SEED + s)
    subset_margins[s] = mlp_margin(m_s, X_te, y_te)
    if s % 100 == 0:
        print(f"  {s}/{N_SUBSETS}  ({time.perf_counter()-t0:.0f}s)")

retrain_time = time.perf_counter() - t0
print(f"  Done in {retrain_time:.1f}s")

# ── Step 6: Batch attribution ─────────────────────────────────────────────────
print(f"\nStep 4: Batch attribution ({os.cpu_count()} cores)...")
with torch.no_grad():
    hidden_te = torch.relu(model.net[0](torch.tensor(X_te)))
    logits_te = model.net[2](hidden_te).squeeze(1)
    probs_te  = torch.sigmoid(logits_te)
    errors_te = probs_te - torch.tensor(y_te)
    G_te = (errors_te[:, None] * hidden_te).numpy().astype(np.float32)

t0 = time.perf_counter()
inf_matrix = engine.attribute_scores_batch(G_te)
t_attr = time.perf_counter() - t0
print(f"  Attribution done in {t_attr:.2f}s")

rand_inf = np.random.randn(N_TEST, N_TRAIN).astype(np.float32)

# ── Step 7: LDS ───────────────────────────────────────────────────────────────
print("\nStep 5: Computing LDS...")

def compute_lds(inf_matrix):
    predicted = inf_matrix @ subset_masks.T
    scores = [spearmanr(predicted[i], subset_margins[:, i]).statistic
              for i in range(len(predicted))]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)), float(np.std(scores))

lds_mean, lds_std = compute_lds(inf_matrix)
rld_mean, rld_std = compute_lds(rand_inf)

# ── Step 8: Provenance query demo ────────────────────────────────────────────
print(f"\nStep 6: Provenance query — most influential training pixels for test[0]...")
scores_0 = inf_matrix[0]
top5 = np.argsort(scores_0)[::-1][:5]
digit_name = {DIGITS[0]: str(DIGITS[0]), DIGITS[1]: str(DIGITS[1])}
print(f"  Test sample 0 (digit {digit_name[DIGITS[1] if y_te[0]==1 else DIGITS[0]]}):")
for rank, idx in enumerate(top5):
    entry = store.get_entry_by_index(idx)
    print(f"    #{rank+1}: training row {idx} "
          f"(label={entry.metadata.get('digit','?')}, "
          f"influence={scores_0[idx]:.4f})")

print()
print("=" * 60)
print(f"{'Method':<40} {'LDS mean':>9} {'± std':>8} {'Time':>7}")
print("-" * 60)
print(f"{'TP-LL + TRAK (PyTorch MLP, 4 vs 9)':<40} {lds_mean:>9.4f} {lds_std:>8.4f} {t_attr:>6.2f}s")
print(f"{'Random baseline':<40} {rld_mean:>9.4f} {rld_std:>8.4f}    ---")
print("=" * 60)
print(f"\nMLP test accuracy: {acc:.4f}")
print(f"MLP retrain time ({N_SUBSETS} subsets): {retrain_time:.1f}s")

results = {
    "experiment": "exp14b_pytorch_mnist_harder",
    "dataset": f"mnist_binary_{DIGITS[0]}vs{DIGITS[1]}",
    "model": "mlp_no_batchnorm_784_256_1",
    "n_train": N_TRAIN, "n_test": N_TEST, "n_subsets": N_SUBSETS,
    "proj_dim": PROJ_DIM, "epochs": EPOCHS, "lr": LR,
    "test_accuracy": round(float(acc), 4),
    "tp_ll_trak": {"lds_mean": round(lds_mean, 4), "lds_std": round(lds_std, 4),
                   "time_s": round(t_attr, 3)},
    "random":     {"lds_mean": round(rld_mean, 4), "lds_std": round(rld_std, 4)},
    "retrain_time_s": round(retrain_time, 1),
    "notes": "Harder MNIST pair (4 vs 9). 500 subsets. Per-sample last-layer gradients. No BatchNorm.",
}

with open("results/exp14b_pytorch_mnist_harder.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp14b_pytorch_mnist_harder.json")
