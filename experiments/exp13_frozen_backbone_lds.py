"""Experiment 13: Frozen backbone + linear probe LDS (CIFAR-2).

New framing for deep vision: instead of training ResNet end-to-end
(where BatchNorm corrupts per-sample gradients), use a pre-trained
ResNet18 (ImageNet, frozen) as a feature extractor. The linear probe
on top IS logistic regression — Traceprop-LL is EXACT here.

This is the dominant production pattern:
  pre-trained backbone (frozen) → linear head (trained on your data)

Expected LDS: high (~0.7-0.9), same as tabular.
No GPU needed for attribution. Demonstrates Traceprop works on vision.

Pipeline:
  1. Load CIFAR-10, binary: airplane (0) vs automobile (1)
  2. Extract 512-dim features via frozen ResNet18 (ImageNet pre-trained)
  3. Train logistic regression linear probe
  4. Compute Traceprop-LL + TRAK estimator LDS (same as exp9)
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("Exp 13: Frozen ResNet18 + Linear Probe LDS (CIFAR-2)")
print("  Pre-trained backbone (frozen) → logistic regression head")
print(f"  {os.cpu_count()} CPU cores")
print("=" * 65)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN   = 10_000   # all CIFAR-2 train (5K airplane + 5K automobile)
N_TEST    = 500
N_SUBSETS = 500
PROJ_DIM  = 4096
LAMBDA    = 1e-3
C_LR      = 10.0
MAX_ITER  = 200
SEED      = 42
np.random.seed(SEED)

# ── Step 1: Extract features via frozen ResNet18 ──────────────────────────────
print("\nStep 1: Extracting ResNet18 features (frozen, ImageNet pre-trained)...")
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

CIFAR2_CLASSES = {0: "airplane", 1: "automobile"}

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True, transform=transform
)

# Filter to airplane (0) and automobile (1)
def filter_binary(dataset):
    indices = [i for i, (_, label) in enumerate(dataset) if label in (0, 1)]
    return indices

tr_indices = filter_binary(train_dataset)
te_indices = filter_binary(test_dataset)
print(f"  CIFAR-2 train: {len(tr_indices)}, test: {len(te_indices)}")

# Load pre-trained ResNet18, remove final FC layer
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = torch.nn.Identity()
backbone.eval()

@torch.no_grad()
def extract_features(dataset, indices, batch_size=256):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    feats, labels = [], []
    for imgs, lbls in loader:
        feats.append(backbone(imgs).numpy())
        labels.append(lbls.numpy())
    return np.vstack(feats), np.concatenate(labels)

t0 = time.perf_counter()
print("  Extracting train features...")
X_tr_raw, y_tr_raw = extract_features(train_dataset, tr_indices)
print("  Extracting test features...")
X_te_raw, y_te_raw = extract_features(test_dataset, te_indices)
feat_time = time.perf_counter() - t0
print(f"  Features: train {X_tr_raw.shape}, test {X_te_raw.shape}  [{feat_time:.1f}s]")

# Subsample train to N_TRAIN, test to N_TEST
rng = np.random.default_rng(SEED)
tr_idx = rng.permutation(len(X_tr_raw))[:N_TRAIN]
te_idx = rng.permutation(len(X_te_raw))[:N_TEST]
X_tr, y_tr = X_tr_raw[tr_idx].astype(np.float32), y_tr_raw[tr_idx].astype(np.float32)
X_te, y_te = X_te_raw[te_idx].astype(np.float32), y_te_raw[te_idx].astype(np.float32)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)
print(f"  After subsample: train {X_tr.shape}, test {X_te.shape}")

# ── Step 2: Train linear probe ────────────────────────────────────────────────
print("\nStep 2: Training linear probe (logistic regression on features)...")
t0 = time.perf_counter()
clf = LogisticRegression(C=C_LR, solver="lbfgs", max_iter=MAX_ITER,
                         random_state=SEED)
clf.fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)
print(f"  Linear probe test accuracy: {acc:.4f}  [{time.perf_counter()-t0:.1f}s]")

# ── Step 3: Build gradient store ──────────────────────────────────────────────
print("\nStep 3: Building gradient store...")
from traceprop.attribution.gradient_store import GradientStore
from traceprop.attribution.attribution_engine import AttributionEngine

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def lr_margin(clf, X, y):
    return (clf.decision_function(X) * (2 * y - 1)).astype(np.float32)

t0 = time.perf_counter()
store = GradientStore(proj_dim=PROJ_DIM, seed=SEED)
coef, intercept = clf.coef_[0], clf.intercept_[0]
err = sigmoid(X_tr @ coef + intercept) - y_tr
G_tr = (err[:, None] * X_tr).astype(np.float32)
for i in range(len(X_tr)):
    store.log_gradient(G_tr[i], sample_index=i, source_id="cifar2_features")
print(f"  {len(store)} entries stored  [{time.perf_counter()-t0:.1f}s]")

# ── Step 4: Retrain on 500 subsets ───────────────────────────────────────────
print(f"\nStep 4: Retraining on {N_SUBSETS} subsets...")
t0 = time.perf_counter()
subset_masks   = np.zeros((N_SUBSETS, N_TRAIN), dtype=np.float32)
subset_margins = np.zeros((N_SUBSETS, N_TEST),  dtype=np.float32)

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < 0.7
    subset_masks[s] = mask.astype(np.float32)
    clf_s = LogisticRegression(C=C_LR, solver="lbfgs", max_iter=MAX_ITER,
                               random_state=SEED)
    clf_s.fit(X_tr[mask], y_tr[mask])
    subset_margins[s] = lr_margin(clf_s, X_te, y_te)
    if s % 100 == 0:
        print(f"  {s}/{N_SUBSETS}  ({time.perf_counter()-t0:.0f}s)")

retrain_time = time.perf_counter() - t0
print(f"  Done in {retrain_time:.1f}s")

# ── Step 5: Batch attribution ─────────────────────────────────────────────────
print(f"\nStep 5: Batch attribution ({os.cpu_count()} cores)...")
engine_dot  = AttributionEngine(store, estimator="dot")
engine_trak = AttributionEngine(store, estimator="trak", lambda_factor=LAMBDA)

err_te = sigmoid(X_te @ coef + intercept) - y_te
G_te   = (err_te[:, None] * X_te).astype(np.float32)

t0 = time.perf_counter()
inf_dot = engine_dot.attribute_scores_batch(G_te)
t_dot = time.perf_counter() - t0

t0 = time.perf_counter()
inf_trak = engine_trak.attribute_scores_batch(G_te)
t_trak = time.perf_counter() - t0

rand_inf = np.random.randn(N_TEST, N_TRAIN).astype(np.float32)
print(f"  dot: {t_dot:.1f}s, trak: {t_trak:.1f}s")

# ── Step 6: LDS ───────────────────────────────────────────────────────────────
print("\nStep 6: Computing LDS...")

def compute_lds(inf_matrix):
    predicted = inf_matrix @ subset_masks.T
    scores = [spearmanr(predicted[i], subset_margins[:, i]).statistic
              for i in range(len(predicted))]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)), float(np.std(scores))

lds_dot_mean,  lds_dot_std  = compute_lds(inf_dot)
lds_trak_mean, lds_trak_std = compute_lds(inf_trak)
lds_rand_mean, lds_rand_std = compute_lds(rand_inf)

print()
print("=" * 65)
print(f"{'Variant':<40} {'LDS mean':>8} {'± std':>8} {'Time':>6}")
print("-" * 65)
print(f"{'TP-LL dot (frozen ResNet18 features)':<40} {lds_dot_mean:>8.4f} {lds_dot_std:>8.4f} {t_dot:>5.1f}s")
print(f"{'TP-LL + TRAK est. (frozen ResNet18)':<40} {lds_trak_mean:>8.4f} {lds_trak_std:>8.4f} {t_trak:>5.1f}s")
print(f"{'Random baseline':<40} {lds_rand_mean:>8.4f} {lds_rand_std:>8.4f}   ---")
print("=" * 65)
print()
print(f"Linear probe accuracy: {acc:.4f}")
print(f"Feature extraction: {feat_time:.1f}s (one-time cost, backbone frozen)")
print()
print("Key result: Traceprop-LL is EXACT for frozen backbone + linear probe.")
print("This is the dominant production pattern for vision in regulated industries.")

results = {
    "experiment": "exp13_frozen_backbone_lds",
    "dataset": "cifar2_airplane_vs_automobile",
    "backbone": "resnet18_imagenet_frozen",
    "feature_dim": int(X_tr.shape[1]),
    "n_train": N_TRAIN, "n_test": N_TEST, "n_subsets": N_SUBSETS,
    "proj_dim": PROJ_DIM, "C": C_LR, "max_iter": MAX_ITER,
    "linear_probe_accuracy": round(float(acc), 4),
    "feature_extraction_time_s": round(feat_time, 1),
    "tp_ll_dot":  {"lds_mean": round(lds_dot_mean, 4),  "lds_std": round(lds_dot_std, 4),  "time_s": round(t_dot, 2)},
    "tp_ll_trak": {"lds_mean": round(lds_trak_mean, 4), "lds_std": round(lds_trak_std, 4), "time_s": round(t_trak, 2)},
    "random":     {"lds_mean": round(lds_rand_mean, 4), "lds_std": round(lds_rand_std, 4)},
    "retrain_time_s": round(retrain_time, 1),
    "notes": "Frozen ResNet18 backbone (ImageNet pre-trained). Linear probe = logistic regression. Traceprop-LL exact for this model class.",
}

with open("results/exp13_frozen_backbone_lds.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp13_frozen_backbone_lds.json")
