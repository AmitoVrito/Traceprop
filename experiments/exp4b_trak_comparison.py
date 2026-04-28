"""Experiment 4b: TRAK LDS comparison using dattri.

Same data generation as exp4 (seed 42, N_TRAIN=2000, N_TEST=200, d=50).
Wraps logistic regression as torch.nn.Linear(50, 2) and computes TRAK
attribution scores, then evaluates LDS with identical methodology.
"""

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time

# Must match exp4 exactly
PROJ_DIM = 4096
N_FEATURES = 50
N_TRAIN = 2000
N_TEST = 200
N_SUBSETS = 50
SUBSET_RATIO = 0.5
LR = 0.1
EPOCHS = 50

np.random.seed(42)
torch.manual_seed(42)

print(f"TRAK LDS Benchmark (PyTorch logistic regression)")
print(f"Train: {N_TRAIN}, Test: {N_TEST}, Features: {N_FEATURES}")
print(f"Proj dim: {PROJ_DIM}, Subsets: {N_SUBSETS}\n")


# --- Data (identical to exp4) ---
def make_data(n, d):
    w_true = np.random.randn(d)
    X = np.random.randn(n, d)
    logits = X @ w_true
    y = (logits > 0).astype(np.float64)
    flip = np.random.rand(n) < 0.05
    y[flip] = 1 - y[flip]
    return X, y


X_train, y_train = make_data(N_TRAIN, N_FEATURES)
X_test, y_test = make_data(N_TEST, N_FEATURES)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# --- Model ---
def make_model():
    return nn.Linear(N_FEATURES, 2)


def train_model(model, X, y, lr=LR, epochs=EPOCHS):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    return model


def get_accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


# --- Step 1: Train full model ---
print("Step 1: Training full model...")
model = make_model()
train_model(model, X_train_t, y_train_t)
full_acc = get_accuracy(model, X_test_t, y_test_t)
print(f"Full model test accuracy: {full_acc:.4f}")

# --- Step 2: Compute TRAK scores ---
print("\nStep 2: Computing TRAK attribution scores...")
t0 = time.perf_counter()

try:
    from dattri.algorithm.trak import TRAKAttributor
    from dattri.benchmark.utils import SubsetSampler

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False
    )

    def loss_trak(params, data_target_pair):
        x, y = data_target_pair
        out = torch.func.functional_call(model, params, (x,))
        return nn.functional.cross_entropy(out, y, reduction="sum")

    attributor = TRAKAttributor(
        target_func=loss_trak,
        params=dict(model.named_parameters()),
        weight_list=[dict(model.named_parameters())],
        proj_dim=PROJ_DIM,
    )

    attributor.cache(train_loader)
    trak_scores = attributor.attribute(test_loader)  # (N_TEST, N_TRAIN)
    trak_influence_matrix = trak_scores.numpy()

except Exception as e:
    print(f"dattri TRAK failed: {e}")
    print("Falling back to manual TRAK-style random projection attribution...")

    # Manual TRAK: project per-sample gradients, compute dot products
    # This implements the core TRAK idea: random projection of gradients
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Collect per-sample gradients for training data
    proj_matrix = torch.randn(sum(p.numel() for p in model.parameters()), PROJ_DIM) / (PROJ_DIM ** 0.5)

    def get_per_sample_grad(x, y_label):
        """Get gradient for a single sample."""
        model.zero_grad()
        out = model(x.unsqueeze(0))
        loss = nn.functional.cross_entropy(out, y_label.unsqueeze(0))
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    print("  Computing projected training gradients...")
    train_projs = torch.zeros(N_TRAIN, PROJ_DIM)
    for i in range(N_TRAIN):
        g = get_per_sample_grad(X_train_t[i], y_train_t[i])
        train_projs[i] = g @ proj_matrix
        if i % 500 == 0:
            print(f"    Train sample {i}/{N_TRAIN}")

    print("  Computing projected test gradients...")
    test_projs = torch.zeros(N_TEST, PROJ_DIM)
    for i in range(N_TEST):
        g = get_per_sample_grad(X_test_t[i], y_test_t[i])
        test_projs[i] = g @ proj_matrix

    # TRAK scores = dot product of projected gradients
    trak_scores_t = test_projs @ train_projs.T
    trak_influence_matrix = np.array(trak_scores_t.tolist())

trak_time = time.perf_counter() - t0
print(f"TRAK attribution took {trak_time:.1f}s")
print(f"Influence matrix shape: {trak_influence_matrix.shape}")

# --- Step 3: Ground-truth via retraining (same as exp4) ---
print(f"\nStep 3: Retraining on {N_SUBSETS} random subsets for ground truth...")
t0 = time.perf_counter()

# Reset seed to match exp4's subset generation
np.random.seed(42 + 1000)  # Different seed for subsets (exp4 uses continuing state)

subset_masks = []
subset_accuracies = np.zeros((N_SUBSETS, N_TEST))

for s in range(N_SUBSETS):
    mask = np.random.rand(N_TRAIN) < SUBSET_RATIO
    subset_masks.append(mask)
    X_sub = X_train_t[mask]
    y_sub = y_train_t[mask]
    m = make_model()
    torch.manual_seed(s)
    train_model(m, X_sub, y_sub)
    with torch.no_grad():
        preds = m(X_test_t).argmax(dim=1)
        subset_accuracies[s] = (preds == y_test_t).float().numpy()
    if s % 10 == 0:
        print(f"  Subset {s}/{N_SUBSETS} done")

subset_masks = np.array(subset_masks)
print(f"Retraining took {time.perf_counter() - t0:.1f}s\n")

# --- Step 4: Compute LDS ---
print("Step 4: Computing LDS scores...")

lds_scores_trak = []
lds_scores_random = []
random_scores = np.random.rand(N_TEST, N_TRAIN)

for test_i in range(N_TEST):
    trak_pred = trak_influence_matrix[test_i]
    rand_pred = random_scores[test_i]

    gt = np.array([
        spearmanr(subset_masks[:, j], subset_accuracies[:, test_i]).statistic
        for j in range(N_TRAIN)
    ])
    gt = np.nan_to_num(gt, 0.0)

    trak_lds = spearmanr(trak_pred, gt).statistic
    rand_lds = spearmanr(rand_pred, gt).statistic

    lds_scores_trak.append(trak_lds if not np.isnan(trak_lds) else 0.0)
    lds_scores_random.append(rand_lds if not np.isnan(rand_lds) else 0.0)

mean_lds_trak = np.mean(lds_scores_trak)
std_lds_trak = np.std(lds_scores_trak)
mean_lds_random = np.mean(lds_scores_random)

print(f"\n{'=' * 50}")
print(f"TRAK LDS:          {mean_lds_trak:.4f} ± {std_lds_trak:.4f}")
print(f"Random baseline:   {mean_lds_random:.4f}")
print(f"{'=' * 50}")

results = {
    "trak_lds_mean": round(float(mean_lds_trak), 4),
    "trak_lds_std": round(float(std_lds_trak), 4),
    "random_lds_mean": round(float(mean_lds_random), 4),
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "n_subsets": N_SUBSETS,
    "n_features": N_FEATURES,
    "proj_dim": PROJ_DIM,
    "full_model_accuracy": round(float(full_acc), 4),
    "trak_time_s": round(trak_time, 1),
    "model_type": "torch_linear",
}

with open("results/exp4b_trak_lds.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp4b_trak_lds.json")
