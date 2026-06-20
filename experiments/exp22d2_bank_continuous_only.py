"""S3 ablation: does Bank Marketing fail because of mixed one-hot encodings
of similar-scale categoricals? Test by restricting features to numeric-only
columns within each source group, eliminating the one-hot encoding."""

import json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_sab_benchmark import (
    baseline_random, baseline_grad_argmax, ss_attrib_block, metrics,
)

N_TEST_PER_SEED = 200
N_SAMPLE = 20_000
SEEDS_VALIDATE = list(range(10, 30))

print("Loading bank-marketing from OpenML...")
ds = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
X_df, y_raw = ds.data, ds.target
y = (y_raw == "2").astype(np.float32).to_numpy()

# Continuous-only subset of each source group (drop categoricals/one-hots)
GROUPS_CONT = {
    "bank_client":   ["V1"],            # only age is continuous
    "last_contact":  ["V10", "V11"],    # day and duration
    "campaign_econ": ["V12", "V13", "V14"],
}

# Keep only continuous columns
keep = sorted(set(sum(GROUPS_CONT.values(), [])))
X_sub = X_df[keep].copy()
for c in X_sub.columns:
    X_sub[c] = pd.to_numeric(X_sub[c].astype(str), errors="coerce")
X = X_sub.to_numpy(dtype=np.float64)

# Median impute NaN — for empty/all-NaN columns fall back to 0
col_med = np.nanmedian(X, axis=0)
col_med = np.where(np.isnan(col_med), 0.0, col_med)
X = np.where(np.isnan(X), col_med, X)

source_cols = {g: [keep.index(c) for c in v] for g, v in GROUPS_CONT.items()}
print(f"  n={len(X):,}  d={X.shape[1]}  groups: "
      f"{dict((g, len(c)) for g, c in source_cols.items())}")

# Audit-realistic primary-source label (continuous-only)
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
ref = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                         random_state=42).fit(X_std, y)
ref_coef = ref.coef_[0]
contribs = np.stack([
    np.abs(X_std[:, source_cols[g]] @ ref_coef[source_cols[g]])
    for g in GROUPS_CONT], axis=1)
src_idx = contribs.argmax(axis=1)
src_names = list(GROUPS_CONT)
src_all = [src_names[i] for i in src_idx]

from collections import Counter
print(f"  Source distribution: {dict(Counter(src_all))}")

def gen_seed(seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(N_SAMPLE, len(X)), replace=False)
    X_s, y_s, src_s = X[idx], y[idx], [src_all[i] for i in idx]
    te_n = min(N_TEST_PER_SEED, len(X_s)//5)
    return (X_s[te_n:], y_s[te_n:], src_s[te_n:],
            X_s[:te_n], y_s[:te_n], src_s[:te_n],
            source_cols, src_names)

per_seed = []
for seed in SEEDS_VALIDATE:
    Xtr, ytr, stt, Xte, yte, ste, sc, sn = gen_seed(seed)
    gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    ss_p   = ss_attrib_block(    Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    per_seed.append({
        "seed": seed,
        "gmag": metrics(gmag_p, ste)["macro_p1"],
        "ss":   metrics(ss_p,   ste)["macro_p1"],
    })

ss_vals   = np.array([r["ss"]   for r in per_seed])
gmag_vals = np.array([r["gmag"] for r in per_seed])
diff      = ss_vals - gmag_vals
t_test    = stats.ttest_rel(ss_vals, gmag_vals)

print()
print("=" * 65)
print(f"Bank Marketing CONTINUOUS-ONLY held-out (n={len(SEEDS_VALIDATE)} seeds)")
print("-" * 65)
print(f"  gmag      {gmag_vals.mean():.4f} ± {gmag_vals.std():.4f}")
print(f"  ss        {ss_vals.mean():.4f} ± {ss_vals.std():.4f}")
print(f"  Paired diff = {diff.mean():+.4f} ± {diff.std():.4f}")
print(f"  SS wins {(diff > 0).sum()}/{len(SEEDS_VALIDATE)} seeds")
print(f"  Paired t = {t_test.statistic:.3f}, p = {t_test.pvalue:.4f}")
print(f"  Original Bank (with one-hots): SS=0.481, gmag=0.503, diff=-2.2pp")
print("=" * 65)

out = {
    "experiment": "exp22d2_bank_continuous_only",
    "dataset":    "bank-marketing-continuous-only-features",
    "source_groups": {g: len(c) for g, c in source_cols.items()},
    "n_seeds":      len(SEEDS_VALIDATE),
    "per_seed":     per_seed,
    "ss_mean":      round(float(ss_vals.mean()), 4),
    "ss_std":       round(float(ss_vals.std()),  4),
    "gmag_mean":    round(float(gmag_vals.mean()), 4),
    "gmag_std":     round(float(gmag_vals.std()),  4),
    "mean_paired_diff": round(float(diff.mean()), 4),
    "paired_t":         round(float(t_test.statistic), 4),
    "paired_p":         round(float(t_test.pvalue),    6),
    "n_seeds_ss_wins":  int((diff > 0).sum()),
    "note": ("Ablation: Bank Marketing restricted to continuous numeric "
             "features only. If S3 hypothesis (mixed one-hot encodings "
             "of similar-scale categoricals are the failure cause) is "
             "right, SS should now beat gmag here."),
}
os.makedirs("results", exist_ok=True)
with open("results/exp22d2_bank_continuous_only.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22d2_bank_continuous_only.json")
