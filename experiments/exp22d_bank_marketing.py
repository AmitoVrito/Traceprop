"""Experiment 22d: SAB v1 generalisation check on Bank Marketing.

Reviewer concern R6 (v3): the +4.2 pp lift was measured on a single
real dataset (Home Credit). Without a second real-data replication a
reviewer can argue significance does not generalise.

We re-run the SAB v1 evaluation on the Bank Marketing UCI dataset
(fetched from OpenML, n=45,211), splitting the 16 features into 3
natural source groups by data origin:

  - bank_client: age, job, marital, education, default, housing, loan
                 (information collected from the client)
  - last_contact: contact, month, day_of_week, duration
                  (information about the current campaign contact)
  - campaign_econ: campaign, pdays, previous, poutcome, emp.var.rate,
                   cons.price.idx, cons.conf.idx, euribor3m, nr.employed
                   (campaign history + macroeconomic context)

Same protocol as exp22b: K = min(1000, n_train/20), 20 disjoint held-out
seeds, paired t-test SS vs gradmag.
"""

import json, os, sys, time
import numpy as np
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_sab_benchmark import (
    baseline_random, baseline_grad_argmax, ss_attrib_block, coef_oracle,
    metrics,
)

N_TEST_PER_SEED = 200
N_SAMPLE = 20_000
SEEDS_VALIDATE = list(range(10, 30))    # disjoint from exp22c sweep
print("Loading bank-marketing from OpenML (one-time)...")
ds = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
X_df, y_raw = ds.data, ds.target
y = (y_raw == "2").astype(np.float32).to_numpy()   # OpenML encodes "2" as positive

GROUPS = {
    "bank_client":   ["V1", "V2", "V3", "V4", "V5", "V6", "V7"],
    "last_contact":  ["V8", "V9", "V10", "V11"],
    "campaign_econ": ["V12", "V13", "V14", "V15"],
}

# One-hot encode while tracking origin
X_oh = pd.get_dummies(X_df, drop_first=True).astype(np.float64)
origin = []
for c in X_df.columns:
    if X_df[c].dtype == "object" or str(X_df[c].dtype) == "category":
        k = X_df[c].nunique() - 1
        origin.extend([c] * k)
    else:
        origin.append(c)

source_cols = {g: [] for g in GROUPS}
for j, col in enumerate(origin):
    for g, members in GROUPS.items():
        if col in members:
            source_cols[g].append(j); break
keep = sorted(sum(source_cols.values(), []))
j_map = {old: new for new, old in enumerate(keep)}
source_cols = {g: [j_map[j] for j in idx] for g, idx in source_cols.items()}
X = X_oh.to_numpy()[:, keep]
print(f"  n={len(X):,}  d={X.shape[1]}  "
      f"groups: {dict((g, len(c)) for g, c in source_cols.items())}")

# Audit-realistic primary-source label: argmax |coef[block] . x[block]|
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
ref = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                         random_state=42).fit(X_std, y)
ref_coef = ref.coef_[0]
contribs = np.stack([
    np.abs(X_std[:, source_cols[g]] @ ref_coef[source_cols[g]])
    for g in GROUPS], axis=1)
src_idx = contribs.argmax(axis=1)
src_names = list(GROUPS)
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

print(f"\nRunning {len(SEEDS_VALIDATE)} disjoint held-out seeds...")
t0 = time.perf_counter()
per_seed = []
for seed in SEEDS_VALIDATE:
    Xtr, ytr, stt, Xte, yte, ste, sc, sn = gen_seed(seed)
    rand_p = baseline_random(    Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    ss_p   = ss_attrib_block(    Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    cref_p = coef_oracle(        Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    per_seed.append({
        "seed":   seed,
        "random": metrics(rand_p, ste)["macro_p1"],
        "gmag":   metrics(gmag_p, ste)["macro_p1"],
        "ss":     metrics(ss_p,   ste)["macro_p1"],
        "coef_ref": metrics(cref_p, ste)["macro_p1"],
    })
    if seed % 5 == 0:
        print(f"  seed={seed:2d}  rand={per_seed[-1]['random']:.3f}  "
              f"gmag={per_seed[-1]['gmag']:.3f}  ss={per_seed[-1]['ss']:.3f}  "
              f"cref={per_seed[-1]['coef_ref']:.3f}")

print(f"\nDone in {time.perf_counter()-t0:.1f}s")

# Stats
ss_vals   = np.array([r["ss"]       for r in per_seed])
gmag_vals = np.array([r["gmag"]     for r in per_seed])
rand_vals = np.array([r["random"]   for r in per_seed])
cref_vals = np.array([r["coef_ref"] for r in per_seed])
diff      = ss_vals - gmag_vals
t_test    = stats.ttest_rel(ss_vals, gmag_vals)
ci_low, ci_high = np.percentile(diff, [2.5, 97.5])

print()
print("=" * 65)
print(f"Bank Marketing held-out (n={len(SEEDS_VALIDATE)} seeds, disjoint from sweep)")
print("-" * 65)
print(f"  random    {rand_vals.mean():.4f} ± {rand_vals.std():.4f}")
print(f"  gmag      {gmag_vals.mean():.4f} ± {gmag_vals.std():.4f}")
print(f"  ss        {ss_vals.mean():.4f} ± {ss_vals.std():.4f}")
print(f"  coef_ref  {cref_vals.mean():.4f} ± {cref_vals.std():.4f}")
print()
print(f"  Paired diff = {diff.mean():+.4f} ± {diff.std():.4f}")
print(f"  SS wins {(diff > 0).sum()}/{len(SEEDS_VALIDATE)} seeds")
print(f"  Paired t = {t_test.statistic:.3f}, p = {t_test.pvalue:.4f}")
print(f"  95% CI of paired diff: [{ci_low:+.4f}, {ci_high:+.4f}]")
print("=" * 65)

out = {
    "experiment": "exp22d_bank_marketing",
    "dataset":    "bank-marketing-openml",
    "n_sample_per_seed": N_SAMPLE,
    "n_test_per_seed":   N_TEST_PER_SEED,
    "n_seeds":           len(SEEDS_VALIDATE),
    "source_groups": {g: len(c) for g, c in source_cols.items()},
    "per_seed":      per_seed,
    "aggregate": {
        "random":   {"mean": round(float(rand_vals.mean()), 4),
                     "std":  round(float(rand_vals.std()),  4)},
        "gmag":     {"mean": round(float(gmag_vals.mean()), 4),
                     "std":  round(float(gmag_vals.std()),  4)},
        "ss":       {"mean": round(float(ss_vals.mean()),   4),
                     "std":  round(float(ss_vals.std()),    4)},
        "coef_ref": {"mean": round(float(cref_vals.mean()), 4),
                     "std":  round(float(cref_vals.std()),  4)},
    },
    "paired_test_ss_vs_gmag": {
        "mean_diff":      round(float(diff.mean()), 4),
        "std_diff":       round(float(diff.std()),  4),
        "t":              round(float(t_test.statistic), 4),
        "p_value":        round(float(t_test.pvalue),    6),
        "n_seeds_ss_wins": int((diff > 0).sum()),
        "ci_95_low":      round(float(ci_low),  4),
        "ci_95_high":     round(float(ci_high), 4),
    },
}
os.makedirs("results", exist_ok=True)
with open("results/exp22d_bank_marketing.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22d_bank_marketing.json")
