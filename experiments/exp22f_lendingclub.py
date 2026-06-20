"""Experiment 22f: fourth real dataset for SAB v1 — Lending Club at scale.

Reviewer concern Q5 (v5): bring a genuinely large dataset.
Lending Club historical data (2007-2018): n=2,260,668 rows, ~150 columns,
real loan-application audit setting. Per-source groups follow the
public Lending Club data dictionary (operational separation between
ingestion pipelines):

  - borrower:    annual income, employment, home ownership, region
                 (collected at application time from the borrower)
  - credit_hist: revolving balance, open accounts, delinquencies,
                 inquiries, DTI (pulled from the credit bureau)
  - loan_terms:  amount, term, interest rate, installment, grade
                 (set by Lending Club's underwriting model)

Same protocol as exp22b/d/e: K = min(1000, max(50, n_train/20)),
disjoint 20-seed hold-out (seeds 10-29), paired t-test.
Subsample size: 50,000 per seed (10x larger than COMPAS; 2.5x larger
than HC) to stress the protocol at scale.
"""

import json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_sab_benchmark import (
    baseline_random, baseline_grad_argmax, ss_attrib_block, coef_oracle,
    metrics,
)

LENDINGCLUB_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "lendingclub", "loan.csv")
N_SAMPLE_PER_SEED = 50_000
N_TEST_PER_SEED   = 200
SEEDS_VALIDATE    = list(range(10, 30))

GROUPS = {
    "borrower":    ["annual_inc", "emp_length", "home_ownership",
                    "addr_state", "verification_status"],
    "credit_hist": ["dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
                    "pub_rec", "revol_bal", "revol_util", "total_acc"],
    "loan_terms":  ["loan_amnt", "term", "int_rate", "installment",
                    "grade", "purpose"],
}
ALL_COLS  = sum(GROUPS.values(), []) + ["loan_status"]

print(f"Loading {LENDINGCLUB_PATH} (one-time)...")
t0 = time.perf_counter()
df = pd.read_csv(LENDINGCLUB_PATH, usecols=lambda c: c in ALL_COLS,
                 low_memory=False)
print(f"  loaded n={len(df):,} cols={df.columns.tolist()} "
      f"[{time.perf_counter()-t0:.1f}s]")

# Binary target: Charged Off vs Fully Paid (drop in-flight loans)
df = df[df["loan_status"].isin(["Charged Off", "Fully Paid"])].reset_index(drop=True)
y = (df["loan_status"] == "Charged Off").astype(np.float32).to_numpy()
df = df.drop(columns=["loan_status"])
print(f"  after filtering to completed loans: n={len(df):,}  "
      f"default rate={y.mean():.3f}")

# Numeric cleanups
for c in ["term"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.extract(r"(\d+)").astype(float)
for c in ["int_rate", "revol_util"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.replace("%", "").astype(float)

# Cast numerics
NUMERIC = {"annual_inc","dti","delinq_2yrs","inq_last_6mths","open_acc",
           "pub_rec","revol_bal","revol_util","total_acc","loan_amnt",
           "term","int_rate","installment"}
for c in df.columns:
    if c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# One-hot encode while tracking origin
X_oh = pd.get_dummies(df, drop_first=True).astype(np.float64)
origin = []
for c in df.columns:
    if c in NUMERIC:
        origin.append(c)
    else:
        k = df[c].nunique(dropna=True) - 1
        origin.extend([c] * max(k, 0))

# Map encoded columns to source groups
source_cols = {g: [] for g in GROUPS}
for j, col in enumerate(origin):
    for g, members in GROUPS.items():
        if col in members:
            source_cols[g].append(j); break
keep = sorted(sum(source_cols.values(), []))
j_map = {old: new for new, old in enumerate(keep)}
source_cols = {g: [j_map[j] for j in idx] for g, idx in source_cols.items()}
X = X_oh.to_numpy()[:, keep]

# Median-impute NaNs (no zero-padding leakage)
col_med = np.nanmedian(X, axis=0)
nan_mask = np.isnan(X)
X = np.where(nan_mask, col_med, X)

print(f"  X shape after encoding+impute: {X.shape}  "
      f"groups: {dict((g, len(c)) for g, c in source_cols.items())}")

# Audit-realistic primary-source label
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(f"  Training reference model on full n={len(X):,}...")
t0 = time.perf_counter()
ref = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                         random_state=42).fit(X_std, y)
ref_coef = ref.coef_[0]
print(f"  ref model done [{time.perf_counter()-t0:.1f}s]")

contribs = np.stack([
    np.abs(X_std[:, source_cols[g]] @ ref_coef[source_cols[g]])
    for g in GROUPS], axis=1)
src_idx = contribs.argmax(axis=1)
src_names = list(GROUPS)
src_all = [src_names[i] for i in src_idx]

from collections import Counter
print(f"  Source distribution (full): {dict(Counter(src_all))}")

def gen_seed(seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), N_SAMPLE_PER_SEED, replace=False)
    X_s, y_s, src_s = X[idx], y[idx], [src_all[i] for i in idx]
    te_n = min(N_TEST_PER_SEED, len(X_s)//5)
    return (X_s[te_n:], y_s[te_n:], src_s[te_n:],
            X_s[:te_n], y_s[:te_n], src_s[:te_n],
            source_cols, src_names)

print(f"\nRunning {len(SEEDS_VALIDATE)} held-out seeds at n={N_SAMPLE_PER_SEED:,}/seed...")
per_seed = []
t0 = time.perf_counter()
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
    if (seed - SEEDS_VALIDATE[0]) % 5 == 0:
        print(f"  seed={seed:2d}  rand={per_seed[-1]['random']:.3f}  "
              f"gmag={per_seed[-1]['gmag']:.3f}  ss={per_seed[-1]['ss']:.3f}  "
              f"cref={per_seed[-1]['coef_ref']:.3f}")

print(f"\nDone in {time.perf_counter()-t0:.1f}s")

ss_vals   = np.array([r["ss"]       for r in per_seed])
gmag_vals = np.array([r["gmag"]     for r in per_seed])
rand_vals = np.array([r["random"]   for r in per_seed])
cref_vals = np.array([r["coef_ref"] for r in per_seed])
diff      = ss_vals - gmag_vals
t_test    = stats.ttest_rel(ss_vals, gmag_vals)
ci_low, ci_high = np.percentile(diff, [2.5, 97.5])

print()
print("=" * 65)
print(f"Lending Club held-out (n={len(SEEDS_VALIDATE)} seeds, "
      f"{N_SAMPLE_PER_SEED:,}/seed, dataset n={len(X):,})")
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
    "experiment": "exp22f_lendingclub",
    "dataset":    "lending-club-2007-2018",
    "n_total":    int(len(X)),
    "n_sample_per_seed": N_SAMPLE_PER_SEED,
    "n_test_per_seed":   N_TEST_PER_SEED,
    "n_seeds":           len(SEEDS_VALIDATE),
    "source_groups":     {g: len(c) for g, c in source_cols.items()},
    "source_distribution_full": dict(Counter(src_all)),
    "per_seed":          per_seed,
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
with open("results/exp22f_lendingclub.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22f_lendingclub.json")
