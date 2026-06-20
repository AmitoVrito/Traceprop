"""SAB v1 — 20-seed paired-t-test on the real Home Credit tier.

Pass-2 reviewer concern P5: the +8.3pp lift of Traceprop-SS over the
gradmag baseline (0.586 ± 0.06 vs 0.503 ± 0.12, n=5) is not significant
at α=0.05 under a two-sample t-test (p ≈ 0.20). We rerun the real-HC
tier only with 20 seeds and report a PAIRED t-test (the gradmag and SS
predictions share the per-seed Home Credit subsample, so paired
comparison is appropriate and substantially more powerful than
two-sample).

Reuses gen_home_credit, baseline_grad_argmax, ss_attrib_block from
exp22_sab_benchmark.py.
"""

import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_sab_benchmark import (
    gen_home_credit, baseline_random, baseline_grad_argmax,
    ss_attrib_block, coef_oracle, metrics, _HC_CACHE
)

SEEDS = list(range(20))
print(f"SAB v1 real-Home-Credit tier, {len(SEEDS)} seeds, paired t-test")
print("=" * 65)

per_seed = []
t0 = time.perf_counter()
for seed in SEEDS:
    X_tr, y_tr, src_tr, X_te, y_te, src_te, src_cols, src_names = \
        gen_home_credit(seed)
    rand_p   = baseline_random(    X_tr, y_tr, src_tr, X_te, y_te, src_names, src_cols, seed)
    gmag_p   = baseline_grad_argmax(X_tr, y_tr, src_tr, X_te, y_te, src_names, src_cols, seed)
    ss_p     = ss_attrib_block(    X_tr, y_tr, src_tr, X_te, y_te, src_names, src_cols, seed)
    orac_p   = coef_oracle(        X_tr, y_tr, src_tr, X_te, y_te, src_names, src_cols, seed)
    per_seed.append({
        "seed":   seed,
        "random": metrics(rand_p, src_te)["macro_p1"],
        "gmag":   metrics(gmag_p, src_te)["macro_p1"],
        "ss":     metrics(ss_p,   src_te)["macro_p1"],
        "oracle": metrics(orac_p, src_te)["macro_p1"],
    })
    print(f"  seed={seed:2d}  rand={per_seed[-1]['random']:.3f}  "
          f"gmag={per_seed[-1]['gmag']:.3f}  ss={per_seed[-1]['ss']:.3f}  "
          f"oracle={per_seed[-1]['oracle']:.3f}")

elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed:.1f}s")

# Aggregate
def stats_of(key):
    vals = np.array([r[key] for r in per_seed])
    return {"mean": round(float(vals.mean()), 4),
            "std":  round(float(vals.std()),  4),
            "n":    len(vals),
            "values": [round(float(v), 4) for v in vals]}

agg = {k: stats_of(k) for k in ("random", "gmag", "ss", "oracle")}

# Paired tests (SS vs gmag, SS vs random)
ss_vals   = np.array([r["ss"]   for r in per_seed])
gmag_vals = np.array([r["gmag"] for r in per_seed])
rand_vals = np.array([r["random"] for r in per_seed])

diff_ss_gmag = ss_vals - gmag_vals
t_paired_ss_gmag = stats.ttest_rel(ss_vals, gmag_vals)
t_paired_ss_rand = stats.ttest_rel(ss_vals, rand_vals)

# Two-sample (unpaired) for comparison
t_two_ss_gmag = stats.ttest_ind(ss_vals, gmag_vals, equal_var=False)

# 95% CI on paired diff
ci_low = float(np.percentile(diff_ss_gmag, 2.5))
ci_high = float(np.percentile(diff_ss_gmag, 97.5))

out = {
    "experiment": "exp22b_sab_realhc_20seed",
    "tier":       "real_homecredit",
    "n_seeds":    len(SEEDS),
    "n_test_per_seed": 200,
    "per_seed":   per_seed,
    "aggregate":  agg,
    "paired_t_test_ss_vs_gmag": {
        "mean_diff":           round(float(diff_ss_gmag.mean()), 4),
        "std_diff":            round(float(diff_ss_gmag.std()),  4),
        "t":                   round(float(t_paired_ss_gmag.statistic), 4),
        "p_value":             round(float(t_paired_ss_gmag.pvalue),    6),
        "df":                  len(SEEDS) - 1,
        "ci_95_low":           round(ci_low, 4),
        "ci_95_high":          round(ci_high, 4),
        "n_seeds_ss_wins":     int((diff_ss_gmag > 0).sum()),
    },
    "paired_t_test_ss_vs_random": {
        "t":       round(float(t_paired_ss_rand.statistic), 4),
        "p_value": round(float(t_paired_ss_rand.pvalue),    6),
    },
    "two_sample_t_test_ss_vs_gmag_for_comparison": {
        "t":       round(float(t_two_ss_gmag.statistic), 4),
        "p_value": round(float(t_two_ss_gmag.pvalue),    6),
    },
}

print()
print("=" * 65)
print("Summary (real-Home-Credit, 20 seeds)")
print("-" * 65)
for k in ("random", "gmag", "ss", "oracle"):
    print(f"  {k:8s}  {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
print()
print(f"  SS - gmag (paired diff):  "
      f"{diff_ss_gmag.mean():+.4f} ± {diff_ss_gmag.std():.4f}")
print(f"  SS wins on {(diff_ss_gmag > 0).sum()}/{len(SEEDS)} seeds")
print(f"  Paired t-test (SS vs gmag): "
      f"t={t_paired_ss_gmag.statistic:.3f}, p={t_paired_ss_gmag.pvalue:.4f}")
print(f"  Two-sample for comparison:  "
      f"t={t_two_ss_gmag.statistic:.3f}, p={t_two_ss_gmag.pvalue:.4f}")
print("=" * 65)

os.makedirs("results", exist_ok=True)
with open("results/exp22b_sab_realhc_20seed.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22b_sab_realhc_20seed.json")
