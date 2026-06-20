"""Robust statistical tests for the four real-data tiers — addresses
the VLDB reviewer's concern #4: t-test, Wilcoxon signed-rank, and
percentile-bootstrap CI on the same paired diff. The reviewer is right
that on n=20 paired differences the bootstrap is the more trustworthy
of the three; we now report all three side-by-side.
"""
import json, os
import numpy as np
from scipy import stats

DATASETS = {
    # exp22c has the K=1000 (tuned) HC numbers under "validate.per_seed"
    "home_credit":   ("results/exp22c_sab_k_sweep.json", "validate"),
    "compas":        ("results/exp22e_compas.json", None),
    "lending_club":  ("results/exp22f_lendingclub.json", None),
    "bank_marketing":("results/exp22d_bank_marketing.json", None),
}

print("=" * 80)
print(f"{'Dataset':<18s}  {'n':>4s}  {'Δ pp':>7s}  {'wins':>5s}  "
      f"{'t-test p':>10s}  {'Wilcoxon p':>12s}  {'boot 95% CI':>22s}")
print("-" * 80)

results = {}
for name, (path, subkey) in DATASETS.items():
    d = json.load(open(path))
    per = d[subkey]["per_seed"] if subkey else d["per_seed"]
    ss   = np.array([r["ss"]   for r in per])
    gmag = np.array([r["gmag"] for r in per])
    diff = ss - gmag

    t_stat, t_p = stats.ttest_rel(ss, gmag)
    w_stat, w_p = stats.wilcoxon(ss, gmag, zero_method="wilcox", alternative="two-sided")
    rng = np.random.default_rng(42)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(10000)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    wins = int((diff > 0).sum())

    print(f"{name:<18s}  {len(diff):>4d}  {100*diff.mean():+6.2f}  "
          f"{wins:>2d}/{len(diff):<2d}  "
          f"{t_p:>10.4f}  {w_p:>12.4f}  "
          f"[{100*ci_lo:+5.2f}, {100*ci_hi:+5.2f}]")
    results[name] = {
        "n_seeds":     len(diff),
        "mean_diff_pp": round(100*diff.mean(), 2),
        "wins":        wins,
        "t_test_t":    round(float(t_stat), 4),
        "t_test_p":    round(float(t_p),    6),
        "wilcoxon_W":  round(float(w_stat), 4),
        "wilcoxon_p":  round(float(w_p),    6),
        "bootstrap_95_ci_pp": [round(100*ci_lo, 2), round(100*ci_hi, 2)],
        "n_bootstrap_iters": 10000,
    }

print("=" * 80)
os.makedirs("results", exist_ok=True)
with open("results/exp22g_robust_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/exp22g_robust_stats.json")

# Bonferroni at α=0.05 over n=4 tests (per-test α = 0.0125)
ALPHA_BONF = 0.05 / 4
print(f"\nBonferroni (4 tests, per-test α = {ALPHA_BONF}):")
for name, r in results.items():
    pass_t  = r["t_test_p"]    <= ALPHA_BONF
    pass_w  = r["wilcoxon_p"]  <= ALPHA_BONF
    ci_pos  = r["bootstrap_95_ci_pp"][0] > 0
    print(f"  {name:<18s}  t {'PASS' if pass_t else 'FAIL':4s}  "
          f"Wilcoxon {'PASS' if pass_w else 'FAIL':4s}  "
          f"bootstrap-CI-positive {'YES' if ci_pos else 'NO'}")
