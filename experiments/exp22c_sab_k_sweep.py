"""SAB v1 — sweep K (top-K parameter in SS_attrib_block) on real-HC.

Goal: improve SS lift over gradmag enough that the paired t-test is
significant. We run K in {10, 25, 50, 100, 200, 500} across 10 seeds and
pick K* that maximises the paired diff. Then run K* on the full 20 seeds
to validate.
"""

import json, os, sys, time, numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp22_sab_benchmark import (
    gen_home_credit, baseline_grad_argmax, metrics,
    fit_and_grad, gram_solve,
)

def ss_attrib_block_k(K):
    def fn(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
        _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
        scores = gram_solve(G_tr, G_te)
        src_tr_arr = np.array(src_tr)
        n_train = scores.shape[0]
        priors = {s: max((src_tr_arr == s).sum(), 1) / n_train for s in src_names}
        out = []
        for j in range(scores.shape[1]):
            block_mag = {s: float(np.linalg.norm(G_te[j, source_cols[s]]))
                         for s in src_names}
            top = np.argsort(-np.abs(scores[:, j]))[:K]
            src_mass = defaultdict(float)
            for i in top: src_mass[src_tr_arr[i]] += abs(scores[i, j])
            lift = {s: src_mass.get(s, 0.0) / priors[s] for s in src_names}
            combined = {s: block_mag[s] * lift[s] for s in src_names}
            out.append(max(combined, key=combined.get))
        return out
    return fn

K_VALUES = [10, 25, 50, 100, 200, 500, 1000]
SEEDS_SWEEP    = list(range(10))           # K selection
SEEDS_VALIDATE = list(range(10, 30))       # disjoint hold-out for honesty

print("=" * 65)
print("K sweep on real-Home-Credit (10 seeds per K)")
print("=" * 65)
sweep = {}
for K in K_VALUES:
    diffs, ss_vals, gm_vals = [], [], []
    method = ss_attrib_block_k(K)
    t0 = time.perf_counter()
    for seed in SEEDS_SWEEP:
        Xtr, ytr, stt, Xte, yte, ste, sc, sn = gen_home_credit(seed)
        gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
        ss_p   = method(             Xtr, ytr, stt, Xte, yte, sn, sc, seed)
        g_m = metrics(gmag_p, ste)["macro_p1"]
        s_m = metrics(ss_p,   ste)["macro_p1"]
        gm_vals.append(g_m); ss_vals.append(s_m); diffs.append(s_m - g_m)
    arr = np.array(diffs)
    t_test = stats.ttest_rel(ss_vals, gm_vals)
    sweep[K] = {
        "mean_ss":   round(float(np.mean(ss_vals)), 4),
        "mean_gmag": round(float(np.mean(gm_vals)), 4),
        "mean_diff": round(float(arr.mean()), 4),
        "std_diff":  round(float(arr.std()),  4),
        "t":         round(float(t_test.statistic), 4),
        "p_value":   round(float(t_test.pvalue),    6),
        "n_seeds_ss_wins": int((arr > 0).sum()),
        "n_seeds": len(SEEDS_SWEEP),
    }
    print(f"  K={K:4d}  ss={sweep[K]['mean_ss']:.3f} gmag={sweep[K]['mean_gmag']:.3f}  "
          f"diff={sweep[K]['mean_diff']:+.4f}  p={sweep[K]['p_value']:.3f}  "
          f"wins={sweep[K]['n_seeds_ss_wins']}/{len(SEEDS_SWEEP)}  "
          f"[{time.perf_counter()-t0:.0f}s]")

# Best K by mean_diff (then by p_value as tiebreaker)
best_K = max(K_VALUES, key=lambda k: (sweep[k]["mean_diff"], -sweep[k]["p_value"]))
print(f"\nBest K by mean_diff: K={best_K}  (mean_diff={sweep[best_K]['mean_diff']:+.4f})")

print(f"\n=== Validate K={best_K} on {len(SEEDS_VALIDATE)} seeds ===")
method = ss_attrib_block_k(best_K)
val_seed = []
for seed in SEEDS_VALIDATE:
    Xtr, ytr, stt, Xte, yte, ste, sc, sn = gen_home_credit(seed)
    gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    ss_p   = method(             Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    val_seed.append({
        "seed":   seed,
        "gmag":   metrics(gmag_p, ste)["macro_p1"],
        "ss":     metrics(ss_p,   ste)["macro_p1"],
    })

ss_vals = np.array([v["ss"]   for v in val_seed])
gm_vals = np.array([v["gmag"] for v in val_seed])
diff = ss_vals - gm_vals
t_test = stats.ttest_rel(ss_vals, gm_vals)
ci_low, ci_high = np.percentile(diff, [2.5, 97.5])

print(f"  ss        {ss_vals.mean():.4f} ± {ss_vals.std():.4f}")
print(f"  gmag      {gm_vals.mean():.4f} ± {gm_vals.std():.4f}")
print(f"  paired diff = {diff.mean():+.4f} ± {diff.std():.4f}")
print(f"  paired t = {t_test.statistic:.3f}, p = {t_test.pvalue:.4f}")
print(f"  SS wins on {(diff > 0).sum()}/{len(val_seed)} seeds")
print(f"  95% CI of paired diff (percentile): [{ci_low:+.4f}, {ci_high:+.4f}]")

out = {
    "experiment": "exp22c_sab_k_sweep",
    "tier":       "real_homecredit",
    "sweep_K_values":     K_VALUES,
    "sweep_n_seeds":      len(SEEDS_SWEEP),
    "sweep":              {str(k): v for k, v in sweep.items()},
    "best_K_by_mean_diff": best_K,
    "validate": {
        "K":            best_K,
        "n_seeds":      len(SEEDS_VALIDATE),
        "per_seed":     val_seed,
        "ss_mean":      round(float(ss_vals.mean()), 4),
        "ss_std":       round(float(ss_vals.std()),  4),
        "gmag_mean":    round(float(gm_vals.mean()), 4),
        "gmag_std":     round(float(gm_vals.std()),  4),
        "mean_paired_diff":   round(float(diff.mean()), 4),
        "std_paired_diff":    round(float(diff.std()),  4),
        "paired_t":           round(float(t_test.statistic), 4),
        "paired_p":           round(float(t_test.pvalue),    6),
        "ci_95_low":          round(float(ci_low),  4),
        "ci_95_high":         round(float(ci_high), 4),
        "n_seeds_ss_wins":    int((diff > 0).sum()),
    },
}
os.makedirs("results", exist_ok=True)
with open("results/exp22c_sab_k_sweep.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22c_sab_k_sweep.json")
