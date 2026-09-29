"""Pooled cross-seed hierarchical bootstrap: LogIX random-init vs. LogIX PCA-init,
both fairness-tuned (attr_logix_preconditioned_tuned).

No prior run saved this comparison to a results file -- it was only printed to a
terminal in an earlier session and recorded in memory text, not independently
reproducible from disk. This reconstructs it from data that DOES exist on disk:
exp35_tiny_track0_fairrand_seed{s}_raw.npz (lora_init=random) and
exp35_tiny_track0_fairpmv3_seed{s}_raw.npz (lora_init=pca) share identical masks/
margins per seed (same retraining subsets, verified via np.array_equal before
trusting this script), so the comparison is apples-to-apples across the same
ground truth. Same hierarchical bootstrap methodology as exp35_pooled_bootstrap.py
(resample seeds, then subsets + eval examples within seed).

Usage:
  python exp35_logix_random_vs_pca_bootstrap.py --seeds 0 1 2 3 4
"""
import argparse
import json

import numpy as np

from exp35_inlineprecond_bootstrap import per_example_lds


def load_seed(base, s):
    z = np.load(f"results/{base}_seed{s}_raw.npz")
    tuned = "attr_logix_preconditioned_tuned"
    if tuned not in z.files:
        raise KeyError(f"{base} seed {s}: missing {tuned}")
    return {
        "masks": z["masks"], "margins": z["margins"],
        "eval_idx": z["precond_eval_idx"], "tuned": z[tuned],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_base", default="exp35_tiny_track0_fairrand")
    ap.add_argument("--pca_base", default="exp35_tiny_track0_fairpmv3")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp35_logix_random_vs_pca_pooled.json")
    args = ap.parse_args()

    rand = {s: load_seed(args.random_base, s) for s in args.seeds}
    pca = {s: load_seed(args.pca_base, s) for s in args.seeds}

    for s in args.seeds:
        if not np.array_equal(rand[s]["masks"], pca[s]["masks"]):
            raise SystemExit(f"seed {s}: masks differ between {args.random_base} and "
                              f"{args.pca_base} -- NOT the same retraining subsets, refusing "
                              f"to compare across a mismatched ground truth.")
        if not np.array_equal(rand[s]["margins"], pca[s]["margins"]):
            raise SystemExit(f"seed {s}: margins differ, refusing to compare.")
    print(f"verified: masks/margins identical across {args.random_base} and {args.pca_base} "
          f"for all {len(args.seeds)} seeds")

    per_seed_point = {}
    for s in args.seeds:
        d = rand[s]; allsub = np.arange(d["masks"].shape[0])
        lds_rand = per_example_lds(d["tuned"], d["masks"], d["margins"], allsub, d["eval_idx"])
        lds_pca = per_example_lds(pca[s]["tuned"], pca[s]["masks"], pca[s]["margins"],
                                   allsub, pca[s]["eval_idx"])
        per_seed_point[s] = round(float(lds_rand - lds_pca), 4)
    pooled_point = float(np.mean(list(per_seed_point.values())))

    rng = np.random.default_rng(args.seed)
    seeds = np.array(args.seeds)
    pooled = np.empty(args.n_boot)
    for b in range(args.n_boot):
        seed_sample = rng.choice(seeds, size=len(seeds), replace=True)
        per = []
        for s in seed_sample:
            dr, dp = rand[s], pca[s]
            n_sub = dr["masks"].shape[0]
            sub_b = rng.integers(0, n_sub, size=n_sub)
            ev = dr["eval_idx"]
            test_b = ev[rng.integers(0, len(ev), size=len(ev))]
            lr = per_example_lds(dr["tuned"], dr["masks"], dr["margins"], sub_b, test_b)
            lp = per_example_lds(dp["tuned"], dp["masks"], dp["margins"], sub_b, test_b)
            per.append(lr - lp)
        pooled[b] = np.nanmean(per)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    frac_gt0 = float(np.mean(pooled > 0))

    out = {
        "random_base": args.random_base, "pca_base": args.pca_base,
        "seeds": args.seeds, "n_boot": args.n_boot,
        "per_seed_point_diff_random_minus_pca": per_seed_point,
        "pooled_point_diff": round(pooled_point, 4),
        "pooled_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "frac_boot_gt_0": round(frac_gt0, 3),
        "pooled_ci_clearly_positive": bool(lo > 0),
        "method": "hierarchical bootstrap: resample seeds, then subsets + eval examples "
                  "within seed. Compares LogIX random-init vs. PCA-init, both fairness-"
                  "tuned (attr_logix_preconditioned_tuned), on IDENTICAL retraining subsets "
                  "per seed (verified before running).",
    }
    print(f"=== LogIX random-init vs. PCA-init (tuned), {len(args.seeds)} seeds ===")
    print(f"  per-seed diffs (random - pca): {per_seed_point}")
    print(f"  pooled mean diff: {pooled_point:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={frac_gt0:.3f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
