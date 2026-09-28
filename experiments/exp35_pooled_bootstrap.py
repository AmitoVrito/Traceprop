"""Pooled cross-seed hierarchical bootstrap for inline-precond vs LogIX-tuned.

Per-seed bootstraps (exp35_inlineprecond_bootstrap.py) give one CI per seed. This
pools all seeds into a SINGLE cross-seed CI via a hierarchical (nested) bootstrap
that respects the design's variance structure:

  each bootstrap iteration:
    1. resample the SEEDS with replacement (between-seed variance)
    2. within each resampled seed, resample the retraining SUBSETS and the
       held-out EVAL test examples with replacement (within-seed variance)
    3. compute the paired diff (inline LDS - LogIX_tuned LDS) per resampled seed,
       average across the resampled seeds -> one pooled diff for this iteration

The 2.5/97.5 percentiles of the pooled diffs are the cross-seed 95% CI.

Wording rule (fixed):
  pooled CI clearly > 0  -> "small but consistent improvement"
  otherwise              -> "matches LogIX's best preconditioning in a single pass"

Usage:
  python exp35_pooled_bootstrap.py --base exp35_tiny_track0_fairpmv3 --seeds 0 1 2 3 4 --kfac 8
"""
import argparse
import json
import os

import numpy as np

from exp35_inlineprecond_bootstrap import per_example_lds  # vectorized rank-Spearman LDS


def load_seed(base, s, kf):
    z = np.load(f"results/{base}_seed{s}_raw.npz")
    ip = f"attr_traceprop_factored_kfac{kf}_inlineprecond"
    tuned = "attr_logix_preconditioned_tuned"
    if ip not in z.files or tuned not in z.files:
        raise KeyError(f"seed {s}: missing {ip} or {tuned} in npz ({[f for f in z.files if f.startswith('attr_')]})")
    return {
        "masks": z["masks"], "margins": z["margins"],
        "eval_idx": z["precond_eval_idx"],
        "inline": z[ip], "tuned": z[tuned],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exp35_tiny_track0_fairpmv3")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--kfac", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/{args.base}_pooled_kfac{args.kfac}.json"

    data = {s: load_seed(args.base, s, args.kfac) for s in args.seeds}

    # per-seed point diffs (full data) + pooled point
    per_seed_point = {}
    for s in args.seeds:
        d = data[s]; allsub = np.arange(d["masks"].shape[0])
        di = per_example_lds(d["inline"], d["masks"], d["margins"], allsub, d["eval_idx"])
        dt = per_example_lds(d["tuned"], d["masks"], d["margins"], allsub, d["eval_idx"])
        per_seed_point[s] = round(float(di - dt), 4)
    pooled_point = float(np.mean(list(per_seed_point.values())))

    rng = np.random.default_rng(args.seed)
    seeds = np.array(args.seeds)
    pooled = np.empty(args.n_boot)
    for b in range(args.n_boot):
        seed_sample = rng.choice(seeds, size=len(seeds), replace=True)
        per = []
        for s in seed_sample:
            d = data[s]
            n_sub = d["masks"].shape[0]
            sub_b = rng.integers(0, n_sub, size=n_sub)
            ev = d["eval_idx"]
            test_b = ev[rng.integers(0, len(ev), size=len(ev))]
            di = per_example_lds(d["inline"], d["masks"], d["margins"], sub_b, test_b)
            dt = per_example_lds(d["tuned"], d["masks"], d["margins"], sub_b, test_b)
            per.append(di - dt)
        pooled[b] = np.nanmean(per)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    frac_gt0 = float(np.mean(pooled > 0))

    clearly_positive = lo > 0
    verdict = ("small but consistent improvement" if clearly_positive
               else "matches LogIX's best preconditioning in a single pass")

    out = {
        "base": args.base, "kfac": args.kfac, "seeds": args.seeds, "n_boot": args.n_boot,
        "per_seed_point_diff": per_seed_point,
        "pooled_point_diff": round(pooled_point, 4),
        "pooled_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "frac_boot_gt_0": round(frac_gt0, 3),
        "pooled_ci_clearly_positive": bool(clearly_positive),
        "verdict": verdict,
        "method": "hierarchical bootstrap: resample seeds, then subsets + eval examples within seed",
    }
    print(f"=== pooled cross-seed (kfac{args.kfac}), {len(args.seeds)} seeds ===")
    print(f"  per-seed paired diffs: {per_seed_point}")
    print(f"  pooled mean diff: {pooled_point:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={frac_gt0:.3f}")
    print(f"  VERDICT: {verdict}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
