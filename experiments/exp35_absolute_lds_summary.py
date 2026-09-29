"""Absolute (not paired-diff) LDS values for the paper's quality table, pooled
across the 5 fairness-campaign seeds, held-out eval split. No new experiments --
reads the existing exp35_tiny_track0_fairrand/fairpmv3 raw npz files.

Re-derivation requested during a main.tex audit: the paper's LDS table needs
absolute values (Traceprop dot/inline-precond/TRAK, LogIX dot/precond/tuned-precond
for both init strategies, random baseline), not just paired diffs, and needs them
computed on data that postdates the train_final reshuffle fix and uses the
fairness-tuned LogIX damping -- neither of which the original main.tex draft had.

Usage: python exp35_absolute_lds_summary.py
"""
import json

import numpy as np

from exp35_inlineprecond_bootstrap import per_example_lds


def pooled(base, key, seeds=range(5)):
    vals = []
    for s in seeds:
        z = np.load(f"results/{base}_seed{s}_raw.npz")
        allsub = np.arange(z["masks"].shape[0])
        v = per_example_lds(z[key], z["masks"], z["margins"], allsub, z["precond_eval_idx"])
        vals.append(v)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std()), [round(float(x), 4) for x in vals]


def main():
    out = {"track": 0, "scope": "all-layer", "seeds": [0, 1, 2, 3, 4],
           "eval_split": "held-out (precond_eval_idx)", "random_init": {}, "pca_init": {}}

    rand_methods = {
        "traceprop_dot": "attr_traceprop_dot",
        "traceprop_factored_kfac8_dot": "attr_traceprop_factored_kfac8_dot",
        "traceprop_factored_kfac8_inlineprecond": "attr_traceprop_factored_kfac8_inlineprecond",
        "traceprop_factored_kfac8_trak": "attr_traceprop_factored_kfac8_trak",
        "traceprop_trak": "attr_traceprop_trak",
        "logix_dot": "attr_logix_dot",
        "logix_preconditioned": "attr_logix_preconditioned",
        "logix_preconditioned_tuned": "attr_logix_preconditioned_tuned",
    }
    print("=== fairrand (LogIX random-init) base, track0/all-layer, held-out eval, 5 seeds ===")
    for label, key in rand_methods.items():
        m, s, vals = pooled("exp35_tiny_track0_fairrand", key)
        out["random_init"][label] = {"mean": round(m, 4), "std": round(s, 4), "per_seed": vals}
        print(f"{label:42s} {m:+.4f} +/- {s:.4f}")

    pca_methods = {
        "logix_dot": "attr_logix_dot",
        "logix_preconditioned": "attr_logix_preconditioned",
        "logix_preconditioned_tuned": "attr_logix_preconditioned_tuned",
    }
    print("\n=== fairpmv3 (LogIX pca-init) base, track0/all-layer, held-out eval, 5 seeds ===")
    for label, key in pca_methods.items():
        m, s, vals = pooled("exp35_tiny_track0_fairpmv3", key)
        out["pca_init"][label] = {"mean": round(m, 4), "std": round(s, 4), "per_seed": vals}
        print(f"{label:42s} {m:+.4f} +/- {s:.4f}")

    rand_random_vals = []
    for s in range(5):
        z = np.load(f"results/exp35_tiny_track0_fairrand_seed{s}_raw.npz")
        ev = z["precond_eval_idx"]
        rand_random_vals.append(float(np.nanmean(z["r_random"][ev])))
    out["random_baseline"] = {
        "mean": round(float(np.mean(rand_random_vals)), 4),
        "std": round(float(np.std(rand_random_vals)), 4),
    }
    print(f"\nrandom baseline: {out['random_baseline']['mean']:+.4f} +/- "
          f"{out['random_baseline']['std']:.4f}")

    with open("results/exp35_absolute_lds_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/exp35_absolute_lds_summary.json")


if __name__ == "__main__":
    main()
