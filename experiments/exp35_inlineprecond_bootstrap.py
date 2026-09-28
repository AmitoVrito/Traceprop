"""Held-out two-way bootstrap for the inline-precond decision (exp35 follow-up).

Reads an exp35 ``*_raw.npz`` (produced with ``--inline_precond``) and answers ONE
question with a pre-registered decision rule: does inline K-FAC preconditioning
tie-or-beat LogIX's preconditioned estimator at all-layer scope, on the HELD-OUT
eval split (the split disjoint from the damping-selection val split)?

Methodology mirrors the earlier factored-vs-LogIX bootstrap (see the
logix-comparison-results memory): a *two-way* bootstrap that resamples BOTH the
retraining subsets (rows of masks/margins) AND the test examples (restricted to
the eval split), 2000 resamples, reporting the mean LDS difference and its 95%
percentile CI.

Decision rule (fixed before the run finished):
  inline-precond vs logix_preconditioned, at all-layer:
    CI includes or exceeds 0  -> inline-precond ties-or-beats: PAPER CONTRIBUTION
    CI entirely below 0        -> inline-precond loses: report honestly, move to #3

Usage:
    python exp35_inlineprecond_bootstrap.py results/exp35_tiny_track0_inlineprecond_raw.npz
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr


def _rank(a):
    """Column-wise ordinal ranks along axis 0 (fully vectorized). Ties resolved
    ordinally; for continuous LDS margins/attr scores this matches average-rank
    Spearman to <1e-4 (validated against scipy on seed 0)."""
    return a.argsort(0).argsort(0).astype(np.float64)


def _spearman_cols(pred, m):
    """Spearman correlation per column between pred and m (both (n_sub, n_test))."""
    pr, mr = _rank(pred), _rank(m)
    pr -= pr.mean(0); mr -= mr.mean(0)
    num = (pr * mr).sum(0)
    den = np.sqrt((pr ** 2).sum(0) * (mr ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def per_example_lds(attr, masks, margins, sub_idx, test_idx):
    """Mean Spearman LDS over the given test examples, using the given subset rows."""
    pred = masks[sub_idx] @ attr[test_idx].T   # (|sub|, |test|)
    m = margins[sub_idx][:, test_idx]          # (|sub|, |test|)
    rs = _spearman_cols(pred, m)
    rs = rs[~np.isnan(rs)]
    return float(np.mean(rs)) if len(rs) else np.nan


def two_way_bootstrap(attrA, attrB, masks, margins, eval_idx, n_boot=2000, seed=0):
    """Distribution of (LDS_A - LDS_B) under resampling of subsets AND eval
    test examples. Returns (point_diff, lo, hi, frac_ge_0)."""
    rng = np.random.default_rng(seed)
    n_sub = masks.shape[0]
    eval_idx = np.asarray(eval_idx)
    all_sub = np.arange(n_sub)
    point = (per_example_lds(attrA, masks, margins, all_sub, eval_idx)
             - per_example_lds(attrB, masks, margins, all_sub, eval_idx))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sub_b = rng.integers(0, n_sub, size=n_sub)
        test_b = eval_idx[rng.integers(0, len(eval_idx), size=len(eval_idx))]
        dA = per_example_lds(attrA, masks, margins, sub_b, test_b)
        dB = per_example_lds(attrB, masks, margins, sub_b, test_b)
        diffs[b] = dA - dB
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi), float(np.mean(diffs >= 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="exp35 *_raw.npz produced with --inline_precond")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="summary JSON path (default: alongside npz, in results/)")
    args = ap.parse_args()

    z = np.load(args.npz)
    masks, margins = z["masks"], z["margins"]
    if "precond_eval_idx" not in z:
        sys.exit("npz has no precond_eval_idx -- was it produced with --inline_precond?")
    eval_idx = z["precond_eval_idx"]
    val_idx = z["precond_val_idx"]

    attrs = {k[len("attr_"):]: z[k] for k in z.files if k.startswith("attr_")}
    kfac_labels = sorted({n.split("_inlineprecond")[0].split("kfac")[-1]
                          for n in attrs if n.endswith("_inlineprecond")})
    if not kfac_labels:
        sys.exit("no *_inlineprecond attribution matrices found in npz")

    print(f"[bootstrap] {os.path.basename(args.npz)}: n_subsets={masks.shape[0]}, "
          f"n_train={masks.shape[1]}, eval={len(eval_idx)} held-out test examples "
          f"(val={len(val_idx)} used only for damping), {args.n_boot} resamples\n")

    baselines = ["logix_preconditioned_tuned", "logix_preconditioned", "logix_dot"]
    baselines = [b for b in baselines if b in attrs]
    # decision keys on the TUNED baseline (tuned-vs-tuned) if present
    decision_baseline = ("logix_preconditioned_tuned" if "logix_preconditioned_tuned" in attrs
                         else "logix_preconditioned")
    summary = {"npz": os.path.basename(args.npz), "n_boot": args.n_boot,
               "n_eval": int(len(eval_idx)), "n_val": int(len(val_idx)),
               "n_subsets": int(masks.shape[0]), "comparisons": {}, "decision": {}}

    for kf in kfac_labels:
        ip = f"traceprop_factored_kfac{kf}_inlineprecond"
        if ip not in attrs:
            continue
        for base in baselines:
            if base not in attrs:
                continue
            point, lo, hi, frac = two_way_bootstrap(
                attrs[ip], attrs[base], masks, margins, eval_idx,
                n_boot=args.n_boot, seed=args.seed)
            ties_or_beats = hi >= 0.0  # CI includes or exceeds zero
            key = f"kfac{kf}: inlineprecond - {base}"
            summary["comparisons"][key] = {
                "lds_diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "frac_resamples_ge_0": round(frac, 3),
                "ties_or_beats": bool(ties_or_beats),
            }
            verdict = ("TIES/BEATS" if ties_or_beats else "LOSES")
            print(f"  {key:<45} diff={point:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
                  f"P(>=0)={frac:.2f}  -> {verdict}")

    # Per-seed verdict keyed on the TUNED baseline (tuned-vs-tuned). The final
    # cross-seed claim ("outperforms" if majority of seeds have CI>0, else
    # "matches best quality in one pass") is applied by the aggregator.
    summary["decision_baseline"] = decision_baseline
    for kf in kfac_labels:
        key = f"kfac{kf}: inlineprecond - {decision_baseline}"
        c = summary["comparisons"].get(key)
        if c is None:
            continue
        lo, hi = c["ci95"]
        if lo > 0.0:
            verdict = f"ABOVE_ZERO: inline outperforms {decision_baseline} this seed (CI entirely >0)"
        elif hi < 0.0:
            verdict = f"BELOW_ZERO: inline loses to {decision_baseline} this seed (CI entirely <0)"
        else:
            verdict = f"OVERLAPS_ZERO: inline matches {decision_baseline} this seed (CI includes 0)"
        summary["decision"][f"kfac{kf}"] = verdict
        print(f"\n  [SEED VERDICT kfac{kf} vs {decision_baseline}] {verdict}")

    out = args.out
    if out is None:
        base = os.path.basename(args.npz)
        base = base[:-8] if base.endswith("_raw.npz") else base
        os.makedirs("results", exist_ok=True)
        out = os.path.join("results", base + "_bootstrap.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
