"""exp32 -- Paired bootstrap for exp29's inline-vs-final LDS comparison.

Reviewer item 9 follow-up: the raw means (inline_dot 0.087 vs final_dot 0.031;
inline_trak 0.174 vs final_trak 0.097) are each within about one standard
deviation of the other, so "inline scores higher" needs a real significance
check, not just a comparison of point estimates. This does a paired bootstrap
over test examples using the raw per-example Spearman r values exp29 now
saves to results/exp29_<tag>_raw.npz (r_final_dot, r_inline_dot, etc., all
aligned to the same test-example order, plus the raw subset masks/margins).

No GPU needed -- pure numpy on saved arrays.

Usage:
    python exp32_paired_bootstrap.py results/exp29_hf_gpt2_raw.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """a, b: paired per-test-example scores (same test examples, same order).
    Returns dict with the observed mean difference (a - b), a bootstrap 95% CI
    for that difference (resampling test examples with replacement), and a
    two-sided bootstrap p-value for the null that the true difference is 0."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diff = a - b
    obs = float(np.mean(diff))

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diff[idx])

    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    # two-sided p-value: fraction of bootstrap draws on the other side of 0
    # from the observed sign (standard bootstrap-CI-inversion p-value)
    if obs >= 0:
        p = 2 * min(np.mean(boot_diffs <= 0), 0.5)
    else:
        p = 2 * min(np.mean(boot_diffs >= 0), 0.5)

    return {
        "n_test": int(n),
        "n_boot": n_boot,
        "observed_mean_diff": round(obs, 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "p_value_two_sided": round(float(p), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def run(npz_path: str, n_boot: int = 10000):
    data = np.load(npz_path)
    keys = [k for k in data.files if k.startswith("r_")]
    print(f"[exp32] loaded {npz_path}: conditions = {[k[2:] for k in keys]}, "
          f"n_test = {data[keys[0]].shape[0]}")

    pairs = [
        ("inline_dot", "final_dot"),
        ("inline_trak", "final_trak"),
        ("inline_dot", "random"),
        ("final_dot", "random"),
        ("inline_trak", "random"),
        ("final_trak", "random"),
    ]

    out = {}
    for a_name, b_name in pairs:
        ka, kb = f"r_{a_name}", f"r_{b_name}"
        if ka not in data or kb not in data:
            continue
        a, b = data[ka], data[kb]
        valid = ~(np.isnan(a) | np.isnan(b))
        result = paired_bootstrap(a[valid], b[valid], n_boot=n_boot)
        out[f"{a_name}_vs_{b_name}"] = result
        sig = "excludes 0 (significant at 95%)" if result["excludes_zero"] else "includes 0 (not significant at 95%)"
        print(f"  {a_name} - {b_name}: diff={result['observed_mean_diff']:+.4f}  "
              f"95% CI=[{result['ci95_low']:+.4f}, {result['ci95_high']:+.4f}]  "
              f"p={result['p_value_two_sided']:.4f}  {sig}")

    fn = npz_path.replace("_raw.npz", "_bootstrap.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    run(args.npz_path, args.n_boot)


if __name__ == "__main__":
    main()
