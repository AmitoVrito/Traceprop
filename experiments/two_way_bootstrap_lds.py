"""Two-way (test-example + subset) bootstrap for comparing two attribution
methods' LDS on the same ground truth.

The naive approach -- bootstrap resampling the already-computed per-test-
example Spearman r values -- treats the 200 retrained subsets and the single
trained model as fixed, so its CI only describes "on this model, with these
particular subsets, which method ranks higher on nearly every test example."
That understates the comparison's uncertainty and can give a CI far too
narrow to generalize from (confirmed: a paired bootstrap over test examples
alone gave p~1e-45, which overstates how far the result generalizes).

This resamples BOTH the subset axis and the test-example axis together, and
recomputes each column's Spearman correlation from the RESAMPLED masks/
margins each time (not from a lookup of precomputed r values) -- the masks/
margins/attr arrays needed for this are in exp35's *_raw.npz files.
"""
from __future__ import annotations

import numpy as np


def _rank_columns(M):
    """Rank each column of M independently (argsort-based, no tie
    correction -- fine here since attribution/margin scores are continuous
    floats where exact ties are essentially impossible). Fully vectorized:
    no Python loop over columns, unlike scipy.stats.rankdata called
    per-column, which dominates runtime across thousands of bootstrap
    replicates."""
    order = np.argsort(M, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(M.shape[0])[:, None].astype(np.float64), axis=0)
    return ranks


def spearman_columns(A, B):
    """Per-column Spearman correlation between two (n_subsets, n_test)
    matrices, vectorized via rank-transform + Pearson formula."""
    Ar = _rank_columns(A)
    Br = _rank_columns(B)
    Ar -= Ar.mean(axis=0, keepdims=True)
    Br -= Br.mean(axis=0, keepdims=True)
    num = (Ar * Br).sum(axis=0)
    den = np.sqrt((Ar ** 2).sum(axis=0) * (Br ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def two_way_bootstrap(masks, margins, attr_a, attr_b, n_boot=2000, seed=0):
    """masks: (n_subsets, n_train). margins: (n_subsets, n_test).
    attr_a, attr_b: (n_test, n_train) attribution score matrices for the two
    methods being compared. Returns (observed_diff, boot_diffs) where
    boot_diffs has shape (n_boot,) -- take percentiles for a CI."""
    n_subsets, n_train = masks.shape
    n_test = margins.shape[1]
    assert attr_a.shape == (n_test, n_train) and attr_b.shape == (n_test, n_train)

    def mean_lds_diff(subset_idx, test_idx):
        masks_b = masks[subset_idx]  # (n_subsets, n_train)
        margins_b = margins[subset_idx]  # (n_subsets, n_test)
        pred_a = masks_b @ attr_a.T  # (n_subsets, n_test)
        pred_b = masks_b @ attr_b.T
        r_a = spearman_columns(pred_a, margins_b)  # (n_test,)
        r_b = spearman_columns(pred_b, margins_b)
        r_a_sel, r_b_sel = r_a[test_idx], r_b[test_idx]
        valid = ~(np.isnan(r_a_sel) | np.isnan(r_b_sel))
        return float(np.mean(r_a_sel[valid]) - np.mean(r_b_sel[valid]))

    observed_diff = mean_lds_diff(np.arange(n_subsets), np.arange(n_test))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        subset_idx = rng.integers(0, n_subsets, size=n_subsets)
        test_idx = rng.integers(0, n_test, size=n_test)
        boot_diffs[i] = mean_lds_diff(subset_idx, test_idx)

    return observed_diff, boot_diffs


def report(npz_path, name_a, name_b, n_boot=2000, seed=0):
    d = np.load(npz_path)
    attr_a = d[f"attr_{name_a}"]
    attr_b = d[f"attr_{name_b}"]
    masks, margins = d["masks"], d["margins"]
    observed, boot_diffs = two_way_bootstrap(masks, margins, attr_a, attr_b, n_boot, seed)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"{npz_path}: {name_a} - {name_b}")
    print(f"  observed mean LDS difference: {observed:+.4f}")
    print(f"  95% CI (two-way bootstrap, {n_boot} resamples): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  excludes zero: {ci_lo > 0 or ci_hi < 0}")
    return observed, ci_lo, ci_hi


if __name__ == "__main__":
    import sys
    npz_path, name_a, name_b = sys.argv[1], sys.argv[2], sys.argv[3]
    n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    report(npz_path, name_a, name_b, n_boot)
