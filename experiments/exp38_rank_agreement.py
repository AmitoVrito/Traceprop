"""Rank-agreement proxy at 1B (review item 9).

Reports how closely Traceprop and LogIX-random rank the same training examples
for the same test queries, without any retraining: for each test query it
correlates the two methods' per-query influence vectors over the training set,
with Spearman rho and Kendall tau, and reports mean and std over queries.

IMPORTANT: the gradient stores from the headline Pythia-1B run were NOT saved to
disk (only the aggregate timing JSON was kept). So this must be run against two
freshly produced influence-score matrices, each of shape (n_test, n_train):

  traceprop_scores.npz : key "scores" = Traceprop factored + inline-precond
  logix_scores.npz     : key "scores" = LogIX random-init, tuned, preconditioned

Both matrices must be scored over the SAME train set and the SAME N test queries,
in the same row/column order. Produce them with one scoring pass per method on
the 1B config (no retraining needed for rank agreement), e.g. via the HF path of
exp35_logix_lds.py with score-matrix dumping enabled, then:

  python exp38_rank_agreement.py \
      --traceprop_scores results/traceprop_1b_scores.npz \
      --logix_scores     results/logix_random_1b_scores.npz \
      --out results/exp38_rank_agreement_pythia1b.json

Nothing is fabricated: the script only reports correlations of matrices it is
given. If the matrices do not exist yet, it exits and prints the rerun note.
"""
import argparse
import json
import os

import numpy as np


def per_query_rank_agreement(A, B):
    """A, B: (n_test, n_train) influence matrices from the two methods.
    Returns per-query Spearman and Kendall tau arrays (length n_test)."""
    from scipy.stats import spearmanr, kendalltau
    assert A.shape == B.shape, f"shape mismatch {A.shape} vs {B.shape}"
    n_test = A.shape[0]
    rho = np.empty(n_test)
    tau = np.empty(n_test)
    for q in range(n_test):
        rho[q] = spearmanr(A[q], B[q]).correlation
        tau[q] = kendalltau(A[q], B[q]).correlation
    return rho, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceprop_scores", required=True)
    ap.add_argument("--logix_scores", required=True)
    ap.add_argument("--key", default="scores")
    ap.add_argument("--out", default="results/exp38_rank_agreement.json")
    args = ap.parse_args()

    for p in (args.traceprop_scores, args.logix_scores):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}. The headline 1B stores were not saved; produce the two "
                "(n_test, n_train) score matrices with a 1B scoring pass per method first "
                "(see this file's header for the command), then rerun.")

    A = np.load(args.traceprop_scores)[args.key]
    B = np.load(args.logix_scores)[args.key]
    rho, tau = per_query_rank_agreement(A, B)

    res = {
        "n_test_queries": int(A.shape[0]),
        "n_train": int(A.shape[1]),
        "spearman_rho_mean": round(float(np.nanmean(rho)), 4),
        "spearman_rho_std": round(float(np.nanstd(rho)), 4),
        "kendall_tau_mean": round(float(np.nanmean(tau)), 4),
        "kendall_tau_std": round(float(np.nanstd(tau)), 4),
        "traceprop_scores": args.traceprop_scores,
        "logix_scores": args.logix_scores,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
