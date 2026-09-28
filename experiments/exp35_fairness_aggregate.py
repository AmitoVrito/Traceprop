"""Aggregate the multi-seed inline-precond fairness campaign into one verdict.

Reads results/exp35_tiny_track0_fair_seed{S}.json (+ _bootstrap.json) for the
given seeds and produces:
  - per-seed held-out eval LDS (inline vs LogIX_tuned vs LogIX_default vs factored dot)
  - per-seed two-way bootstrap CI of (inline - LogIX_tuned), kfac 7 & 8
  - Check 2 evidence: chosen inline damping per seed + whether it sits at the grid
    edge (cliff risk) or interior (peak), plus the val-LDS curve
  - LogIX default damping + chosen tuned lambda per seed
  - final cross-seed decision (fixed rule):
      majority of seeds CI>0  -> "outperforms"
      else                    -> "matches LogIX's best quality in one pass, not two"

Usage: python exp35_fairness_aggregate.py [--seeds 0 1 2 3 4] [--kfac 8]
"""
import argparse
import glob
import json
import os


def load(seed, base_stem):
    base = f"results/{base_stem}_seed{seed}"
    with open(base + ".json") as f:
        main = json.load(f)
    with open(base + "_bootstrap.json") as f:
        boot = json.load(f)
    return main, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--prefix", default="fair",
                    help="short tag for the tiny-backend campaigns (fair | fairpm); expands to "
                         "base stem exp35_tiny_track0_<prefix>. Ignored if --base is given.")
    ap.add_argument("--base", default=None,
                    help="full filename stem before _seed{S} (e.g. exp35_pythia160m_sst2). "
                         "Overrides --prefix; use for non-tiny campaigns.")
    ap.add_argument("--kfac", type=int, default=8, help="kfac bracket to headline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base_stem = args.base or f"exp35_tiny_track0_{args.prefix}"
    if args.out is None:
        args.out = f"results/{base_stem}_summary_kfac{args.kfac}.json"

    seeds = args.seeds
    if seeds is None:
        seeds = sorted(int(f.split("seed")[-1].split(".json")[0])
                       for f in glob.glob(f"results/{base_stem}_seed*.json")
                       if "_bootstrap" not in f)
    kf = args.kfac
    ip = f"traceprop_factored_kfac{kf}_inlineprecond"
    fd = f"traceprop_factored_kfac{kf}_dot"

    rows, per_seed = [], {}
    above = matches = below = 0
    grid_extended = ["1e-06", "1e-05", "0.0001"]

    for s in seeds:
        try:
            main_j, boot = load(s, base_stem)
        except FileNotFoundError:
            print(f"[seed {s}] missing files, skipping")
            continue
        he = main_j["lds_heldout_eval"]
        lg_tuned = main_j.get("logix_tuned_selection", {})
        ip_sel = main_j.get("inline_precond_selection", {}).get(str(kf), {})
        dbase = boot.get("decision_baseline", "logix_preconditioned")
        comp = boot["comparisons"].get(f"kfac{kf}: inlineprecond - {dbase}", {})
        lo, hi = comp.get("ci95", [None, None])
        chosen_damp = ip_sel.get("chosen_damping")
        grid = ip_sel.get("grid_val_lds", {})
        # peak vs cliff: is the chosen damping at the smallest grid point?
        at_edge = (grid and min(float(k) for k in grid) == float(chosen_damp)) if chosen_damp else None

        verdict = ("ABOVE" if (lo is not None and lo > 0)
                   else "BELOW" if (hi is not None and hi < 0) else "OVERLAP")
        above += verdict == "ABOVE"; matches += verdict == "OVERLAP"; below += verdict == "BELOW"

        per_seed[s] = {
            "heldout_eval_lds": {
                "inline_precond": he.get(ip, {}).get("mean"),
                "logix_preconditioned_tuned": he.get("logix_preconditioned_tuned", {}).get("mean"),
                "logix_preconditioned_default": he.get("logix_preconditioned", {}).get("mean"),
                "logix_dot": he.get("logix_dot", {}).get("mean"),
                "factored_dot": he.get(fd, {}).get("mean"),
            },
            "bootstrap_inline_minus_tuned": {
                "decision_baseline": dbase, "lds_diff": comp.get("lds_diff"),
                "ci95": [lo, hi], "verdict": verdict,
            },
            "inline_chosen_damping": chosen_damp, "inline_damping_at_grid_edge": at_edge,
            "inline_val_grid": grid,
            "logix_default_damping_abs": lg_tuned.get("logix_default_damping_abs_nominal"),
            "logix_tuned_chosen_lambda_rel": lg_tuned.get("chosen_lambda_rel"),
        }
        rows.append((s, verdict, comp.get("lds_diff"), lo, hi, chosen_damp, at_edge,
                     lg_tuned.get("chosen_lambda_rel")))

    n = above + matches + below
    if n and above > n / 2:
        final = (f"OUTPERFORMS: inline-precond CI > 0 on {above}/{n} seeds (majority) vs "
                 f"LogIX TUNED -> claim 'outperforms LogIX's best preconditioning'.")
    elif n:
        final = (f"MATCHES: inline-precond CI overlaps/loses on {matches+below}/{n} seeds vs "
                 f"LogIX TUNED -> claim 'matches LogIX's best quality in ONE pass instead of two'.")
    else:
        final = "NO DATA"

    any_edge = any(r[6] for r in rows if r[6] is not None)

    print(f"\n=== Inline-precond fairness summary (kfac={kf}), {n} seeds ===")
    print(f"{'seed':>4} {'verdict':>8} {'diff':>9} {'CI_lo':>9} {'CI_hi':>9} "
          f"{'inl_damp':>9} {'edge?':>6} {'lgx_lam':>8}")
    for s, v, d, lo, hi, cd, edge, ll in rows:
        print(f"{s:>4} {v:>8} {d:>+9.4f} {lo:>+9.4f} {hi:>+9.4f} "
              f"{cd:>9g} {str(edge):>6} {ll:>8g}")
    print(f"\nvotes: ABOVE={above} OVERLAP={matches} BELOW={below}")
    if any_edge:
        print("WARNING (Check 2): some seeds chose the SMALLEST grid damping -> "
              "possible cliff, extend grid lower and re-check.")
    else:
        print("Check 2 OK: no seed sits at the grid edge -> peak is interior, not a cliff.")
    print(f"\n[FINAL DECISION] {final}")

    summary = {"seeds": seeds, "kfac": kf, "per_seed": per_seed,
               "votes": {"above": above, "overlap": matches, "below": below},
               "any_seed_at_grid_edge": any_edge, "final_decision": final}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
