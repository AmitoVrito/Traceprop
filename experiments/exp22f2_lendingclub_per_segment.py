"""Per-segment P@1 breakdown for Lending Club — T1/T6 in v7 review."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from exp22_sab_benchmark import (
    baseline_grad_argmax, ss_attrib_block, metrics,
)
import exp22f_lendingclub as e

per_seg_aggregate = {"gmag": defaultdict(list), "ss": defaultdict(list)}
for seed in range(10, 30):
    Xtr, ytr, stt, Xte, yte, ste, sc, sn = e.gen_seed(seed)
    gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    ss_p   = ss_attrib_block(    Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    for method_name, preds in [("gmag", gmag_p), ("ss", ss_p)]:
        m = metrics(preds, ste)
        for seg, p1 in m["per_segment"].items():
            per_seg_aggregate[method_name][seg].append(p1)

print("Per-segment P@1 (mean over 20 seeds):")
print(f"{'segment':22s} {'gmag':>8} {'ss':>8} {'Δ':>8}")
for seg in ("borrower", "credit_hist", "loan_terms"):
    g_list = per_seg_aggregate["gmag"][seg]
    s_list = per_seg_aggregate["ss"][seg]
    if not g_list:
        continue
    g = float(np.mean(g_list)); s = float(np.mean(s_list))
    print(f"  {seg:20s} {g:>8.3f} {s:>8.3f} {s-g:>+8.3f}")

out = {seg: {"gmag_mean_p1": float(np.mean(per_seg_aggregate["gmag"][seg])),
             "ss_mean_p1":   float(np.mean(per_seg_aggregate["ss"][seg]))}
       for seg in ("borrower", "credit_hist", "loan_terms")
       if per_seg_aggregate["gmag"][seg]}
with open("results/exp22f2_lendingclub_per_segment.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp22f2_lendingclub_per_segment.json")
