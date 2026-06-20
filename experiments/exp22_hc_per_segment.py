"""Per-segment for Home Credit — needed for T1 nuance."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from exp22_sab_benchmark import (
    gen_home_credit, baseline_grad_argmax, ss_attrib_block, metrics,
)

agg = {"gmag": defaultdict(list), "ss": defaultdict(list)}
for seed in range(10, 30):
    Xtr, ytr, stt, Xte, yte, ste, sc, sn = gen_home_credit(seed)
    gmag_p = baseline_grad_argmax(Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    ss_p   = ss_attrib_block(    Xtr, ytr, stt, Xte, yte, sn, sc, seed)
    for n, p in [("gmag", gmag_p), ("ss", ss_p)]:
        for seg, v in metrics(p, ste)["per_segment"].items():
            agg[n][seg].append(v)

print(f"{'segment':24s} {'gmag':>8} {'ss':>8} {'Δ':>8}")
for seg in ("application", "previous_application", "bureau"):
    g = float(np.mean(agg["gmag"][seg])); s = float(np.mean(agg["ss"][seg]))
    print(f"  {seg:22s} {g:>8.3f} {s:>8.3f} {s-g:>+8.3f}")

out = {seg: {"gmag_mean_p1": float(np.mean(agg["gmag"][seg])),
             "ss_mean_p1":   float(np.mean(agg["ss"][seg]))}
       for seg in ("application", "previous_application", "bureau")
       if agg["gmag"][seg]}
with open("results/exp22_hc_per_segment.json", "w") as f:
    json.dump(out, f, indent=2)
