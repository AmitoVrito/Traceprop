"""Robust overhead of inline K-FAC covariance accumulation (paper claim check).

The single-shot overhead numbers exp35 prints are noisy (sub-second passes on a
loaded CPU). This measures the SAME factored logging pass with covariance
accumulation OFF vs ON, repeated many times, and reports the median relative
overhead with an IQR -- the honest number behind "K-FAC covariance is built in
the same pass, no second sweep, at negligible cost".

Run: python exp35_inlineprecond_overhead.py [--kfac 8] [--repeats 25]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from exp27_lds_quality import build_tiny_classifier, synthetic_data
from traceprop.attribution.gradient_store import GradientStore
from traceprop.llm import LoRAGradientLogger, select_lora_linears


def time_pass(model, X, y, targets, kfac, inline_precond, batch, logits_fn):
    store = GradientStore(proj_dim=512, seed=42)
    lg = LoRAGradientLogger(store, targets, proj_dim=512, factored=True,
                            kfac=kfac, inline_precond=inline_precond)
    t0 = time.perf_counter()
    for s in range(0, len(X), batch):
        xb, yb = X[s:s + batch], y[s:s + batch]
        model.zero_grad(set_to_none=True)
        F.cross_entropy(logits_fn(model, xb), yb, reduction="sum").backward()
        lg.flush_step(sample_indices=range(s, s + len(xb)))
    dt = time.perf_counter() - t0
    lg.detach()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfac", type=int, default=8)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--out", default="results/exp35_inlineprecond_overhead.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    X, y = synthetic_data(args.n_train, args.seq, args.vocab, 0)
    X, y = torch.tensor(X), torch.tensor(y)
    model = build_tiny_classifier(args.vocab, seq=args.seq, r=8, n_blocks=2)
    model.train()
    logits_fn = lambda m, xb: m(xb)
    targets = select_lora_linears(model, ("lora_A", "lora_B", "score", "classifier"))
    print(f"[overhead] {len(targets)} tracked layers, kfac={args.kfac}, "
          f"n_train={args.n_train}, batch={args.batch}, {args.repeats} repeats")

    # interleave off/on to average out CPU thermal/scheduler drift; drop a warmup
    off, on = [], []
    for r in range(args.repeats + 1):
        d_off = time_pass(model, X, y, targets, args.kfac, False, args.batch, logits_fn)
        d_on = time_pass(model, X, y, targets, args.kfac, True, args.batch, logits_fn)
        if r == 0:
            continue  # warmup
        off.append(d_off)
        on.append(d_on)
    off, on = np.array(off), np.array(on)
    rel = (on - off) / off * 100.0

    res = {
        "kfac": args.kfac, "n_tracked_layers": len(targets),
        "n_train": args.n_train, "batch": args.batch, "repeats": args.repeats,
        "off_ms_median": round(float(np.median(off)) * 1e3, 2),
        "on_ms_median": round(float(np.median(on)) * 1e3, 2),
        "overhead_pct_median": round(float(np.median(rel)), 3),
        "overhead_pct_iqr": [round(float(np.percentile(rel, 25)), 3),
                             round(float(np.percentile(rel, 75)), 3)],
        "note": "off = factored logging pass, covariance accumulation OFF; on = SAME "
                "pass with inline K-FAC covariance of the projected factors ON. Both "
                "are single passes over the training set -- 'on' needs no separate "
                "covariance sweep, unlike LogIX. Interleaved measurement, warmup dropped.",
    }
    print(json.dumps(res, indent=2))
    print(f"\n[overhead] median: {res['off_ms_median']}ms -> {res['on_ms_median']}ms "
          f"= {res['overhead_pct_median']:+.2f}% "
          f"(IQR {res['overhead_pct_iqr'][0]:+.2f}%..{res['overhead_pct_iqr'][1]:+.2f}%)")
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
