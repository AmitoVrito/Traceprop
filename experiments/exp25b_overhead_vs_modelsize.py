"""exp25b — Inline-logging overhead as a function of model compute.

The absolute overhead of Traceprop's inline per-sample logging is a roughly
fixed per-step cost (form per-sample outer products on the tracked LoRA
linears, one on-device JL projection). As the *model* grows, the training
step's forward/backward dominates and the relative overhead falls toward the
sub-1% regime that matters for real fine-tunes.

This sweep runs the self-contained tiny transformer at increasing width/depth
on CPU and records (baseline step time, overhead %). It demonstrates the trend
without a GPU; the absolute headline number comes from exp25 --backend hf on a
real GPT-2/Pythia fine-tune where a single step is 10^2-10^3x heavier than the
toy model's, pushing overhead below 1%.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from exp25_llm_inline_overhead import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--proj_dim", type=int, default=512)
    args = ap.parse_args()

    # (width d, depth n_blocks) from toy -> approaching a small real model
    configs = [
        (128, 2),
        (256, 2),
        (384, 4),
        (512, 6),
        (768, 6),
    ]
    rows = []
    for d, nb in configs:
        ns = SimpleNamespace(
            backend="tiny", model="gpt2", device="cpu",
            steps=args.steps, warmup=3, batch=args.batch, seq=args.seq,
            rank=8, proj_dim=args.proj_dim, d=d, n_blocks=nb, track=0,
        )
        r = run(ns)
        rows.append({
            "d": d, "n_blocks": nb,
            "baseline_ms": round(r["baseline_median_s"] * 1e3, 3),
            "overhead_pct": r["overhead_pct"],
            "grad_dim": r["per_sample_grad_dim"],
        })

    print("\n=== overhead vs model size ===")
    print(f"{'d':>5} {'blocks':>7} {'base_ms':>9} {'overhead%':>10}")
    for row in rows:
        print(f"{row['d']:>5} {row['n_blocks']:>7} {row['baseline_ms']:>9} {row['overhead_pct']:>10}")

    os.makedirs("results", exist_ok=True)
    with open("results/exp25b_overhead_vs_modelsize.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved -> results/exp25b_overhead_vs_modelsize.json")


if __name__ == "__main__":
    main()
