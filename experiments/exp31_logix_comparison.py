"""exp31 — Head-to-head vs. LogIX (Choe et al., github.com/logix-project/logix).

Reviewer item 12: getting per-sample gradients from forward/backward hooks is
the same trick LogIX and Opacus use. This measures LogIX's own per-step
training overhead, hooked into the SAME tiny-clf training loop and SAME
tracked-module scope (last-block LoRA adapters + head) used by exp30's
LoRAGradientLogger measurement, so the two overhead numbers are directly
comparable rather than each being quoted from a different paper's setup.

Requires logix-ai, which caps at python<3.11:
    pip install logix-ai   (or: pip install git+https://github.com/logix-project/logix.git)

Run with a Python 3.10 (or earlier) interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from exp27_lds_quality import build_tiny_classifier, synthetic_data


def run(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import logix

    device = args.device
    torch.manual_seed(1234)
    np.random.seed(1234)
    model = build_tiny_classifier(args.vocab, seq=args.seq, r=args.rank).to(device)

    Xtr, ytr = synthetic_data(args.n_train, args.seq, args.vocab, seed=0)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)

    # Same scope as exp30/exp27 "last_block": last transformer block's LoRA
    # adapters + the classification head. Exclude the frozen `.base` Linear
    # layers -- Traceprop never tracks those either, only the small adapters.
    n_blocks = len(model.blocks)
    last_block_idx = n_blocks - 1
    tracked_names = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and (n == "score" or (f"blocks.{last_block_idx}." in n and ("lora_A" in n or "lora_B" in n)))
    ]
    print(f"[exp31] tracking {len(tracked_names)} modules: {tracked_names}")

    def loss_fn(xb, yb):
        return F.cross_entropy(model(xb), yb, reduction="sum")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    def batch(step):
        s = (step * args.batch) % (args.n_train - args.batch)
        return Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch], s

    def build_run():
        run_ = logix.init(project=f"exp31_{os.getpid()}", config="exp31_config.yaml")
        run_.watch(model, name_filter=tracked_names, type_filter=[nn.Linear])
        run_.setup({"grad": ["log"]})
        run_.save(False)  # keep logs in memory; don't pay disk I/O in the timing
        return run_

    lrun = build_run()
    for step in range(args.warmup):
        xb, yb, s = batch(step)
        opt.zero_grad(set_to_none=True)
        with lrun(data_id=[str(s + i) for i in range(len(xb))]):
            loss_fn(xb, yb).backward()
        opt.step()

    def block(use_logix):
        t0 = time.perf_counter()
        for step in range(args.steps):
            xb, yb, s = batch(step)
            opt.zero_grad(set_to_none=True)
            if use_logix:
                with lrun(data_id=[str(s + i) for i in range(len(xb))]):
                    loss_fn(xb, yb).backward()
            else:
                loss_fn(xb, yb).backward()
            opt.step()
        return time.perf_counter() - t0

    overheads, base_times = [], []
    for _ in range(args.repeats):
        b = block(False)
        i = block(True)
        overheads.append((i - b) / b * 100.0)
        base_times.append(b)

    med = statistics.median(overheads)
    std = statistics.pstdev(overheads) if len(overheads) > 1 else 0.0
    base_step_ms = statistics.median(base_times) / args.steps * 1e3

    out = {
        "tool": "logix (logix-project/logix)",
        "model": "tiny-clf (exp27 from-scratch-LoRA quality model)",
        "device": device,
        "n_train": args.n_train, "seq": args.seq, "rank": args.rank,
        "tracked_modules": tracked_names,
        "steps": args.steps, "repeats": args.repeats,
        "base_step_ms": round(base_step_ms, 4),
        "overhead_pct_median": round(med, 3),
        "overhead_pct_std": round(std, 3),
        "overhead_samples": [round(x, 3) for x in overheads],
        "note": "directly comparable to results/exp30_same_setting_overhead.json "
                "(LoRAGradientLogger, same model/scope/methodology).",
    }
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/exp31_logix_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/exp31_logix_comparison.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
