"""exp30 — Inline logging overhead, measured on the SAME model that produced
the paper's from-scratch-LoRA quality number (LDS 0.51/0.50, exp27 tiny-clf).

Reviewer finding: the paper's overhead numbers (exp25/exp26) come from LoRA
fine-tunes of pretrained GPT-2/Pythia, where the paper's own scope statement
says attribution quality is near-zero for every method. The paper's quality
numbers for the regime where attribution actually works (frozen backbone;
from-scratch LoRA transformer) come from a completely different model
(exp27/28's small from-scratch classifier). No single experiment shows both
overhead and quality in the same setting.

This script times inline-logging overhead (exp25's interleaved,
per-step-synced-vs-throughput methodology) on exactly the `tiny-clf` model
and training config used by exp27's from-scratch-LoRA quality measurement,
so overhead and quality can finally be quoted from the same run.
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
    import torch.nn.functional as F

    from traceprop.attribution.gradient_store import GradientStore
    from traceprop.llm import LoRAGradientLogger, select_lora_linears

    device = args.device
    torch.manual_seed(1234)
    np.random.seed(1234)
    model = build_tiny_classifier(args.vocab, seq=args.seq, r=args.rank).to(device)

    Xtr, ytr = synthetic_data(args.n_train, args.seq, args.vocab, seed=0)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)

    HEAD = ("score", "classifier")
    patterns = ("lora_A", "lora_B") + HEAD
    last_n = None if args.track <= 0 else args.track

    def loss_fn(xb, yb):
        return F.cross_entropy(model(xb), yb, reduction="sum")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    def build_logger():
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        logger = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)
        return logger

    def batch(step):
        s = (step * args.batch) % (args.n_train - args.batch)
        return Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch], s

    logger = build_logger()
    for step in range(args.warmup):
        xb, yb, s = batch(step)
        opt.zero_grad(set_to_none=True)
        loss_fn(xb, yb).backward()
        logger.flush_step(sample_indices=range(s, s + args.batch))
        opt.step()

    def block(use_logger):
        t0 = time.perf_counter()
        for step in range(args.steps):
            xb, yb, s = batch(step)
            opt.zero_grad(set_to_none=True)
            loss_fn(xb, yb).backward()
            if use_logger:
                logger.flush_step(sample_indices=range(s, s + args.batch))
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
        "model": "tiny-clf (exp27 from-scratch-LoRA quality model)",
        "device": device,
        "n_train": args.n_train, "seq": args.seq, "rank": args.rank,
        "proj_dim": args.proj_dim, "track_last_n_blocks": args.track,
        "steps": args.steps, "repeats": args.repeats,
        "base_step_ms": round(base_step_ms, 4),
        "overhead_pct_median": round(med, 3),
        "overhead_pct_std": round(std, 3),
        "overhead_samples": [round(x, 3) for x in overheads],
        "note": "same model/config as the from-scratch-LoRA quality number "
                "(LDS 0.51 last-block / 0.50 all-layers, exp27 tiny-clf).",
    }
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/exp30_same_setting_overhead.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/exp30_same_setting_overhead.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
