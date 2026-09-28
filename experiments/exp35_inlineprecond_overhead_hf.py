"""Run B: in-TRAINING overhead of factored logging + inline K-FAC covariance on a
real HF model (e.g. Pythia-1B) on GPU. This is the honest "no second pass" number
for the paper title; the tiny-CPU number is only a loose upper bound.

Measures three configs in a real LoRA training loop (forward/backward/opt.step),
interleaved with a dropped warmup:
  (a) train only, NO attribution logging                 -> baseline
  (b) train + factored logging, inline_precond OFF       -> logging cost
  (c) train + factored logging, inline_precond ON        -> logging + covariance

Reports:
  total_overhead_pct   = (c - a)/a   <- decides the paper framing ("~X% in-training")
  logging_only_pct     = (b - a)/a
  covariance_only_pct  = (c - b)/b   <- the marginal cost of inline K-FAC vs plain logging

Colab A100 usage (Pythia-1B):
  python exp35_inlineprecond_overhead_hf.py --model EleutherAI/pythia-1b --device cuda \
     --steps 200 --batch 8 --seq 128 --track 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from exp27_lds_quality import build_hf_classifier
from traceprop.attribution.gradient_store import GradientStore
from traceprop.llm import LoRAGradientLogger, select_lora_linears


def build(model_name, rank, device):
    torch.manual_seed(0)
    m = build_hf_classifier(model_name, r=rank).to(device)
    return m


def run_loop(model, batches, mode, kfac, track, device):
    """mode in {'none','log','log_cov'}. Returns wall-clock seconds for the loop."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    lg = None
    if mode in ("log", "log_cov"):
        patterns = ("lora_A", "lora_B", "score", "classifier")
        last_n = None if track <= 0 else track
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        store = GradientStore(proj_dim=512, seed=42)
        lg = LoRAGradientLogger(store, targets, proj_dim=512, factored=True,
                                kfac=kfac, inline_precond=(mode == "log_cov"))
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for xb, yb in batches:
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        logits = out if not hasattr(out, "logits") else out.logits
        F.cross_entropy(logits, yb, reduction="mean").backward()
        if lg is not None:
            lg.flush_step(buffer=True)   # buffer=True keeps the device->host copy off the step
        opt.step()
    if lg is not None:
        lg.drain()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if lg is not None:
        lg.detach()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--kfac", type=int, default=8)
    ap.add_argument("--track", type=int, default=0, help="0 = all blocks")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3, help="interleaved repeats + 1 warmup")
    ap.add_argument("--out", default="results/exp35_inlineprecond_overhead_hf.json")
    args = ap.parse_args()

    device = args.device
    model = build(args.model, args.rank, device)
    n_tracked = len(select_lora_linears(
        model, ("lora_A", "lora_B", "score", "classifier"),
        last_n_blocks=(None if args.track <= 0 else args.track)))
    # fixed synthetic token batches (overhead is data-independent; avoids a dataset download)
    g = torch.Generator().manual_seed(0)
    batches = [(torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device),
                torch.randint(0, 2, (args.batch,), generator=g).to(device))
               for _ in range(args.steps)]

    print(f"[overhead-hf] {args.model} on {device}, {n_tracked} tracked layers, kfac={args.kfac}, "
          f"{args.steps} steps x batch {args.batch} x seq {args.seq}, {args.repeats} repeats")

    a, b, c = [], [], []
    for r in range(args.repeats + 1):
        d_a = run_loop(model, batches, "none", args.kfac, args.track, device)
        d_b = run_loop(model, batches, "log", args.kfac, args.track, device)
        d_c = run_loop(model, batches, "log_cov", args.kfac, args.track, device)
        if r == 0:
            continue  # warmup
        a.append(d_a); b.append(d_b); c.append(d_c)
        print(f"  repeat {r}: none={d_a:.3f}s log={d_b:.3f}s log_cov={d_c:.3f}s")
    a, b, c = map(np.array, (a, b, c))

    res = {
        "model": args.model, "device": device, "n_tracked_layers": n_tracked,
        "kfac": args.kfac, "track": args.track, "steps": args.steps,
        "batch": args.batch, "seq": args.seq, "repeats": args.repeats,
        "none_s_median": round(float(np.median(a)), 4),
        "log_s_median": round(float(np.median(b)), 4),
        "log_cov_s_median": round(float(np.median(c)), 4),
        "total_overhead_pct": round(float(np.median((c - a) / a) * 100), 3),
        "logging_only_pct": round(float(np.median((b - a) / a) * 100), 3),
        "covariance_only_pct": round(float(np.median((c - b) / b) * 100), 3),
        "note": "total_overhead_pct = full attribution cost during training (factored logging + "
                "inline K-FAC covariance) vs plain training; covariance_only_pct = marginal cost "
                "of inline covariance over plain logging. Single training pass; LogIX needs a "
                "SEPARATE covariance pass on top of its logging pass.",
    }
    print(json.dumps(res, indent=2))
    print(f"\n[overhead-hf] TOTAL in-training overhead (log+cov vs none): "
          f"{res['total_overhead_pct']:+.2f}%  |  covariance-only marginal: "
          f"{res['covariance_only_pct']:+.2f}%")
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
