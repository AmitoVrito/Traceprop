"""exp31 — Head-to-head vs. LogIX (Choe et al., github.com/logix-project/logix).

Reviewer item 12: getting per-sample gradients from forward/backward hooks is
the same trick LogIX and Opacus use. This measures LogIX's own per-step
training overhead, hooked into the SAME training loop and SAME tracked-module
scope (last transformer block's LoRA adapters) used by exp25/exp30's
LoRAGradientLogger measurement, so the two overhead numbers are directly
comparable rather than each being quoted from a different paper's setup.

Two backends:
  --backend tiny  Small from-scratch classifier (exp27's tiny-clf, CPU). Ran
                  already: LogIX 12.9% +/- 19.3%, LoRAGradientLogger 25.9% +/-
                  14.7% (n=100 repeats each). Held pending this backend's GPU
                  number before writing anything into the paper -- that CPU
                  result is the opposite regime from the paper's ~1% claim.
  --backend hf    GPT-2 / Pythia + PEFT LoRA, matching exp25 exactly (same
                  model, same tracked scope, same batch/seq). This is the
                  number that actually matters for the paper.

Requires logix-ai, which caps at python<3.11:
    pip install logix-ai   (or: pip install git+https://github.com/logix-project/logix.git)
Run with a Python 3.10 (or earlier) interpreter. On Colab, use
!pip -q install "logix-ai" (Colab's default Python is 3.10/3.11 -- check
`python --version`; if it's 3.11, this will fail the same way it does here).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from exp25_llm_inline_overhead import build_tiny_model, build_hf_model, hf_batch, tiny_batch
from exp27_lds_quality import build_tiny_classifier, synthetic_data


def run(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import logix

    device = args.device

    if args.backend == "tiny":
        torch.manual_seed(1234)
        np.random.seed(1234)
        model = build_tiny_classifier(args.vocab, seq=args.seq, r=args.rank).to(device)
        Xtr, ytr = synthetic_data(args.n_train, args.seq, args.vocab, seed=0)
        Xtr_t = torch.tensor(Xtr, device=device)
        ytr_t = torch.tensor(ytr, device=device)

        n_blocks = len(model.blocks)
        last_block_idx = n_blocks - 1
        tracked_names = [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear)
            and (n == "score" or (f"blocks.{last_block_idx}." in n and ("lora_A" in n or "lora_B" in n)))
        ]

        def loss_fn(xb, yb):
            return F.cross_entropy(model(xb), yb, reduction="sum")

        def batch(step):
            s = (step * args.batch) % (args.n_train - args.batch)
            return Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch], s

    else:  # hf: exact same model/scope as exp25
        model = build_hf_model(args.model, r=args.rank).to(device)
        x = hf_batch(args.model, args.seq, args.batch, device)
        last_n = None if args.track <= 0 else args.track
        tracked_names = [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n)
        ]
        if args.track > 0:
            # keep only last-N-block adapters, matching exp25's select_lora_linears(last_n_blocks=...)
            import re
            def block_idx(name):
                m = re.search(r"(?:^|\.)(?:h|layers)\.(\d+)\.", name)
                return int(m.group(1)) if m else None
            idxs = sorted({block_idx(n) for n in tracked_names if block_idx(n) is not None})
            keep = set(idxs[-args.track:])
            tracked_names = [n for n in tracked_names if block_idx(n) in keep]

        def loss_fn(xb, yb=None):
            logits = model(xb).logits
            return F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                xb[:, 1:].reshape(-1),
            )

        def batch(step):
            return x, None, step * args.batch

    print(f"[exp31] backend={args.backend} tracking {len(tracked_names)} modules")

    # --- Storage matching -------------------------------------------------
    # Ours: a single dense sparse-JL projection to proj_dim floats, total
    # args.proj_dim * 4 bytes/example regardless of tracked-layer count.
    # LogIX: a rank x rank core matrix PER tracked layer (LoraLinear inserts
    # A: in->rank, B: rank->rank, C: rank->out; the per-example-varying part
    # is the B core), so its per-example storage is
    # n_layers * rank^2 * 4 bytes and grows with both rank AND layer count.
    # Its own default (rank=64) with our 6-layer GPT-2 scope would store
    # ~96KB/example -- ~48x our 2KB budget -- which would make "LogIX is
    # slower" partly a "LogIX is doing more work by default" result, not a
    # mechanism result. Solve for the rank that puts LogIX at the same
    # per-example budget we report elsewhere in the paper (2KB) instead.
    our_bytes_per_example = args.proj_dim * 4
    n_tracked_layers = len(tracked_names)
    matched_rank = max(1, int((our_bytes_per_example / (4 * n_tracked_layers)) ** 0.5))
    logix_bytes_per_example = n_tracked_layers * (matched_rank ** 2) * 4
    print(f"[exp31] storage matching: ours={our_bytes_per_example}B/example "
          f"({args.proj_dim} floats), LogIX rank set to {matched_rank} for "
          f"{n_tracked_layers} tracked layers -> {logix_bytes_per_example}B/example "
          f"(analytical -- not yet verified against LogIX's actual serialized log size)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    def build_run(save_to_disk):
        # logix.init() is a process-level singleton; use LogIX(...) directly so we
        # can build a second instance for the second (matched-buffering) config.
        run_ = logix.LogIX(project=f"exp31_{os.getpid()}_{save_to_disk}", config="exp31_config.yaml")
        run_.config.lora.rank = matched_rank  # storage-matched, see above
        run_.watch(model, name_filter=tracked_names, type_filter=[nn.Linear])
        # {"grad": ["log"]} only -- deliberately NOT requesting "covariance" or
        # "hessian" statistics. LogIX's Hessian/EK-FAC machinery is opt-in via
        # those keys; omitting them means no covariance accumulation happens
        # during this timed comparison, so it isn't silently doing extra work
        # Traceprop doesn't do. (Not yet independently verified by inspecting
        # LogIX's internal state after a run -- flagging the assumption.)
        run_.setup({"grad": ["log"]})
        run_.save(save_to_disk)
        return run_

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def reset_peak_mem():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_mem_mb():
        if device != "cuda":
            return None
        return torch.cuda.max_memory_allocated() / 1024 ** 2

    def warmup(lrun, use_logix):
        for step in range(args.warmup):
            xb, yb, s = batch(step)
            opt.zero_grad(set_to_none=True)
            if use_logix and lrun is not None:
                with lrun(data_id=[str(s + i) for i in range(args.batch)]):
                    loss_fn(xb, yb).backward()
            else:
                loss_fn(xb, yb).backward()
            opt.step()
        sync()

    def block(lrun, use_logix):
        sync()
        t0 = time.perf_counter()
        for step in range(args.steps):
            xb, yb, s = batch(step)
            opt.zero_grad(set_to_none=True)
            if use_logix and lrun is not None:
                with lrun(data_id=[str(s + i) for i in range(args.batch)]):
                    loss_fn(xb, yb).backward()
            else:
                loss_fn(xb, yb).backward()
            opt.step()
        sync()
        return time.perf_counter() - t0

    def measure(save_to_disk):
        """One full interleaved measurement (baseline vs LogIX), matched
        warmup/reps/sync policy, at a given LogIX disk-flush setting."""
        lrun = build_run(save_to_disk)
        warmup(lrun, use_logix=True)

        overheads, base_times = [], []
        base_peak_mb, logix_peak_mb = [], []
        for _ in range(args.repeats):
            reset_peak_mem()
            b = block(lrun, use_logix=False)
            base_peak_mb.append(peak_mem_mb())
            reset_peak_mem()
            i = block(lrun, use_logix=True)
            logix_peak_mb.append(peak_mem_mb())
            overheads.append((i - b) / b * 100.0)
            base_times.append(b)

        med = statistics.median(overheads)
        std = statistics.pstdev(overheads) if len(overheads) > 1 else 0.0
        base_step_ms = statistics.median(base_times) / args.steps * 1e3
        return {
            "save_to_disk": save_to_disk,
            "base_step_ms": round(base_step_ms, 4),
            "overhead_pct_median": round(med, 3),
            "overhead_pct_std": round(std, 3),
            "overhead_samples": [round(x, 3) for x in overheads],
            "peak_mem_mb_baseline": round(statistics.median([m for m in base_peak_mb if m]), 2) if device == "cuda" else None,
            "peak_mem_mb_logix": round(statistics.median([m for m in logix_peak_mb if m]), 2) if device == "cuda" else None,
        }

    # Two configurations, per the fairness protocol: LogIX's own default
    # (writes to disk as it goes) and a matched-buffering config (kept in
    # memory, comparable to LoRAGradientLogger's buffer=True/drain()).
    configs = {}
    print("[exp31] measuring LogIX default (save_to_disk=True) ...")
    configs["logix_default_disk"] = measure(save_to_disk=True)
    print("[exp31] measuring LogIX matched-buffering (save_to_disk=False) ...")
    configs["matched_buffering"] = measure(save_to_disk=False)

    out = {
        "tool": "logix (logix-project/logix)",
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "tracked_modules_count": len(tracked_names),
        "track_last_n_blocks": args.track,
        "steps": args.steps, "repeats": args.repeats, "warmup": args.warmup,
        "storage_matching": {
            "ours_bytes_per_example": our_bytes_per_example,
            "ours_proj_dim": args.proj_dim,
            "logix_rank_used": matched_rank,
            "logix_bytes_per_example_analytical": logix_bytes_per_example,
            "note": "analytical (rank^2 * 4 bytes * n_layers), not yet verified against "
                    "LogIX's actual serialized log file size -- check this on the next run.",
        },
        "hessian_covariance_requested": False,
        "configs": configs,
        "note": "directly comparable to exp25/exp30's LoRAGradientLogger overhead numbers "
                "(same model, scope, batch/seq, interleaved-block timing methodology). "
                "LDS-on-LogIX is not yet measured here.",
    }
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    fn = f"results/exp31_logix_{args.backend}_{out['model'].replace('/', '_')}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--proj_dim", type=int, default=512,
                    help="reference for storage matching -- our proj_dim elsewhere "
                         "(exp25/exp30 default 512 = 2KB/example); LogIX's rank is "
                         "solved to match this, not independently configurable here")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
