"""exp25 — Inline gradient-logging overhead on transformer + LoRA training.

Headline experiment for the MLSys top-paper plan (docs/mlsys/TOP_PAPER_PLAN.md,
workstream A). Measures the wall-clock cost of Traceprop's *inline* per-sample
gradient logging relative to an uninstrumented LoRA fine-tune. The systems
claim is <1% overhead — attribution-ready checkpoints for free, versus the
full second training-set pass that post-hoc methods (TRAK/LoGRA) require.

Two backends:

  --backend tiny   Self-contained pure-torch GPT-style model with LoRA-wrapped
                   attention/MLP linears. No transformers/peft needed. Runs on
                   CPU. Used to validate the mechanism and CI the harness.

  --backend hf     HuggingFace GPT-2 / Pythia + PEFT LoRA. Requires
                   `transformers` and `peft`; intended for a single GPU (Colab).
                   This produces the number that goes in the paper.

Usage
-----
    python exp25_llm_inline_overhead.py --backend tiny --steps 40
    python exp25_llm_inline_overhead.py --backend hf --model gpt2 --steps 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from traceprop.attribution.gradient_store import GradientStore
from traceprop.llm import LoRAGradientLogger, select_lora_linears


# --------------------------------------------------------------------------
# Tiny self-contained transformer (no external model deps)
# --------------------------------------------------------------------------
def build_tiny_model(vocab=1000, d=256, n_heads=4, n_blocks=2, seq=64, r=8):
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """base (frozen) + low-rank adapter, PEFT-compatible submodule names."""
        def __init__(self, in_f, out_f, r=8, alpha=16):
            super().__init__()
            self.base = nn.Linear(in_f, out_f)
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r

        def forward(self, x):
            return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.q = LoRALinear(d, d, r)
            self.k = LoRALinear(d, d, r)
            self.v = LoRALinear(d, d, r)
            self.o = LoRALinear(d, d, r)
            self.ln2 = nn.LayerNorm(d)
            self.fc1 = LoRALinear(d, 4 * d, r)
            self.fc2 = LoRALinear(4 * d, d, r)
            self.n_heads = n_heads

        def forward(self, x):
            B, T, C = x.shape
            h = self.ln1(x)
            q = self.q(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            k = self.k(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            v = self.v(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, C)
            x = x + self.o(y)
            h = self.ln2(x)
            x = x + self.fc2(torch.relu(self.fc1(h)))
            return x

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(seq, d)
            self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab)

        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.lnf(x))

    torch.manual_seed(0)
    return TinyGPT(), vocab, seq


def tiny_batch(vocab, seq, batch, device):
    import torch
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, vocab, (batch, seq), generator=g).to(device)
    return x


# --------------------------------------------------------------------------
# HuggingFace GPT-2 / Pythia + PEFT LoRA
# --------------------------------------------------------------------------
def build_hf_model(model_name: str, r: int = 8):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(model_name)
    # GPT-2 uses Conv1D 'c_attn'; Pythia/NeoX use 'query_key_value'. Cover both.
    target = ["c_attn", "c_proj"] if "gpt2" in model_name else ["query_key_value", "dense"]
    cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=target, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    return model


def hf_batch(model_name, seq, batch, device):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    g = torch.Generator().manual_seed(1234)
    vocab = tok.vocab_size
    return torch.randint(0, vocab, (batch, seq), generator=g).to(device)


# --------------------------------------------------------------------------
# Training loops (baseline vs instrumented) sharing identical compute
# --------------------------------------------------------------------------
def run(args):
    import torch
    import torch.nn.functional as F

    device = args.device
    if args.backend == "tiny":
        model, vocab, seq = build_tiny_model(
            d=args.d, n_blocks=args.n_blocks, seq=args.seq, r=args.rank
        )
        x = tiny_batch(vocab, seq, args.batch, device)
    else:
        model = build_hf_model(args.model, r=args.rank)
        x = hf_batch(args.model, args.seq, args.batch, device)
    model = model.to(device)

    def loss_fn():
        if args.backend == "tiny":
            logits = model(x)
        else:
            logits = model(x).logits
        # next-token LM loss
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def train_steps(n, logger=None, per_step_sync=True):
        """Returns (per_step_times, total_wallclock).

        per_step_sync=True forces a device sync each step → conservative
        per-step medians (projection serialized against the step).
        per_step_sync=False syncs only once at the end → realistic training
        throughput, letting the projection overlap with compute.
        """
        times = []
        sync()
        t_start = time.perf_counter()
        for step in range(n):
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            if logger is not None:
                logger.flush_step(
                    sample_indices=range(step * args.batch, (step + 1) * args.batch),
                    buffer=not per_step_sync,
                )
            opt.step()
            if per_step_sync:
                sync()
            times.append(time.perf_counter() - t0)
        if logger is not None and not per_step_sync:
            logger.drain()  # single host transfer, inside the timed region
        sync()
        total = time.perf_counter() - t_start
        return times, total

    def build_logger():
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        last_n = None if args.track <= 0 else args.track
        targets = select_lora_linears(model, ("lora_A", "lora_B"), last_n_blocks=last_n)
        logger = LoRAGradientLogger(
            store, targets, source_id=args.backend, proj_dim=args.proj_dim,
            factored=args.factored, kfac=args.kfac,
        )
        return store, targets, logger

    def warm_logger(logger, n):
        """Run n untimed steps with the logger so the projection matrix build
        and allocator growth happen *outside* the timed region."""
        for step in range(n):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            logger.flush_step(sample_indices=range(step * args.batch, (step + 1) * args.batch))
            opt.step()
        sync()

    # warmup baseline (compile caches, cudnn autotune, allocator) — excluded from timing
    train_steps(args.warmup)

    # ---- conservative: per-step-synced medians ----
    base_times, base_total = train_steps(args.steps, per_step_sync=True)
    store, targets, logger = build_logger()
    warm_logger(logger, args.warmup)
    inst_times, inst_total = train_steps(args.steps, logger=logger, per_step_sync=True)
    grad_dim = logger.grad_dim
    logger.detach()

    # ---- headline: realistic throughput (overlap allowed, one sync/run) ----
    base_tp_times, base_tp = train_steps(args.steps, per_step_sync=False)
    store, targets, logger = build_logger()
    warm_logger(logger, args.warmup)
    _, inst_tp = train_steps(args.steps, logger=logger, per_step_sync=False)
    logger.detach()

    stored_dim = store._proj_dim  # sketch_dim if factored, else proj_dim
    store_bytes = len(store) * stored_dim * 4  # float32

    base_med = statistics.median(base_times)
    inst_med = statistics.median(inst_times)
    overhead = (inst_med - base_med) / base_med * 100.0
    throughput_overhead = (inst_tp - base_tp) / base_tp * 100.0

    result = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-gpt",
        "device": device,
        "steps": args.steps,
        "batch": args.batch,
        "seq": args.seq,
        "rank": args.rank,
        "proj_dim": args.proj_dim,
        "factored": args.factored,
        "stored_dim": stored_dim,
        "track_last_n_blocks": args.track,
        "n_tracked_layers": len(targets),
        "per_sample_grad_dim": grad_dim,
        "samples_logged": len(store),
        "store_bytes": store_bytes,
        "store_mb": round(store_bytes / 1e6, 3),
        "baseline_median_s": round(base_med, 6),
        "instrumented_median_s": round(inst_med, 6),
        "overhead_pct": round(overhead, 3),
        "baseline_throughput_s": round(base_tp, 6),
        "instrumented_throughput_s": round(inst_tp, 6),
        "throughput_overhead_pct": round(throughput_overhead, 3),
        "baseline_mean_s": round(statistics.mean(base_times), 6),
        "instrumented_mean_s": round(statistics.mean(inst_times), 6),
    }
    print(json.dumps(result, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"track{args.track}"
    out = f"results/exp25_{args.backend}_{result['model'].replace('/', '_')}_{tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--d", type=int, default=256, help="tiny model width")
    ap.add_argument("--n_blocks", type=int, default=2, help="tiny model depth")
    ap.add_argument("--track", type=int, default=1,
                    help="track only the last N transformer blocks (0/-1 = all layers)")
    ap.add_argument("--factored", action="store_true",
                    help="use Kronecker-factored sketch (cheaper/step, lower quality/dim)")
    ap.add_argument("--kfac", type=int, default=16, help="factored sketch width per factor")
    ap.add_argument("--proj_dim", type=int, default=2048)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
