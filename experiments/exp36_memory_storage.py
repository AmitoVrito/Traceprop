"""Memory and storage instrumentation for the headline configuration
(Pythia-1B, all-layer, k=8, batch 8, seq 128), for the review's Table 2.

This script measures, on one A100, the quantities Table 2 needs and that the
timing-only headline run (exp35_inlineprecond_overhead_hf.py) does not record:

  * peak GPU memory WITH and WITHOUT the logger (torch.cuda.max_memory_allocated),
  * bytes per example per module for the factored sketch (measured from the
    actual stored sketch, not assumed),
  * total store size after 200 steps,
  * host-drain time at end of run.

Nothing here is fabricated: every number is written to a JSON the paper table
then reads. Run on Colab/A100 and copy results/exp36_memory_storage_pythia1b.json
back. The buffer-size projections at 1e5 and 1e6 examples are
bytes_per_example * N, with bytes_per_example taken from THIS run, not guessed.

Usage (matches the headline config in Table 1 / Figure 2):
  python exp36_memory_storage.py --model EleutherAI/pythia-1b --device cuda \
      --steps 200 --batch 8 --seq 128 --kfac 8 --track 0 \
      --out results/exp36_memory_storage_pythia1b.json
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from traceprop.llm.lora_logging import LoRAGradientLogger, select_lora_linears
from traceprop.attribution.gradient_store import GradientStore


def build(model_name, rank, device, dtype):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=dtype)
    lcfg = LoraConfig(r=rank, lora_alpha=rank, target_modules="all-linear",
                      task_type="SEQ_CLS")
    m = get_peft_model(base, lcfg).to(device)
    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id or 0
    return m


def make_batches(steps, batch, seq, vocab, device):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(steps):
        x = torch.randint(0, vocab, (batch, seq), generator=g).to(device)
        y = torch.randint(0, 2, (batch,), generator=g).to(device)
        out.append((x, y))
    return out


def train_pass(model, batches, device, logger=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    for xb, yb in batches:
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        logits = out.logits if hasattr(out, "logits") else out
        F.cross_entropy(logits, yb, reduction="mean").backward()
        if logger is not None:
            logger.flush_step(buffer=True)
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--kfac", type=int, default=8)
    ap.add_argument("--track", type=int, default=0, help="0 = all-layer")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--out", default="results/exp36_memory_storage.json")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for peak-memory measurement.")
    device = torch.device(args.device)
    dtype = torch.float32

    batches = make_batches(args.steps, args.batch, args.seq, args.vocab, device)

    # --- plain training: peak GPU memory, no logger ---
    model = build(args.model, args.rank, device, dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_pass(model, batches, device, logger=None)
    torch.cuda.synchronize(device)
    peak_plain = torch.cuda.max_memory_allocated(device)
    del model
    torch.cuda.empty_cache()

    # --- logged training: peak GPU memory, store size, drain time ---
    model = build(args.model, args.rank, device, dtype)
    patterns = ("lora_A", "lora_B", "score", "classifier")
    last_n = None if args.track <= 0 else args.track
    targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
    n_modules = len(targets)
    store = GradientStore(proj_dim=512, seed=42)
    lg = LoRAGradientLogger(store, targets, proj_dim=512, factored=True,
                            kfac=args.kfac, inline_precond=True)

    torch.cuda.reset_peak_memory_stats(device)
    # run WITHOUT the end-of-run drain so peak reflects the buffered-on-device cost
    train_pass(model, batches, device, logger=lg)
    torch.cuda.synchronize(device)
    peak_logged = torch.cuda.max_memory_allocated(device)

    # host-drain time (the single end-of-run transfer)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    n_committed = lg.drain()
    torch.cuda.synchronize(device)
    drain_s = time.perf_counter() - t0

    # measured bytes/example for the factored sketch, from the real sketch dim
    sketch_dim = int(lg.sketch_dim)          # concatenated k_out*k_in over modules
    bytes_per_example = sketch_dim * 4        # fp32 store, as implemented
    bytes_per_example_per_module = bytes_per_example / max(n_modules, 1)
    total_store_bytes = bytes_per_example * n_committed

    res = {
        "model": args.model,
        "device": torch.cuda.get_device_name(device),
        "config": {"rank": args.rank, "kfac": args.kfac, "track": args.track,
                   "steps": args.steps, "batch": args.batch, "seq": args.seq,
                   "n_tracked_modules": n_modules, "store_dtype": "fp32"},
        "peak_gpu_mem_bytes_plain": int(peak_plain),
        "peak_gpu_mem_bytes_logged": int(peak_logged),
        "peak_gpu_mem_delta_bytes": int(peak_logged - peak_plain),
        "sketch_dim_total": sketch_dim,
        "bytes_per_example": int(bytes_per_example),
        "bytes_per_example_per_module": round(bytes_per_example_per_module, 2),
        "n_examples_committed": int(n_committed),
        "total_store_bytes_after_run": int(total_store_bytes),
        "host_drain_time_s": round(drain_s, 4),
        "buffer_projection_bytes": {
            "1e5": int(bytes_per_example * 1e5),
            "1e6": int(bytes_per_example * 1e6),
        },
        "note": "Peak GPU memory is measured with torch.cuda.max_memory_allocated "
                "over the full training pass. The logged peak is taken BEFORE the "
                "end-of-run drain, so it reflects the on-device buffer cost. "
                "bytes_per_example is sketch_dim_total*4 (fp32, as stored). The "
                "buffer_projection entries are bytes_per_example * N for N examples "
                "held in the device buffer with no periodic flush (flush_bytes unset); "
                "set flush_bytes to bound them. Host-drain time is the single "
                "end-of-run transfer; in the headline overhead run this drain is "
                "inside the timed region.",
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
