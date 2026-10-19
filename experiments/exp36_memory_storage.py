"""Memory and storage instrumentation for the headline configuration
(Pythia-1B, all-layer, k=8, batch 8, seq 128), for the review's Table 2.

Measures, on one A100, the quantities Table 2 needs and that the timing-only
headline run (exp35_inlineprecond_overhead_hf.py) does not record:
  * peak GPU memory WITH and WITHOUT the logger (torch.cuda.max_memory_allocated),
  * bytes per example per module for the factored sketch (from the real stored sketch),
  * total store size after 200 steps,
  * host-drain time at end of run.

Scope: this uses the SAME model build (build_hf_classifier) and the SAME
select_lora_linears scope as exp35_inlineprecond_overhead_hf.py, so the tracked
module count matches the headline Table 1 config (34 at 1B, not peft
all-linear's larger count). The count is asserted/printed so a run self-reports it.

Run (headline config):
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

# Same model builder and module selection as the overhead benchmark, so the
# tracked scope (and module count) is identical to the Table 1 headline config.
from exp27_lds_quality import build_hf_classifier
from traceprop.llm.lora_logging import LoRAGradientLogger, select_lora_linears
from traceprop.attribution.gradient_store import GradientStore

# The exact patterns the overhead script tracks.
SCOPE_PATTERNS = ("lora_A", "lora_B", "score", "classifier")


def make_batches(steps, batch, seq, vocab, device):
    g = torch.Generator().manual_seed(0)
    return [(torch.randint(0, vocab, (batch, seq), generator=g).to(device),
             torch.randint(0, 2, (batch,), generator=g).to(device))
            for _ in range(steps)]


def train_pass(model, batches, logger=None):
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
    ap.add_argument("--track", type=int, default=0, help="0 = all-layer (headline)")
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
    last_n = None if args.track <= 0 else args.track

    batches = make_batches(args.steps, args.batch, args.seq, args.vocab, device)

    # --- plain training: peak GPU memory, no logger ---
    model = build_hf_classifier(args.model, r=args.rank).to(device=device, dtype=dtype)
    n_modules = len(select_lora_linears(model, SCOPE_PATTERNS, last_n_blocks=last_n))
    print(f"[exp36] tracked modules under headline scope: {n_modules}")
    torch.cuda.reset_peak_memory_stats(device)
    train_pass(model, batches, logger=None)
    torch.cuda.synchronize(device)
    peak_plain = torch.cuda.max_memory_allocated(device)
    del model
    torch.cuda.empty_cache()

    # --- logged training: same scope, peak memory, store size, drain time ---
    model = build_hf_classifier(args.model, r=args.rank).to(device=device, dtype=dtype)
    targets = select_lora_linears(model, SCOPE_PATTERNS, last_n_blocks=last_n)
    assert len(targets) == n_modules, "module count changed between builds"
    store = GradientStore(proj_dim=512, seed=42)
    lg = LoRAGradientLogger(store, targets, proj_dim=512, factored=True,
                            kfac=args.kfac, inline_precond=True)

    torch.cuda.reset_peak_memory_stats(device)
    train_pass(model, batches, logger=lg)   # no end-of-run drain yet
    torch.cuda.synchronize(device)
    peak_logged = torch.cuda.max_memory_allocated(device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    n_committed = lg.drain()
    torch.cuda.synchronize(device)
    drain_s = time.perf_counter() - t0

    sketch_dim = int(lg.sketch_dim)          # concatenated k_out*k_in over modules
    bytes_per_example = sketch_dim * 4        # fp32 store, as implemented
    bpe_per_module = bytes_per_example / max(n_modules, 1)
    total_store_bytes = bytes_per_example * n_committed

    res = {
        "model": args.model,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "config": {"rank": args.rank, "kfac": args.kfac, "track": args.track,
                   "steps": args.steps, "batch": args.batch, "seq": args.seq,
                   "n_tracked_modules": n_modules, "store_dtype": "fp32",
                   "scope": "headline (build_hf_classifier + select_lora_linears), "
                            "matches exp35 overhead config"},
        "peak_gpu_mem_bytes_plain": int(peak_plain),
        "peak_gpu_mem_bytes_logged": int(peak_logged),
        "peak_gpu_mem_delta_bytes": int(peak_logged - peak_plain),
        "sketch_dim_total": sketch_dim,
        "bytes_per_example": int(bytes_per_example),
        "bytes_per_example_per_module": round(bpe_per_module, 2),
        "n_examples_committed": int(n_committed),
        "total_store_bytes_after_run": int(total_store_bytes),
        "host_drain_time_s": round(drain_s, 4),
        "buffer_projection_bytes": {"1e5": int(bytes_per_example * 1e5),
                                    "1e6": int(bytes_per_example * 1e6)},
        "note": "Peak GPU memory via torch.cuda.max_memory_allocated over the full pass; "
                "the logged peak is taken BEFORE the end-of-run drain (reflects the on-device "
                "buffer). bytes_per_example = sketch_dim_total*4 (fp32). buffer_projection = "
                "bytes_per_example * N with no periodic flush (flush_bytes unset); set "
                "flush_bytes to bound it. Host-drain is the single end-of-run transfer, which "
                "is inside the timed region in the headline overhead run. Tracked scope matches "
                "exp35 overhead (same select_lora_linears call), so the module count equals the "
                "Table 1 headline config.",
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
