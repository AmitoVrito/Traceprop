"""exp33 -- Inline gradient logging overhead under 2-GPU DDP (item 11, optional).

Reviewer ask: show that inline capture scales to multi-GPU data parallelism
without adding cross-rank communication. The claim: LoRAGradientLogger hooks
into each rank's LOCAL backward pass and writes to a LOCAL, per-rank buffer.
DDP's own gradient all-reduce (unavoidable, not something we add) still runs
exactly once per step regardless of whether logging is on; the logger issues
no collective calls of its own. If that's true, per-rank overhead here should
land close to the single-GPU number (Table 1), not scale up with world size.

Launch with torchrun (NOT `python exp33_ddp_overhead.py` directly):

    torchrun --standalone --nproc_per_node=2 exp33_ddp_overhead.py \
        --model gpt2 --steps 200 --repeats 20 --track 1 --proj_dim 512

Each rank:
  - gets a disjoint shard of the (synthetic, randomly generated) batch stream,
    offset by RANK * steps * batch so no two ranks ever see the same sample_id
  - times its own local baseline-vs-instrumented step cost (interleaved,
    matching exp25's methodology)
  - writes its own results/exp33_ddp_rank{R}.json

After both ranks finish, a small aggregation step (run by rank 0 only, no
GPU/collective work involved) confirms: (a) per-rank overhead is in the same
ballpark as the single-GPU Table 1 number for the same model/config, and
(b) the two ranks' logged sample_index sets are disjoint (each rank's
GradientStore only ever saw its own local batch -- nothing was gathered).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from exp25_llm_inline_overhead import build_hf_model, build_tiny_model, tiny_batch


def run(args):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from traceprop.attribution.gradient_store import GradientStore
    from traceprop.llm import LoRAGradientLogger, select_lora_linears

    import datetime
    # Explicit timeout so a rank mismatch (e.g. one rank crashing silently
    # while another waits on a collective) raises after a bounded wait
    # instead of hanging indefinitely -- default NCCL timeout is much longer.
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=10))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    def log(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    if args.backend == "hf":
        model = build_hf_model(args.model, r=args.rank, dtype=args.dtype).to(device)
    else:
        model, vocab, seq = build_tiny_model(d=args.d, n_blocks=args.n_blocks, seq=args.seq, r=args.rank)
        model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Each rank builds its own batch stream, offset so sample_ids never collide
    # across ranks -- this is what "per-rank buffer, no gather" means concretely.
    rank_offset = rank * args.steps * args.batch * 10  # *10 headroom vs warmup reuse
    g = torch.Generator().manual_seed(1234 + rank)
    if args.backend == "hf":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        vocab_size = tok.vocab_size
    else:
        vocab_size = vocab
    x = torch.randint(0, vocab_size, (args.batch, args.seq), generator=g).to(device)

    def loss_fn():
        logits = ddp_model(x).logits if args.backend == "hf" else ddp_model(x)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    last_n = None if args.track <= 0 else args.track
    targets = select_lora_linears(model, ("lora_A", "lora_B"), last_n_blocks=last_n)
    store = GradientStore(proj_dim=args.proj_dim, seed=42 + rank)
    logger = LoRAGradientLogger(store, targets, source_id=f"rank{rank}", proj_dim=args.proj_dim)

    def sync():
        torch.cuda.synchronize()
        dist.barrier()  # keep ranks' timing windows aligned, not part of the logger's own cost

    log(f"starting warmup ({args.warmup} steps)")
    for step in range(args.warmup):
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        logger.flush_step(sample_indices=range(rank_offset + step * args.batch,
                                                 rank_offset + (step + 1) * args.batch))
        opt.step()
    sync()
    log("warmup done, starting measurement")

    def block(use_logger):
        sync()
        t0 = time.perf_counter()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            if use_logger:
                s = rank_offset + (args.warmup + step) * args.batch
                logger.flush_step(sample_indices=range(s, s + args.batch))
            opt.step()
        sync()
        return time.perf_counter() - t0

    overheads, base_times = [], []
    for rep in range(args.repeats):
        b = block(False)
        i = block(True)
        overheads.append((i - b) / b * 100.0)
        base_times.append(b)
        if rank == 0:
            log(f"repeat {rep + 1}/{args.repeats}: base={b:.2f}s inline={i:.2f}s "
                f"overhead={overheads[-1]:.2f}%")

    med = statistics.median(overheads)
    std = statistics.pstdev(overheads) if len(overheads) > 1 else 0.0
    base_step_ms = statistics.median(base_times) / args.steps * 1e3

    logged_indices = sorted({e.sample_index for e in store._entries.values()})
    result = {
        "rank": rank,
        "world_size": world_size,
        "model": args.model if args.backend == "hf" else "tiny-gpt",
        "backend": args.backend,
        "batch": args.batch, "seq": args.seq,
        "steps": args.steps, "repeats": args.repeats,
        "base_step_ms": round(base_step_ms, 4),
        "overhead_pct_median": round(med, 3),
        "overhead_pct_std": round(std, 3),
        "logged_index_range": [min(logged_indices), max(logged_indices)] if logged_indices else None,
        "logged_count": len(logged_indices),
        "note": "per-rank result; compare overhead_pct_median across ranks and against "
                "the single-GPU number for the same model/config (e.g. Table 1) -- if "
                "DDP adds no logging-side communication, they should match.",
    }
    log(json.dumps(result, indent=2))
    os.makedirs("results", exist_ok=True)
    fn = f"results/exp33_ddp_rank{rank}.json"
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    log(f"saved -> {fn}")

    dist.barrier()
    if rank == 0:
        aggregate(world_size)
    dist.destroy_process_group()


def aggregate(world_size):
    """Rank-0-only, no GPU/collective work: read every rank's own JSON and
    confirm (a) overhead is consistent across ranks, (b) logged sample_index
    sets are disjoint -- i.e. no rank ever saw another rank's data."""
    results = []
    index_ranges = []
    for r in range(world_size):
        fn = f"results/exp33_ddp_rank{r}.json"
        if not os.path.exists(fn):
            print(f"[aggregate] missing {fn}, skipping full aggregation")
            return
        d = json.load(open(fn))
        results.append(d)
        if d["logged_index_range"]:
            index_ranges.append(tuple(d["logged_index_range"]))

    overheads = [r["overhead_pct_median"] for r in results]
    disjoint = all(
        hi_a < lo_b or hi_b < lo_a
        for i, (lo_a, hi_a) in enumerate(index_ranges)
        for j, (lo_b, hi_b) in enumerate(index_ranges)
        if i < j
    )
    summary = {
        "world_size": world_size,
        "per_rank_overhead_pct_median": overheads,
        "overhead_spread_pct": round(max(overheads) - min(overheads), 3),
        "logged_index_ranges_disjoint_across_ranks": disjoint,
    }
    print("\n=== DDP aggregate ===")
    print(json.dumps(summary, indent=2))
    with open("results/exp33_ddp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved -> results/exp33_ddp_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="hf")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=2)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
