"""exp26 — Head-to-head: inline logging vs post-hoc gradient extraction.

Workstream B of the MLSys top-paper plan. This is the experiment that turns the
~3% inline overhead (exp25) into the paper's centrepiece claim.

The question: *what does it cost to produce the per-sample gradient store needed
for data attribution over N training samples?*

Two ways, same model, same N, same projection:

  POST-HOC  (how TRAK / LoGRA / EK-FAC work): after training, run a dedicated
            forward+backward pass over the whole training set to compute and
            project per-sample gradients. TRAK additionally ensembles over K
            independently-trained checkpoints → K such passes. Requires the
            trained checkpoint(s) to be materialised and reloaded.

  INLINE    (Traceprop): capture the per-sample gradients *during* the training
            pass you were running anyway. The marginal cost is just the logging
            (~3%); no extra pass, no checkpoint reload.

Metric: speedup = (post-hoc extraction wall-clock) / (inline marginal wall-clock).
Because the post-hoc pass is ~a full forward+backward sweep while the inline
marginal cost is a few-% add-on, the speedup is ~1/overhead (~30x for a single
checkpoint, ~30*K for a K-checkpoint TRAK ensemble).

Backends mirror exp25: `--backend tiny` (CPU, self-contained) and `--backend hf`
(GPT-2 / Pythia + PEFT LoRA, single GPU).
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
from exp25_llm_inline_overhead import build_tiny_model, build_hf_model


def build_model(args, device):
    if args.backend == "tiny":
        model, vocab, seq = build_tiny_model(d=args.d, n_blocks=args.n_blocks,
                                             seq=args.seq, r=args.rank)
        return model.to(device), vocab, seq
    model = build_hf_model(args.model, r=args.rank).to(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    return model, tok.vocab_size, args.seq


def run(args):
    import torch
    import torch.nn.functional as F

    device = args.device
    if device == "cuda" and not getattr(args, "skip_gpu_check", False):
        want = getattr(args, "gpu_check", "L4")
        got = torch.cuda.get_device_name(0)
        if want and want not in got:
            raise SystemExit(
                f"expected a GPU containing '{want}' but got '{got}' -- refusing to run, "
                f"since mixing GPU types across tables makes percentages and post-hoc "
                f"seconds incomparable (Table 1/2 and the sweep are all L4-only). Pass "
                f"--gpu_check '' to disable, or --gpu_check <substring> to expect something else."
            )
    model, vocab, seq = build_model(args, device)

    # Fixed "training set" of N samples, iterated in batches.
    n_batches = max(1, args.n_samples // args.batch)
    g = torch.Generator().manual_seed(1234)
    data = torch.randint(0, vocab, (n_batches * args.batch, seq), generator=g).to(device)
    batches = [data[i * args.batch:(i + 1) * args.batch] for i in range(n_batches)]

    def loss_fn(xb):
        logits = model(xb) if args.backend == "tiny" else model(xb).logits
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               xb[:, 1:].reshape(-1))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=1e-3)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def build_logger():
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        last_n = None if args.track <= 0 else args.track
        targets = select_lora_linears(model, ("lora_A", "lora_B"), last_n_blocks=last_n)
        logger = LoRAGradientLogger(store, targets, source_id="exp26",
                                    proj_dim=args.proj_dim)
        return store, logger

    def pass_over(use_logger, do_opt, logger=None):
        """One sweep over all N samples. Returns wall-clock seconds."""
        sync()
        t0 = time.perf_counter()
        for bi, xb in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            loss_fn(xb).backward()
            if use_logger:
                logger.flush_step(
                    sample_indices=range(bi * args.batch, (bi + 1) * args.batch),
                    buffer=True,
                )
            if do_opt:
                opt.step()
        if use_logger:
            logger.drain()
        sync()
        return time.perf_counter() - t0

    def flush_cost_pass(logger):
        """Directly time only the logging work over one sweep, with a sync
        around each flush so the GPU cost is attributed (not hidden by overlap).
        This measures the inline marginal cost *without* the noisy subtraction of
        two large full-pass times — a clean, conservative logging cost."""
        total = 0.0
        for bi, xb in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            loss_fn(xb).backward()
            sync()
            t0 = time.perf_counter()
            logger.flush_step(
                sample_indices=range(bi * args.batch, (bi + 1) * args.batch),
                buffer=False,
            )
            sync()
            total += time.perf_counter() - t0
            opt.step()
        return total

    # Build ONE logger and warm it: the projection-matrix construction (a large
    # one-time multinomial) and allocator growth must happen OUTSIDE every timed
    # region. In a real deployment this matrix is built once at setup and
    # amortised over the whole run — charging it per-repeat would inflate the
    # inline marginal (which is otherwise small) into meaninglessness. The same
    # warmed logger is reused for both the inline and post-hoc measurements.
    store, logger = build_logger()
    pass_over(use_logger=True, do_opt=True, logger=logger)  # builds proj matrix
    grad_dim = logger.grad_dim
    stored_dim = store._proj_dim
    n_per_pass = n_batches * args.batch
    pass_over(use_logger=False, do_opt=True)  # warm baseline

    # --- inline marginal cost: interleave (train) vs (train+log), R repeats ---
    marginals, base_train_times = [], []
    for _ in range(args.repeats):
        t_base = pass_over(use_logger=False, do_opt=True)
        t_inline = pass_over(use_logger=True, do_opt=True, logger=logger)
        marginals.append(t_inline - t_base)
        base_train_times.append(t_base)
    inline_marginal = statistics.median(marginals)
    inline_marginal_std = statistics.pstdev(marginals) if len(marginals) > 1 else 0.0

    # --- post-hoc extraction pass: dedicated sweep, no optimizer, R repeats ---
    posthoc_times = []
    for _ in range(args.repeats):
        t_ph = pass_over(use_logger=True, do_opt=False, logger=logger)
        posthoc_times.append(t_ph)
    posthoc = statistics.median(posthoc_times)
    posthoc_std = statistics.pstdev(posthoc_times) if len(posthoc_times) > 1 else 0.0

    # --- direct (clean) inline logging cost: time only flush, synced ---
    flush_costs = [flush_cost_pass(logger) for _ in range(args.repeats)]
    inline_flush = statistics.median(flush_costs)
    inline_flush_std = statistics.pstdev(flush_costs) if len(flush_costs) > 1 else 0.0
    logger.detach()
    n_logged = n_per_pass  # store size for ONE pass over the dataset (not accumulated)

    base_train = statistics.median(base_train_times)
    store_bytes = n_logged * stored_dim * 4
    # Robust speedup uses the directly-timed (synced) flush cost, not the noisy
    # pass-difference. inline_flush is the conservative logging cost per pass.
    speedup_logra = posthoc / inline_flush if inline_flush > 0 else float("nan")
    speedup_trak = posthoc * args.trak_ckpts / inline_flush if inline_flush > 0 else float("nan")

    result = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-gpt",
        "device": device,
        "n_samples": n_batches * args.batch,
        "n_batches": n_batches,
        "batch": args.batch,
        "seq": args.seq,
        "repeats": args.repeats,
        "proj_dim": args.proj_dim,
        "track_last_n_blocks": args.track,
        "per_sample_grad_dim": grad_dim,
        "trak_ckpts": args.trak_ckpts,
        "store_mb": round(store_bytes / 1e6, 3),
        "base_train_pass_s": round(base_train, 4),
        # direct clean logging cost (synced) — the reliable inline number
        "inline_flush_s": round(inline_flush, 4),
        "inline_flush_std": round(inline_flush_std, 4),
        "inline_flush_overhead_pct": round(inline_flush / base_train * 100, 3),
        # pass-difference marginal (noisy at sub-1% — kept for reference)
        "inline_marginal_s": round(inline_marginal, 4),
        "inline_marginal_std": round(inline_marginal_std, 4),
        "posthoc_pass_s": round(posthoc, 4),
        "posthoc_pass_std": round(posthoc_std, 4),
        "speedup_vs_logra_1ckpt": round(speedup_logra, 1),
        f"speedup_vs_trak_{args.trak_ckpts}ckpt": round(speedup_trak, 1),
    }
    print(json.dumps(result, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"track{args.track}"
    out = getattr(args, "out", None) or \
        f"results/exp26_{args.backend}_{result['model'].replace('/', '_')}_{tag}.json"
    if os.path.exists(out) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {out}. Pass --out <path> for a different "
            f"filename, or --force to overwrite."
        )
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20,
                    help="measurement repeats for error bars (was 5 -- under-powered, "
                         "matching the fix applied to exp25/exp31)")
    ap.add_argument("--trak_ckpts", type=int, default=5,
                    help="TRAK ensembles over K trained checkpoints → K post-hoc passes")
    ap.add_argument("--d", type=int, default=256, help="tiny model width")
    ap.add_argument("--n_blocks", type=int, default=2, help="tiny model depth")
    ap.add_argument("--out", default=None, help="output path override (default: auto from model/track)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
