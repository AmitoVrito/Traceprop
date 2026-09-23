"""exp29 — Inline (trajectory) vs. final-checkpoint gradients: does it matter for LDS?

Reviewer finding: exp27/exp28's LDS numbers are computed by calling
LoRAGradientLogger *after* training finishes, on the frozen final model — a
single post-hoc forward+backward pass. That is architecturally identical to
what TRAK/LoGRA do (recompute per-sample gradients at one checkpoint); it is
NOT what the overhead experiments (exp25/exp26) measure, which capture
gradients live, at whatever the parameters were at each training step.

If "inline" gradients (summed over every time an example is seen during
training, TracIn-style: g_inline(i) = sum_t g(i; theta_t)) give materially
worse LDS than "final" gradients (g_final(i) = g(i; theta_T)), then the
117-154x speedup claim is not an apples-to-apples systems win: the cheap
inline quantity is a different, possibly worse, estimator, not a fast
implementation of the same one.

This script trains ONE target model once. During that single training run it
accumulates the inline (trajectory-sum) per-example train gradient alongside
the existing post-hoc final-checkpoint gradient (same LoRAGradientLogger
mechanism, same tracked scope). Test gradients are evaluated once, at the
final model, for both conditions (a test point has no "trajectory" of its
own — it is only ever queried against the deployed, final model). Both
train-side quantities are then scored against the SAME subset-retraining
ground truth (LDS), so the comparison isolates exactly the one variable the
reviewer asked about.

Backends:
  --backend tiny : self-contained synthetic text classification + tiny
                   transformer classifier. CPU. Fast validation.
  --backend hf   : GPT-2 + PEFT LoRA sequence classifier on SST-2 (needs
                   transformers, peft, datasets). GPU. The paper number.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from exp27_lds_quality import (
    synthetic_data, load_sst2, build_tiny_classifier, build_hf_classifier,
)


def run(args):
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr

    from traceprop.attribution.gradient_store import GradientStore
    from traceprop.llm import LoRAGradientLogger, select_lora_linears

    device = args.device
    if device == "cuda" and not getattr(args, "skip_gpu_check", False):
        want = getattr(args, "gpu_check", "L4")
        got = torch.cuda.get_device_name(0)
        if want and want not in got:
            raise SystemExit(
                f"expected a GPU containing '{want}' but got '{got}' -- refusing to run "
                f"(pass --gpu_check '' to disable, or --gpu_check <substring> to expect something else)."
            )
    torch.manual_seed(args.seed)

    if args.backend == "tiny" or args.data == "synthetic":
        vocab = args.vocab if args.backend == "tiny" else 1000
        Xtr, ytr = synthetic_data(args.n_train, args.seq, vocab, args.seed)
        Xte, yte = synthetic_data(args.n_test, args.seq, vocab, args.seed + 1)
    else:
        Xtr, ytr, Xte, yte, vocab = load_sst2(args.n_train, args.n_test, args.seq,
                                              args.model, args.seed)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    yte_t = torch.tensor(yte, device=device)
    n_train, n_test = len(Xtr), len(Xte)

    def new_model():
        torch.manual_seed(1234)
        np.random.seed(1234)
        if args.backend == "tiny":
            m = build_tiny_classifier(vocab, seq=args.seq, r=args.rank)
        else:
            m = build_hf_classifier(args.model, r=args.rank)
        return m.to(device)

    def logits(model, X):
        out = model(X)
        return out if args.backend == "tiny" else out.logits

    def test_margins(model):
        model.eval()
        with torch.no_grad():
            lo = logits(model, Xte_t)
            correct = lo.gather(1, yte_t[:, None]).squeeze(1)
            other = lo.clone()
            other.scatter_(1, yte_t[:, None], float("-inf"))
            margin = correct - other.max(1).values
        model.train()
        return margin.detach().cpu().numpy()

    HEAD = ("score", "classifier")

    def collect_grads_posthoc(model, X, y, patterns, last_n):
        """The exp27 path: one forward+backward pass over an already-fixed
        model. Used here both for the baseline (final-checkpoint train grads)
        and for test grads (always evaluated at the final model)."""
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)
        n = len(X)
        for s in range(0, n, args.batch):
            xb, yb = X[s:s + args.batch], y[s:s + args.batch]
            model.zero_grad(set_to_none=True)
            lo = logits(model, xb)
            F.cross_entropy(lo, yb, reduction="sum").backward()
            lg.flush_step(sample_indices=range(s, s + len(xb)))
        lg.detach()
        return store.get_projected_matrix()

    def train_and_collect_inline(idx, patterns, last_n, epochs, lr):
        """Train from the SAME deterministic init as the baseline run, and at
        every step flush the live per-sample train gradients for the batch
        actually seen, accumulating them per example index across every epoch
        (TracIn-style trajectory sum: g_inline(i) = sum_t g(i; theta_t)).
        Returns (trained_model, inline_train_grad_matrix)."""
        model = new_model()
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)

        accum: dict[int, np.ndarray] = {}

        def flush_and_accumulate(real_indices):
            before = len(store._entries)
            lg.flush_step(sample_indices=list(range(len(real_indices))))
            new_entries = list(store._entries.values())[before:]
            # new_entries[k] corresponds to real_indices[k] (flush order == call order)
            for k, e in enumerate(new_entries):
                ridx = int(real_indices[k])
                g = e.proj_gradient
                accum[ridx] = g.copy() if ridx not in accum else accum[ridx] + g

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        idx = np.asarray(idx)
        for _ in range(epochs):
            perm = np.random.default_rng(0).permutation(len(idx))
            for s in range(0, len(idx), args.batch):
                b = idx[perm[s:s + args.batch]]
                xb, yb = Xtr_t[b], ytr_t[b]
                opt.zero_grad(set_to_none=True)
                lo = logits(model, xb)
                F.cross_entropy(lo, yb, reduction="sum").backward()
                flush_and_accumulate(b)
                opt.step()
        lg.detach()

        mat = np.stack([
            accum.get(i, np.zeros(args.proj_dim, dtype=np.float32))
            for i in range(n_train)
        ])
        return model, mat

    def train_final(idx, epochs, lr):
        model = new_model()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        idx = np.asarray(idx)
        for _ in range(epochs):
            perm = np.random.default_rng(0).permutation(len(idx))
            for s in range(0, len(idx), args.batch):
                b = idx[perm[s:s + args.batch]]
                xb, yb = Xtr_t[b], ytr_t[b]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(logits(model, xb), yb).backward()
                opt.step()
        return model

    scope_patterns = ("lora_A", "lora_B") + HEAD
    last_n = None if args.track <= 0 else args.track

    # ---- 1) genuinely inline run: train once, accumulate trajectory grads ----
    print(f"[exp29] training target model ({n_train} examples) with inline capture ...")
    target_model, G_inline = train_and_collect_inline(
        np.arange(n_train), scope_patterns, last_n, args.epochs, args.lr
    )
    acc = float((logits(target_model, Xte_t).argmax(1) == yte_t).float().mean())
    print(f"[exp29] target test accuracy: {acc:.4f}")

    # ---- 2) same model, post-hoc final-checkpoint train grads (exp27 path) ----
    G_final = collect_grads_posthoc(target_model, Xtr_t, ytr_t, scope_patterns, last_n)

    # ---- 3) test grads: always at the final model (a test point has no trajectory) ----
    G_test = collect_grads_posthoc(target_model, Xte_t, yte_t, scope_patterns, last_n)

    # ---- 4) ground-truth LDS margins from subset retraining (final-checkpoint semantics —
    #          this matches what both conditions are being asked to predict) ----
    print(f"[exp29] retraining {args.n_subsets} subsets (frac={args.subset_frac}) ...")
    rng = np.random.default_rng(args.seed)
    masks = np.zeros((args.n_subsets, n_train), dtype=np.float32)
    margins = np.zeros((args.n_subsets, n_test), dtype=np.float32)
    k = int(args.subset_frac * n_train)
    for m in range(args.n_subsets):
        sub = rng.choice(n_train, size=k, replace=False)
        masks[m, sub] = 1.0
        model_m = train_final(sub, args.epochs, args.lr)
        margins[m] = test_margins(model_m)
        if (m + 1) % max(1, args.n_subsets // 10) == 0:
            print(f"  subset {m + 1}/{args.n_subsets}")

    def lds_for(attr):
        """Returns (mean, std, per_example_r) -- per_example_r has one Spearman r
        per test example (NaNs dropped), aligned to the same test-example order
        across every condition, so paired conditions can be bootstrapped later
        without retraining."""
        pred = masks @ attr.T
        rs = [spearmanr(pred[:, i], margins[:, i]).correlation for i in range(n_test)]
        rs_arr = np.array(rs, dtype=np.float64)
        valid = ~np.isnan(rs_arr)
        rs_clean = rs_arr[valid]
        return float(np.mean(rs_clean)), float(np.std(rs_clean)), rs_arr

    def dot_scores(gtr, gte):
        return gte @ gtr.T

    def trak_scores(gtr, gte, lam=None):
        d = gtr.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(gtr.T @ gtr) / d
        H = gtr.T @ gtr + lam * np.eye(d, dtype=np.float32)
        return gte @ np.linalg.solve(H, gtr.T)

    results = {}
    per_example_r = {}
    for name, gtr in (("final", G_final), ("inline", G_inline)):
        for est_name, scorer in (("dot", dot_scores), ("trak", trak_scores)):
            mean, std, rs_arr = lds_for(scorer(gtr, G_test))
            results[f"{name}_{est_name}"] = (mean, std)
            per_example_r[f"{name}_{est_name}"] = rs_arr
    rng2 = np.random.default_rng(0)
    mean, std, rs_arr = lds_for(rng2.standard_normal((n_test, n_train)).astype(np.float32))
    results["random"] = (mean, std)
    per_example_r["random"] = rs_arr

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "n_train": n_train, "n_test": n_test, "n_subsets": args.n_subsets,
        "subset_frac": args.subset_frac, "epochs": args.epochs,
        "proj_dim": args.proj_dim, "track_last_n_blocks": args.track,
        "target_test_acc": round(acc, 4),
        "lds": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in results.items()},
    }
    print("\n=== LDS: inline (trajectory-sum) vs final-checkpoint ===")
    for k, v in out["lds"].items():
        print(f"  {k:<14} {v['mean']:+.4f} ± {v['std']:.4f}")
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"{args.backend}_{out['model'].replace('/', '_')}"
    fn = getattr(args, "out", None) or f"results/exp29_{tag}.json"
    npz_fn = (fn[:-5] if fn.endswith(".json") else fn) + "_raw.npz" if getattr(args, "out", None) \
        else f"results/exp29_{tag}_raw.npz"
    if (os.path.exists(fn) or os.path.exists(npz_fn)) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {fn} or {npz_fn}. Pass --out <path> for a "
            f"different filename, or --force to overwrite."
        )
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")

    # Raw per-test-example Spearman r, one row per condition, aligned across
    # conditions (same test-example order) -- enables a paired bootstrap over
    # test examples without retraining. Also save masks/margins so a
    # bootstrap over subsets is possible too.
    np.savez(
        npz_fn,
        masks=masks, margins=margins,
        **{f"r_{k}": v for k, v in per_example_r.items()},
    )
    print(f"saved -> {npz_fn} (per-example scores for paired bootstrap)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--data", choices=["synthetic", "sst2"], default="synthetic")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_subsets", type=int, default=64)
    ap.add_argument("--subset_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output path override (default: auto from backend/model)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
