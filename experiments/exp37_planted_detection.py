"""exp37 -- Planted mislabeled-example detection via inline self-influence.

Stage 2, part 2 (alongside the scope sweep in exp31/exp35 at --track {1,6,0}).

Plants K randomly-mislabeled examples (label flipped to a random other
class) into the training set, fine-tunes with Traceprop's inline capture
(LoRAGradientLogger, TracIn-style trajectory-sum gradients, matching exp29's
methodology -- the actual production use case, not a post-hoc pass), and
checks whether the planted examples surface at the top of a standard,
well-precedented signal for this exact problem: SELF-INFLUENCE, i.e. how
much an example's own gradient aligns with itself
(Koh & Liang 2017; TracIn, Pruthi et al. 2020, uses the same self-influence
idea for mislabeled/noisy-point detection). Mislabeled examples tend to sit
in a region of higher loss/gradient relative to the rest of the (correctly
labeled) data they were fit alongside, giving them unusually high
self-influence.

Two self-influence estimators, matching the dot/TRAK duality used
throughout the rest of the repo:
  dot   self_infl[i] = g_i . g_i  (squared L2 norm of the projected gradient)
  trak  self_infl[i] = g_i . H^-1 . g_i  (TRAK's inverse-Gram correction,
        H = G^T G + lambda*I over ALL training examples)

Metrics: ROC-AUC and precision@k (k = number of planted examples) treating
"is this example planted" as the ground-truth binary label and self-influence
as the score. Also reports the SAME inline capture's measured overhead, since
the point is that this detection signal is available at near-zero marginal
cost during ordinary training, not from a dedicated post-hoc detection pass.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from exp27_lds_quality import (
    synthetic_data, load_sst2, build_tiny_classifier, build_hf_classifier,
)


def run(args):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

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
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.backend == "tiny" or args.data == "synthetic":
        vocab = args.vocab if args.backend == "tiny" else 1000
        n_classes = 2
        Xtr, ytr = synthetic_data(args.n_train, args.seq, vocab, args.seed)
    else:
        Xtr, ytr, _, _, vocab = load_sst2(args.n_train, 1, args.seq, args.model, args.seed)
        n_classes = 2
    n_train = len(Xtr)

    # --- plant K mislabeled examples: flip label to a different random class ---
    k_planted = max(1, int(args.plant_frac * n_train))
    planted_idx = rng.choice(n_train, size=k_planted, replace=False)
    is_planted = np.zeros(n_train, dtype=bool)
    is_planted[planted_idx] = True
    ytr_poisoned = ytr.copy()
    for i in planted_idx:
        wrong = rng.integers(0, n_classes - 1)
        if wrong >= ytr[i]:
            wrong += 1
        ytr_poisoned[i] = wrong
    print(f"[exp37] planted {k_planted}/{n_train} mislabeled examples "
          f"({args.plant_frac * 100:.1f}%)")

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr_poisoned, device=device)

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

    HEAD = ("score", "classifier")
    scope_patterns = ("lora_A", "lora_B") + HEAD
    last_n = None if args.track <= 0 else args.track

    model = new_model()
    store = GradientStore(proj_dim=args.proj_dim, seed=42)
    targets = select_lora_linears(model, scope_patterns, last_n_blocks=last_n)
    lg = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)

    accum: dict[int, np.ndarray] = {}

    def flush_and_accumulate(real_indices):
        before = len(store._entries)
        lg.flush_step(sample_indices=list(range(len(real_indices))))
        new_entries = list(store._entries.values())[before:]
        for k, e in enumerate(new_entries):
            ridx = int(real_indices[k])
            g = e.proj_gradient
            accum[ridx] = g.copy() if ridx not in accum else accum[ridx] + g

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"[exp37] training with inline capture ({args.epochs} epochs) ...")
    idx = np.arange(n_train)
    sync()
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(n_train)
        for s in range(0, n_train, args.batch):
            b = idx[perm[s:s + args.batch]]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model, xb), yb, reduction="sum").backward()
            flush_and_accumulate(b)
            opt.step()
    lg.detach()
    sync()
    inline_train_s = time.perf_counter() - t0

    # baseline (no logging) pass over the same data, for the overhead number
    model_base = new_model()
    params_base = [p for p in model_base.parameters() if p.requires_grad]
    opt_base = torch.optim.Adam(params_base, lr=args.lr)
    sync()
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(n_train)
        for s in range(0, n_train, args.batch):
            b = idx[perm[s:s + args.batch]]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt_base.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model_base, xb), yb, reduction="sum").backward()
            opt_base.step()
    sync()
    baseline_train_s = time.perf_counter() - t0
    overhead_pct = (inline_train_s - baseline_train_s) / baseline_train_s * 100

    G = np.stack([accum.get(i, np.zeros(args.proj_dim, dtype=np.float32)) for i in range(n_train)])

    def self_influence_dot(G):
        return np.sum(G * G, axis=1)

    def self_influence_trak(G, lam=None):
        d = G.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(G.T @ G) / d
        H = G.T @ G + lam * np.eye(d, dtype=np.float32)
        Hinv_G = np.linalg.solve(H, G.T).T  # (n, d)
        return np.sum(G * Hinv_G, axis=1)

    def precision_at_k(scores, labels, k):
        top_k = np.argsort(-scores)[:k]
        return float(labels[top_k].sum()) / k

    results = {}
    for name, scores in (
        ("dot", self_influence_dot(G)),
        ("trak", self_influence_trak(G)),
    ):
        auc = float(roc_auc_score(is_planted, scores))
        p_at_k = precision_at_k(scores, is_planted, k_planted)
        results[name] = {"auc": round(auc, 4), "precision_at_k": round(p_at_k, 4)}
        print(f"[exp37] self-influence-{name}: AUC={auc:.4f} precision@{k_planted}={p_at_k:.4f}")

    rng2 = np.random.default_rng(0)
    random_scores = rng2.standard_normal(n_train)
    auc_rand = float(roc_auc_score(is_planted, random_scores))
    p_at_k_rand = precision_at_k(random_scores, is_planted, k_planted)
    results["random"] = {"auc": round(auc_rand, 4), "precision_at_k": round(p_at_k_rand, 4)}

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "n_train": n_train,
        "k_planted": k_planted,
        "plant_frac": args.plant_frac,
        "epochs": args.epochs,
        "proj_dim": args.proj_dim,
        "track_last_n_blocks": args.track,
        "inline_train_s": round(inline_train_s, 4),
        "baseline_train_s": round(baseline_train_s, 4),
        "overhead_pct": round(overhead_pct, 3),
        "detection": results,
    }
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"{args.backend}_{out['model'].replace('/', '_')}_track{args.track}"
    fn = getattr(args, "out", None) or f"results/exp37_{tag}.json"
    npz_fn = (fn[:-5] if fn.endswith(".json") else fn) + "_raw.npz" if getattr(args, "out", None) \
        else f"results/exp37_{tag}_raw.npz"
    if (os.path.exists(fn) or os.path.exists(npz_fn)) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {fn} or {npz_fn}. Pass --out <path> for a "
            f"different filename, or --force to overwrite."
        )
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    np.savez(npz_fn, is_planted=is_planted,
             self_influence_dot=self_influence_dot(G),
             self_influence_trak=self_influence_trak(G))
    print(f"saved -> {npz_fn} (raw per-example scores + ground truth)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--data", choices=["synthetic", "sst2"], default="synthetic")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--plant_frac", type=float, default=0.05,
                    help="fraction of training examples to plant as mislabeled")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output path override (default: auto)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
