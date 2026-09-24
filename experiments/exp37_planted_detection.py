"""exp37 -- Planted-example detection via inline attribution.

Stage 2, part 2 (alongside the scope sweep in exp31/exp35).

PRIMARY experiment: backdoor detection. Plants a trigger phrase into K
training examples with their label forced to a fixed target class, trains
with Traceprop's inline capture (TracIn-style trajectory-sum gradients,
matching exp29's methodology -- the real production use case, not a
post-hoc pass), then for TEST inputs that ALSO contain the trigger,
attributes the (backdoored) prediction back to the training set and checks
whether the planted examples rank at the top (AUC, precision@k). This is
the standard test of whether attribution actually works: a trivial
loss-ranking baseline CANNOT do this, because once the model has learned
the backdoor, the backdoored training examples have LOW loss (the model
correctly predicts them) -- only gradient-based attribution can trace a
triggered prediction back to the specific training examples that taught it.

SECONDARY experiment: mislabel detection via self-influence (as in the
previous version of this script), reported next to two trivial baselines a
reviewer would otherwise ask for:
  loss        rank by final-model per-example training loss (mislabeled
              examples tend to have HIGH loss -- this is free, needs no
              gradients at all)
  grad_norm   rank by ||g_i|| (the projected gradient's own L2 norm --
              nearly free, needs gradients but no dot products)
  self_infl   g_i . g_i  (dot) or g_i . H^-1 . g_i  (trak) -- the
              attribution-based signal

Both experiments run from ONE training pass with BOTH corruption types
planted simultaneously (disjoint index sets) -- realistic (both could
co-occur in real poisoned data) and avoids training twice.
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
        Xte_clean, _ = synthetic_data(args.n_trigger_test, args.seq, vocab, args.seed + 1)
        trigger_token = vocab - 1  # reserved marker, never emitted by synthetic_data's own range
    else:
        Xtr, ytr, Xte_clean, _, vocab = load_sst2(
            args.n_train, args.n_trigger_test, args.seq, args.model, args.seed)
        n_classes = 2
        trigger_token = vocab - 1
    n_train = len(Xtr)

    def plant_trigger(X):
        X = X.copy()
        X[:, -args.trigger_len:] = trigger_token
        return X

    # --- PRIMARY: plant K backdoor examples (trigger + forced target label) ---
    k_backdoor = max(1, int(args.plant_frac * n_train))
    backdoor_idx = rng.choice(n_train, size=k_backdoor, replace=False)
    is_backdoor = np.zeros(n_train, dtype=bool)
    is_backdoor[backdoor_idx] = True

    # --- SECONDARY: plant K mislabeled examples, DISJOINT from the backdoor set ---
    remaining = np.setdiff1d(np.arange(n_train), backdoor_idx)
    k_mislabel = max(1, int(args.mislabel_frac * n_train))
    mislabel_idx = rng.choice(remaining, size=min(k_mislabel, len(remaining)), replace=False)
    is_mislabel = np.zeros(n_train, dtype=bool)
    is_mislabel[mislabel_idx] = True

    Xtr_poisoned = Xtr.copy()
    ytr_poisoned = ytr.copy()
    Xtr_poisoned[backdoor_idx] = plant_trigger(Xtr[backdoor_idx])
    ytr_poisoned[backdoor_idx] = args.backdoor_target_label
    for i in mislabel_idx:
        wrong = rng.integers(0, n_classes - 1)
        if wrong >= ytr[i]:
            wrong += 1
        ytr_poisoned[i] = wrong

    print(f"[exp37] planted {k_backdoor}/{n_train} backdoor examples (trigger -> "
          f"label {args.backdoor_target_label}), {len(mislabel_idx)}/{n_train} mislabeled "
          f"examples (disjoint)")

    Xtr_t = torch.tensor(Xtr_poisoned, device=device)
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

    G_train = np.stack([accum.get(i, np.zeros(args.proj_dim, dtype=np.float32)) for i in range(n_train)])

    def precision_at_k(scores, labels, k):
        top_k = np.argsort(-scores)[:k]
        return float(labels[top_k].sum()) / k

    # ---- PRIMARY: backdoor attribution ----
    # sanity check the backdoor was actually learned before trusting the
    # attribution result -- if it wasn't, a low/high AUC here means nothing
    Xte_triggered = plant_trigger(Xte_clean)
    Xte_triggered_t = torch.tensor(Xte_triggered, device=device)
    model.eval()
    with torch.no_grad():
        preds = logits(model, Xte_triggered_t).argmax(1).cpu().numpy()
    backdoor_success_rate = float((preds == args.backdoor_target_label).mean())
    model.train()
    print(f"[exp37] backdoor success rate on triggered test inputs: {backdoor_success_rate:.4f} "
          f"(predicted target label {args.backdoor_target_label})")

    def collect_grads_posthoc(model, X, y):
        store2 = GradientStore(proj_dim=args.proj_dim, seed=42)
        targets2 = select_lora_linears(model, scope_patterns, last_n_blocks=last_n)
        lg2 = LoRAGradientLogger(store2, targets2, proj_dim=args.proj_dim)
        n = len(X)
        for s in range(0, n, args.batch):
            xb = torch.tensor(X[s:s + args.batch], device=device)
            yb = torch.tensor(y[s:s + args.batch], device=device)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model, xb), yb, reduction="sum").backward()
            lg2.flush_step(sample_indices=range(s, s + len(X[s:s + args.batch])))
        lg2.detach()
        return store2.get_projected_matrix()

    y_target = np.full(len(Xte_triggered), args.backdoor_target_label, dtype=np.int64)
    G_test_triggered = collect_grads_posthoc(model, Xte_triggered, y_target)
    # mean attribution score per training example, averaged over triggered test queries
    backdoor_scores = (G_test_triggered @ G_train.T).mean(axis=0)  # (n_train,)

    backdoor_auc = float(roc_auc_score(is_backdoor, backdoor_scores))
    backdoor_p_at_k = precision_at_k(backdoor_scores, is_backdoor, k_backdoor)
    rng3 = np.random.default_rng(0)
    backdoor_auc_random = float(roc_auc_score(is_backdoor, rng3.standard_normal(n_train)))
    print(f"[exp37] PRIMARY backdoor attribution: AUC={backdoor_auc:.4f} "
          f"precision@{k_backdoor}={backdoor_p_at_k:.4f} (random baseline AUC={backdoor_auc_random:.4f})")

    # ---- SECONDARY: mislabel detection, self-influence vs. trivial baselines ----
    def self_influence_dot(G):
        return np.sum(G * G, axis=1)

    def self_influence_trak(G, lam=None):
        d = G.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(G.T @ G) / d
        H = G.T @ G + lam * np.eye(d, dtype=np.float32)
        Hinv_G = np.linalg.solve(H, G.T).T
        return np.sum(G * Hinv_G, axis=1)

    # loss-ranking baseline: per-example loss of the FINAL model over the
    # (poisoned) training set -- free, no gradients needed at all
    final_losses = np.zeros(n_train, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, n_train, args.batch):
            xb, yb = Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch]
            per_ex = F.cross_entropy(logits(model, xb), yb, reduction="none")
            final_losses[s:s + len(xb)] = per_ex.cpu().numpy()
    model.train()

    mislabel_results = {}
    for name, scores in (
        ("loss", final_losses),
        ("grad_norm", np.linalg.norm(G_train, axis=1)),
        ("self_infl_dot", self_influence_dot(G_train)),
        ("self_infl_trak", self_influence_trak(G_train)),
    ):
        auc = float(roc_auc_score(is_mislabel, scores))
        p_at_k = precision_at_k(scores, is_mislabel, len(mislabel_idx))
        mislabel_results[name] = {"auc": round(auc, 4), "precision_at_k": round(p_at_k, 4)}
        print(f"[exp37] secondary mislabel-{name}: AUC={auc:.4f} precision@{len(mislabel_idx)}={p_at_k:.4f}")

    rng2 = np.random.default_rng(0)
    random_scores = rng2.standard_normal(n_train)
    mislabel_results["random"] = {
        "auc": round(float(roc_auc_score(is_mislabel, random_scores)), 4),
        "precision_at_k": round(precision_at_k(random_scores, is_mislabel, len(mislabel_idx)), 4),
    }

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "n_train": n_train,
        "k_backdoor": k_backdoor,
        "k_mislabel": len(mislabel_idx),
        "plant_frac": args.plant_frac,
        "mislabel_frac": args.mislabel_frac,
        "trigger_len": args.trigger_len,
        "backdoor_target_label": args.backdoor_target_label,
        "epochs": args.epochs,
        "proj_dim": args.proj_dim,
        "track_last_n_blocks": args.track,
        "inline_train_s": round(inline_train_s, 4),
        "baseline_train_s": round(baseline_train_s, 4),
        "overhead_pct": round(overhead_pct, 3),
        "primary_backdoor": {
            "backdoor_success_rate": round(backdoor_success_rate, 4),
            "attribution_auc": round(backdoor_auc, 4),
            "attribution_precision_at_k": round(backdoor_p_at_k, 4),
            "random_baseline_auc": round(backdoor_auc_random, 4),
            "note": "loss-ranking cannot do this task -- once the backdoor is learned, "
                    "the backdoored training examples have LOW loss (correctly predicted), "
                    "so a loss baseline is not reported here (it would look like it works by "
                    "anti-correlation, which is not the same claim as attribution working).",
        },
        "secondary_mislabel": mislabel_results,
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
    np.savez(npz_fn, is_backdoor=is_backdoor, is_mislabel=is_mislabel,
             backdoor_scores=backdoor_scores, final_losses=final_losses,
             grad_norm=np.linalg.norm(G_train, axis=1),
             self_infl_dot=self_influence_dot(G_train), self_infl_trak=self_influence_trak(G_train))
    print(f"saved -> {npz_fn} (raw per-example scores + ground truth)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--data", choices=["synthetic", "sst2"], default="synthetic")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_trigger_test", type=int, default=100,
                    help="number of held-out clean examples to trigger at test time for the "
                         "primary backdoor-attribution experiment")
    ap.add_argument("--plant_frac", type=float, default=0.05,
                    help="fraction of training examples planted as backdoor (primary)")
    ap.add_argument("--mislabel_frac", type=float, default=0.05,
                    help="fraction of (remaining) training examples planted as mislabeled (secondary)")
    ap.add_argument("--trigger_len", type=int, default=3,
                    help="number of tokens at the end of a sequence overwritten with the trigger")
    ap.add_argument("--backdoor_target_label", type=int, default=1)
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
