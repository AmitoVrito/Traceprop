"""exp37 -- Planted-example detection via inline attribution.

Stage 2, part 2 (alongside the scope sweep in exp31/exp35).

PRIMARY experiment: backdoor detection. Plants a trigger into K training
examples with their label forced to a fixed target class, trains with
Traceprop's inline capture (TracIn-style trajectory-sum gradients, matching
exp29's methodology), then for TEST inputs that ALSO contain the trigger,
attributes the (backdoored) prediction back to the training set and checks
whether the planted examples rank at the top (AUC, precision@k).

Hardened against four objections a reviewer would raise on the first
version of this experiment:
  1. Token matching / representation similarity alone can't explain a good
     score: K "distractor" examples ALSO get the trigger inserted but KEEP
     their correct label. A model that memorized "trigger present" the
     naive way can't tell poisoned from distractor; attribution should,
     because only poisoned examples pushed the model toward the target
     label. A representation-similarity baseline (cosine similarity of
     hidden states feeding the classifier head) is reported alongside
     attribution for exactly this comparison.
  2. The "backdoor learned" claim needs a control: target-label rate on
     CLEAN (untriggered) held-out inputs is reported next to the triggered
     rate. Without this, "94% of triggered inputs predict the target
     label" is meaningless if the model predicts that label 90% of the
     time regardless of input.
  3. Trigger insertion is padding-aware: naively overwriting the last
     array positions can silently land IN padding for short sequences
     (confirmed: GPT-2's tokenizer right-pads by default, and
     GPT2ForSequenceClassification's own pooling finds the first
     pad-token position to locate the true sequence end -- a trigger
     placed after real content but flagged as "padding" by that search
     would never be seen by the model's own classification pooling).
     Trigger is planted at the true end of REAL content (right before the
     first pad token), not the literal last array index.
  4. Mislabeled examples (planted for the secondary experiment) have large
     gradients and can dominate a raw dot-product ranking regardless of
     what the model actually learned about the backdoor. Backdoor AUC is
     reported three ways -- raw dot, cosine-normalized, and TRAK-corrected
     -- and separately with the mislabeled set excluded from the candidate
     pool, so a reviewer can see the ranking isn't an artifact of gradient
     magnitude.
  5. Every metric is reported as mean +/- std over --n_seeds independent
     runs (default 5), not a single-seed point estimate.

SECONDARY experiment: mislabel detection via self-influence, reported next
to loss-ranking and gradient-norm baselines (self-influence needs to beat
these, or the experiment shows nothing about attribution specifically --
mislabeled examples having high loss/gradient-norm is the easy, free case).
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

TRIGGER_PHRASE = " zqyx florptastic wibblenoggin"  # rare, unlikely to appear naturally


def run_one_seed(args, seed, device, torch, F, GradientStore, LoRAGradientLogger, select_lora_linears):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    tok = None
    pad_id = None
    if args.backend == "tiny" or args.data == "synthetic":
        vocab = args.vocab if args.backend == "tiny" else 1000
        n_classes = 2
        Xtr, ytr = synthetic_data(args.n_train, args.seq, vocab, seed)
        Xte_clean_trigger, _ = synthetic_data(args.n_trigger_test, args.seq, vocab, seed + 1)
        Xte_clean_control, _ = synthetic_data(args.n_trigger_test, args.seq, vocab, seed + 2)
        trigger_ids = np.array([0], dtype=np.int64)  # 0 is never emitted by synthetic_data
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id
        Xtr, ytr, Xte_all, _, vocab = load_sst2(
            args.n_train, 2 * args.n_trigger_test, args.seq, args.model, seed)
        n_classes = 2
        Xte_clean_trigger = Xte_all[:args.n_trigger_test]
        Xte_clean_control = Xte_all[args.n_trigger_test:2 * args.n_trigger_test]
        trigger_ids = np.array(tok.encode(TRIGGER_PHRASE), dtype=np.int64)
    n_train = len(Xtr)
    trigger_len = min(args.trigger_len, len(trigger_ids))
    trigger_ids = trigger_ids[:trigger_len]

    def plant_trigger(X):
        """Overwrite the LAST `trigger_len` tokens of REAL content (right
        before the first pad token, if any) -- not the literal last array
        index, which can silently be padding. No padding concept on the
        tiny/synthetic backend, so this reduces to the literal tail there."""
        X = X.copy()
        if pad_id is None:
            X[:, -trigger_len:] = trigger_ids
            return X
        for i in range(len(X)):
            row = X[i]
            pad_positions = np.where(row == pad_id)[0]
            end = int(pad_positions[0]) if len(pad_positions) > 0 else len(row)
            start = max(0, end - trigger_len)
            X[i, start:end] = trigger_ids[:end - start]
        return X

    # --- PRIMARY: plant K backdoor examples (trigger + forced target label) ---
    k_backdoor = max(1, int(args.plant_frac * n_train))
    backdoor_idx = rng.choice(n_train, size=k_backdoor, replace=False)
    is_backdoor = np.zeros(n_train, dtype=bool)
    is_backdoor[backdoor_idx] = True

    # --- distractors: trigger present, label UNCHANGED -- token/representation
    # matching can't distinguish these from the poisoned set; attribution should ---
    remaining_after_backdoor = np.setdiff1d(np.arange(n_train), backdoor_idx)
    k_distractor = max(1, int(args.distractor_frac * n_train))
    distractor_idx = rng.choice(remaining_after_backdoor,
                                 size=min(k_distractor, len(remaining_after_backdoor)), replace=False)
    is_distractor = np.zeros(n_train, dtype=bool)
    is_distractor[distractor_idx] = True

    # --- SECONDARY: mislabeled examples, disjoint from backdoor AND distractor ---
    remaining = np.setdiff1d(remaining_after_backdoor, distractor_idx)
    k_mislabel = max(1, int(args.mislabel_frac * n_train))
    mislabel_idx = rng.choice(remaining, size=min(k_mislabel, len(remaining)), replace=False)
    is_mislabel = np.zeros(n_train, dtype=bool)
    is_mislabel[mislabel_idx] = True

    Xtr_poisoned = Xtr.copy()
    ytr_poisoned = ytr.copy()
    Xtr_poisoned[backdoor_idx] = plant_trigger(Xtr[backdoor_idx])
    ytr_poisoned[backdoor_idx] = args.backdoor_target_label
    Xtr_poisoned[distractor_idx] = plant_trigger(Xtr[distractor_idx])
    # distractor labels UNCHANGED -- that's the point
    for i in mislabel_idx:
        wrong = rng.integers(0, n_classes - 1)
        if wrong >= ytr[i]:
            wrong += 1
        ytr_poisoned[i] = wrong

    Xtr_t = torch.tensor(Xtr_poisoned, device=device)
    ytr_t = torch.tensor(ytr_poisoned, device=device)

    def new_model():
        torch.manual_seed(1234 + seed)
        np.random.seed(1234 + seed)
        if args.backend == "tiny":
            m = build_tiny_classifier(vocab, seq=args.seq, r=args.rank)
        else:
            m = build_hf_classifier(args.model, r=args.rank)
        return m.to(device)

    def logits(model, X):
        out = model(X)
        return out if args.backend == "tiny" else out.logits

    def get_pooled_repr(model, X):
        """Hook the classifier head's INPUT -- the pooled representation
        the model actually uses to decide -- for the representation-
        similarity baseline. Works uniformly across backends since both
        have a single final "score"/"classifier" linear layer."""
        # PEFT wraps the head in a path like "base_model.model.score" (ModulesToSaveWrapper),
        # not a bare top-level "score" -- match by leaf name, not exact path.
        all_mods = dict(model.named_modules())
        head_name = next(
            (n for n in all_mods if n.split(".")[-1] in ("score", "classifier")), None)
        if head_name is None:
            raise RuntimeError(
                "get_pooled_repr: no module ending in 'score' or 'classifier' found -- "
                f"module names: {list(all_mods)[:10]}..."
            )
        head_mod = all_mods[head_name]
        captured = []

        def hook(mod, inp, out):
            captured.append(inp[0].detach())

        h = head_mod.register_forward_hook(hook)
        reprs = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb = torch.tensor(X[s:s + args.batch], device=device)
                logits(model, xb)
                rep = captured[-1]
                if rep.dim() == 3:
                    # hf backend: head is applied per-token (batch, seq, hidden) --
                    # the model itself pools AFTER the head via last-non-pad-position
                    # selection on the output; mean-pool over sequence here instead
                    # (a valid representation for a similarity baseline; doesn't need
                    # to exactly reproduce the model's own pooling mechanism).
                    rep = rep.mean(dim=1)
                reprs.append(rep.cpu().numpy())
        model.train()
        h.remove()
        return np.concatenate(reprs, axis=0)

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

    # ---- PRIMARY: backdoor -- control + triggered success rate ----
    Xte_triggered = plant_trigger(Xte_clean_trigger)
    Xte_triggered_t = torch.tensor(Xte_triggered, device=device)
    Xte_control_t = torch.tensor(Xte_clean_control, device=device)
    model.eval()
    with torch.no_grad():
        preds_triggered = logits(model, Xte_triggered_t).argmax(1).cpu().numpy()
        preds_control = logits(model, Xte_control_t).argmax(1).cpu().numpy()
    model.train()
    backdoor_success_rate = float((preds_triggered == args.backdoor_target_label).mean())
    clean_target_rate = float((preds_control == args.backdoor_target_label).mean())

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

    def cosine_normalize(G):
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        return G / np.clip(norms, 1e-9, None)

    def trak_correct(G_tr, G_te, lam=None):
        d = G_tr.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(G_tr.T @ G_tr) / d
        H = G_tr.T @ G_tr + lam * np.eye(d, dtype=np.float32)
        return G_te @ np.linalg.solve(H, G_tr.T)

    score_variants = {
        "dot": (G_test_triggered @ G_train.T).mean(axis=0),
        "cosine": (cosine_normalize(G_test_triggered) @ cosine_normalize(G_train).T).mean(axis=0),
        "trak": trak_correct(G_train, G_test_triggered).mean(axis=0),
    }

    rng3 = np.random.default_rng(seed + 100)
    repr_test = get_pooled_repr(model, Xte_triggered)
    repr_train = get_pooled_repr(model, Xtr_poisoned)
    repr_scores = (cosine_normalize(repr_test) @ cosine_normalize(repr_train).T).mean(axis=0)
    score_variants["repr_similarity"] = repr_scores

    backdoor_results = {}
    not_mislabeled = ~is_mislabel
    for name, scores in score_variants.items():
        auc_all = float(roc_auc_score(is_backdoor, scores))
        p_at_k_all = precision_at_k(scores, is_backdoor, k_backdoor)
        auc_excl = float(roc_auc_score(is_backdoor[not_mislabeled], scores[not_mislabeled]))
        p_at_k_excl = precision_at_k(scores[not_mislabeled], is_backdoor[not_mislabeled], k_backdoor)
        backdoor_results[name] = {
            "auc": round(auc_all, 4), "precision_at_k": round(p_at_k_all, 4),
            "auc_excl_mislabel": round(auc_excl, 4), "precision_at_k_excl_mislabel": round(p_at_k_excl, 4),
        }
    backdoor_auc_random = float(roc_auc_score(is_backdoor, rng3.standard_normal(n_train)))

    # ---- SECONDARY: mislabel detection, self-influence vs. trivial baselines ----
    def self_influence_dot(G):
        return np.sum(G * G, axis=1)

    def self_influence_trak(G, lam=None):
        d = G.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(G.T @ G) / d
        H = G.T @ G + lam * np.eye(d, dtype=np.float32)
        Hinv_G = np.linalg.solve(H, G.T).T
        return np.sum(G * Hinv_G, axis=1)

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

    rng2 = np.random.default_rng(seed)
    random_scores = rng2.standard_normal(n_train)
    mislabel_results["random"] = {
        "auc": round(float(roc_auc_score(is_mislabel, random_scores)), 4),
        "precision_at_k": round(precision_at_k(random_scores, is_mislabel, len(mislabel_idx)), 4),
    }

    return {
        "n_train": n_train, "k_backdoor": k_backdoor, "k_distractor": len(distractor_idx),
        "k_mislabel": len(mislabel_idx),
        "overhead_pct": overhead_pct,
        "backdoor_success_rate": backdoor_success_rate,
        "clean_target_rate": clean_target_rate,
        "backdoor_random_auc": backdoor_auc_random,
        "backdoor": backdoor_results,
        "mislabel": mislabel_results,
    }


def run(args):
    import torch
    import torch.nn.functional as F

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

    per_seed = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"[exp37] === seed {seed} ({seed - args.seed + 1}/{args.n_seeds}) ===")
        r = run_one_seed(args, seed, device, torch, F, GradientStore, LoRAGradientLogger, select_lora_linears)
        print(f"  backdoor_success={r['backdoor_success_rate']:.4f} "
              f"clean_control={r['clean_target_rate']:.4f} overhead={r['overhead_pct']:.2f}%")
        for name, v in r["backdoor"].items():
            print(f"  backdoor-{name}: AUC={v['auc']:.4f} (excl_mislabel={v['auc_excl_mislabel']:.4f})")
        per_seed.append(r)

    def agg_scalar(name):
        vals = [r[name] for r in per_seed]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    backdoor_agg = {}
    for name in per_seed[0]["backdoor"]:
        backdoor_agg[name] = {}
        for metric in ("auc", "precision_at_k", "auc_excl_mislabel", "precision_at_k_excl_mislabel"):
            vals = [r["backdoor"][name][metric] for r in per_seed]
            backdoor_agg[name][metric] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    mislabel_agg = {}
    for name in per_seed[0]["mislabel"]:
        mislabel_agg[name] = {}
        for metric in ("auc", "precision_at_k"):
            vals = [r["mislabel"][name][metric] for r in per_seed]
            mislabel_agg[name][metric] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "n_seeds": args.n_seeds,
        "seeds": list(range(args.seed, args.seed + args.n_seeds)),
        "n_train": per_seed[0]["n_train"],
        "k_backdoor": per_seed[0]["k_backdoor"],
        "k_distractor": per_seed[0]["k_distractor"],
        "k_mislabel": per_seed[0]["k_mislabel"],
        "trigger_len": args.trigger_len,
        "backdoor_target_label": args.backdoor_target_label,
        "epochs": args.epochs,
        "proj_dim": args.proj_dim,
        "track_last_n_blocks": args.track,
        "overhead_pct": agg_scalar("overhead_pct"),
        "backdoor_success_rate": agg_scalar("backdoor_success_rate"),
        "clean_target_rate": agg_scalar("clean_target_rate"),
        "backdoor_random_auc": agg_scalar("backdoor_random_auc"),
        "primary_backdoor": backdoor_agg,
        "secondary_mislabel": mislabel_agg,
        "note": "backdoor AUC reported as dot/cosine/trak, each with and without the "
                "mislabeled set in the candidate pool (mislabeled examples have large "
                "gradients that could otherwise dominate a raw dot-product ranking). "
                "distractors (trigger present, correct label kept) are counted as "
                "negatives in every backdoor AUC -- token/representation matching alone "
                "cannot separate them from poisoned examples; repr_similarity is reported "
                "as exactly that baseline for direct comparison.",
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
    np.savez(npz_fn, per_seed=np.array(per_seed, dtype=object))
    print(f"saved -> {npz_fn} (per-seed raw results)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--data", choices=["synthetic", "sst2"], default="synthetic")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_trigger_test", type=int, default=100,
                    help="number of held-out clean examples to trigger (and, separately, "
                         "to leave clean as a control) at test time")
    ap.add_argument("--plant_frac", type=float, default=0.05,
                    help="fraction of training examples planted as backdoor (primary)")
    ap.add_argument("--distractor_frac", type=float, default=0.05,
                    help="fraction planted with the trigger but CORRECT label kept -- "
                         "defeats token/representation-matching baselines")
    ap.add_argument("--mislabel_frac", type=float, default=0.05,
                    help="fraction of (remaining) training examples planted as mislabeled (secondary)")
    ap.add_argument("--trigger_len", type=int, default=5,
                    help="max tokens overwritten at the true end of real content (hf backend: "
                         "encoded length of the fixed trigger phrase, capped to this)")
    ap.add_argument("--backdoor_target_label", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="first seed; runs seed..seed+n_seeds-1")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", default=None, help="output path override (default: auto)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
