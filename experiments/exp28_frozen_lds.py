"""exp28 — Frozen-backbone LDS: last-layer attribution on real GPT-2 features.

Workstream C, reliable version. GPT-2 LoRA *fine-tuning* attribution is an
intrinsically weak-signal regime (exp27: near-zero LDS for Traceprop AND TRAK —
a pretrained model barely depends on any single fine-tuning example). The regime
where last-layer attribution is *well-posed and exact* is the frozen backbone +
trained linear head: this is the tabular LDS setting (Traceprop-LL 0.88) applied
to real GPT-2 features.

Setup: freeze GPT-2, extract pooled features once, train a linear head (logistic
regression). Retrain the head on M random data subsets (cheap) for the ground
truth. Attribute via the exact per-sample last-layer gradient
g_j = (p_j − onehot(y_j)) ⊗ feature_j, projected by sparse JL. Compare:

  Traceprop-LL (dot)      — exact last-layer influence, dot product
  Traceprop-LL + TRAK     — same grads, TRAK estimator (ΦᵀΦ+λI)⁻¹
  Random                  — null

This validates that last-layer/last-block attribution (the <1%-overhead,
~100×-cheaper config from exp25/26) recovers strong, correct attribution on a
real pretrained LLM — the quality side of the cost win.

Backends: --backend hf (frozen GPT-2, needs transformers+datasets) and
--backend tiny (frozen random features, validates the pipeline on CPU).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def synthetic_features(n, d, seed, noise=0.1):
    """Linearly-separable-ish features with a planted direction + noise, so a
    linear head learns a non-trivial, data-dependent boundary (LDS-meaningful).
    The planted direction `w` is fixed (seed-independent) so train and test
    share the same labeling function."""
    rng = np.random.default_rng(seed)
    w = np.random.default_rng(12345).standard_normal(d)  # shared across splits
    F = rng.standard_normal((n, d)).astype(np.float32)
    logits = F @ w + noise * rng.standard_normal(n)
    y = (logits > np.median(logits)).astype(np.int64)
    flip = rng.random(n) < 0.15
    y[flip] = 1 - y[flip]
    return F, y


def gpt2_features(n_train, n_test, seq, model_name, seed, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        ds = load_dataset("nyu-mll/glue", "sst2")
    except Exception:
        ds = load_dataset("stanfordnlp/sst2")
    tr = ds["train"].shuffle(seed=seed).select(range(n_train))
    te = ds["validation"].select(range(min(n_test, len(ds["validation"]))))

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def feats(split):
        out = []
        sents = list(split["sentence"])
        for s in range(0, len(sents), 32):
            enc = tok(sents[s:s + 32], truncation=True, padding="max_length",
                      max_length=seq, return_tensors="pt").to(device)
            with torch.no_grad():
                h = model(**enc).last_hidden_state          # (B, T, d)
                mask = enc["attention_mask"][:, :, None].float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool
            out.append(pooled.cpu().numpy())
        return np.concatenate(out).astype(np.float32), np.array(split["label"], dtype=np.int64)

    Ftr, ytr = feats(tr)
    Fte, yte = feats(te)
    return Ftr, ytr, Fte, yte


# --------------------------------------------------------------------------
def run(args):
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr
    from traceprop.attribution.gradient_store import GradientStore

    if args.backend == "tiny":
        d = args.feat_dim
        Ftr, ytr = synthetic_features(args.n_train, d, args.seed)
        Fte, yte = synthetic_features(args.n_test, d, args.seed + 1)
    else:
        Ftr, ytr, Fte, yte = gpt2_features(args.n_train, args.n_test, args.seq,
                                           args.model, args.seed, args.device)
    # standardize (fit on train)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd
    Fte = (Fte - mu) / sd
    n_train, n_test, d = len(Ftr), len(Fte), Ftr.shape[1]
    print(f"[exp28] features: train {Ftr.shape} test {Fte.shape}")

    def fit(idx):
        clf = LogisticRegression(C=args.C, max_iter=1000)
        clf.fit(Ftr[idx], ytr[idx])
        return clf

    def margins(clf, F, y):
        # signed margin = decision_function * (2y-1) for the true class
        df = clf.decision_function(F)  # (n,) for binary
        return df * (2 * y - 1)

    # ---- target model + exact per-sample last-layer gradients ----
    target = fit(np.arange(n_train))
    acc = float((target.predict(Fte) == yte).mean())
    print(f"[exp28] target test accuracy: {acc:.4f}")

    def grad_matrix(clf, F, y):
        """Exact per-sample gradient of logistic loss wrt the linear head:
        g_j = (p_j − y_j) · feature_j  (binary). Projected by sparse JL."""
        p = 1.0 / (1.0 + np.exp(-clf.decision_function(F)))  # P(class=1)
        G = ((p - y)[:, None] * F).astype(np.float32)          # (n, d)
        store = GradientStore(proj_dim=min(args.proj_dim, d), seed=42)
        store.log_batch(G)
        return store.get_projected_matrix()

    gtr = grad_matrix(target, Ftr, ytr)
    gte = grad_matrix(target, Fte, yte)

    # ---- ground-truth LDS margins over subsets ----
    print(f"[exp28] retraining {args.n_subsets} subsets (frac={args.subset_frac}) ...")
    rng = np.random.default_rng(args.seed)
    k = int(args.subset_frac * n_train)
    masks = np.zeros((args.n_subsets, n_train), dtype=np.float32)
    M = np.zeros((args.n_subsets, n_test), dtype=np.float32)
    for m in range(args.n_subsets):
        sub = rng.choice(n_train, size=k, replace=False)
        masks[m, sub] = 1.0
        M[m] = margins(fit(sub), Fte, yte)

    # ---- attribution + LDS ----
    def lds(attr):
        pred = masks @ attr.T
        rs = [spearmanr(pred[:, i], M[:, i]).correlation for i in range(n_test)]
        rs = [r for r in rs if not np.isnan(r)]
        return float(np.mean(rs)), float(np.std(rs))

    def trak(gtr, gte, lam=None):
        dd = gtr.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(gtr.T @ gtr) / dd
        H = gtr.T @ gtr + lam * np.eye(dd, dtype=np.float32)
        return gte @ np.linalg.solve(H, gtr.T)

    rng2 = np.random.default_rng(0)
    results = {
        "traceprop_ll_dot": lds(gte @ gtr.T),
        "traceprop_ll_trak": lds(trak(gtr, gte)),
        "random": lds(rng2.standard_normal((n_test, n_train)).astype(np.float32)),
    }
    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "frozen-synth",
        "n_train": n_train, "n_test": n_test, "feat_dim": d,
        "n_subsets": args.n_subsets, "subset_frac": args.subset_frac, "C": args.C,
        "proj_dim": min(args.proj_dim, d), "target_test_acc": round(acc, 4),
        "lds": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in results.items()},
    }
    print("\n=== LDS (frozen GPT-2 features + linear head) ===")
    for k, v in out["lds"].items():
        print(f"  {k:<22} {v['mean']:+.4f} ± {v['std']:.4f}")
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    fn = f"results/exp28_{args.backend}_{out['model'].replace('/', '_')}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_subsets", type=int, default=128)
    ap.add_argument("--subset_frac", type=float, default=0.5)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--feat_dim", type=int, default=64)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
