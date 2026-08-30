"""exp27 — LDS quality parity: last-block (Traceprop) vs all-layers (TRAK).

Workstream C of the MLSys plan. The cost win (exp25/exp26) only matters if the
cheap last-block attribution is *as good* as full-model post-hoc attribution.
This measures the Linear Datamodeling Score (LDS, Park et al. 2023) for:

  - Traceprop-LL          : last-block LoRA grads, dot-product influence
  - Traceprop-LL + TRAK   : last-block LoRA grads, TRAK estimator (ΦᵀΦ+λI)⁻¹
  - TRAK (all layers)     : all-layer LoRA grads, TRAK estimator  ← the baseline
  - Random                : shuffled scores (null)

All four use the *same* gradient mechanism (`LoRAGradientLogger`); the only
difference is which layers are tracked (`--track 1` vs `--track 0`) and the
estimator. If last-block ≈ all-layers on LDS, the cost win has no quality cost.

LDS: for each test example z, over M random training subsets S, the datamodel
predicts margin(z; S) ≈ Σ_{j∈S} attr_j(z). We correlate (Spearman) that
prediction with the *actual* margin from retraining on S, and average over z.

Backends:
  --backend tiny : self-contained synthetic text classification + tiny
                   transformer classifier. CPU. Validates the whole LDS pipeline.
  --backend hf   : GPT-2 + PEFT LoRA sequence classifier on SST-2 (needs
                   transformers, peft, datasets). Single GPU. The paper number.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def synthetic_data(n, seq, vocab, seed, noise=0.15):
    """Learnable binary task with a weak, noisy signal so the model does NOT hit
    100% accuracy (saturated margins give degenerate LDS — the MNIST-0v1 trap).
    Label ≈ (more token-1 than token-2), but with few signal tokens and `noise`
    label flips, so per-example informativeness varies → LDS-meaningful."""
    rng = np.random.default_rng(seed)
    X = rng.integers(3, vocab, size=(n, seq))
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c1 = rng.integers(0, 3)          # 0-2 token-1s
        c2 = rng.integers(0, 3)          # 0-2 token-2s
        pos = rng.choice(seq, size=c1 + c2, replace=False)
        X[i, pos[:c1]] = 1
        X[i, pos[c1:]] = 2
        y[i] = 1 if c1 > c2 else (0 if c2 > c1 else rng.integers(0, 2))
        if rng.random() < noise:
            y[i] = 1 - y[i]
    return X.astype(np.int64), y


def load_sst2(n_train, n_test, seq, model_name, seed):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Legacy load_dataset("glue","sst2") uses a dataset *script* that breaks with
    # newer huggingface_hub URI parsing. Use namespaced data-only repos instead.
    try:
        ds = load_dataset("nyu-mll/glue", "sst2")
    except Exception:
        ds = load_dataset("stanfordnlp/sst2")
    rng = np.random.default_rng(seed)
    tr = ds["train"].shuffle(seed=seed).select(range(n_train))
    te = ds["validation"].select(range(min(n_test, len(ds["validation"]))))

    def enc(split):
        out = tok([x for x in split["sentence"]], truncation=True, padding="max_length",
                  max_length=seq, return_tensors="np")
        return out["input_ids"].astype(np.int64), np.array(split["label"], dtype=np.int64)

    Xtr, ytr = enc(tr)
    Xte, yte = enc(te)
    return Xtr, ytr, Xte, yte, tok.vocab_size


# --------------------------------------------------------------------------
# Models (classifier variants)
# --------------------------------------------------------------------------
def build_tiny_classifier(vocab, d=128, n_heads=4, n_blocks=2, seq=32, r=8, n_classes=2):
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
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
            self.q, self.k, self.v, self.o = (LoRALinear(d, d, r) for _ in range(4))
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
            att = att.softmax(-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, C)
            x = x + self.o(y)
            x = x + self.fc2(torch.relu(self.fc1(self.ln2(x))))
            return x

    class TinyClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(seq, d)
            self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
            self.lnf = nn.LayerNorm(d)
            self.score = nn.Linear(d, n_classes)

        def forward(self, idx):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
            for blk in self.blocks:
                x = blk(x)
            return self.score(self.lnf(x).mean(1))  # mean-pool → logits

    torch.manual_seed(0)
    return TinyClf()


def build_hf_classifier(model_name, r=8, n_classes=2):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id
    target = ["c_attn"] if "gpt2" in model_name else ["query_key_value"]
    cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=target, task_type="SEQ_CLS",
                     modules_to_save=["score", "classifier"])
    return get_peft_model(base, cfg)


# --------------------------------------------------------------------------
# LDS harness
# --------------------------------------------------------------------------
def run(args):
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr

    from traceprop.attribution.gradient_store import GradientStore
    from traceprop.llm import LoRAGradientLogger, select_lora_linears

    device = args.device
    torch.manual_seed(args.seed)

    # ---- data ----
    if args.backend == "tiny":
        vocab = args.vocab
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
        # Deterministic init: every subset-retrain must start from the SAME
        # parameters so the only variation across subsets is the training data
        # (a hard requirement for LDS — a random head/LoRA init per subset adds
        # variance that drowns the attribution signal).
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

    def train(model, idx, epochs, lr):
        """Train (LoRA + head) on the given training indices; return the model."""
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

    def test_margins(model):
        """Signed margin (logit_correct - logit_other) on the test set."""
        model.eval()
        with torch.no_grad():
            lo = logits(model, Xte_t)
            correct = lo.gather(1, yte_t[:, None]).squeeze(1)
            other = lo.clone()
            other.scatter_(1, yte_t[:, None], float("-inf"))
            margin = correct - other.max(1).values
        model.train()
        return margin.detach().cpu().numpy()

    def collect_grads(model, X, y, track, kind):
        """Projected per-sample gradients (n, proj_dim) via LoRAGradientLogger.
        kind='loss' → grad of CE loss (train); kind='margin' → grad of the signed
        margin (test query), matching TRAK's output-function gradient."""
        store = GradientStore(proj_dim=args.proj_dim, seed=42)
        last_n = None if track <= 0 else track
        targets = select_lora_linears(model, ("lora_A", "lora_B"), last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)
        n = len(X)
        for s in range(0, n, args.batch):
            xb = X[s:s + args.batch]
            yb = y[s:s + args.batch]
            model.zero_grad(set_to_none=True)
            lo = logits(model, xb)
            if kind == "loss":
                obj = F.cross_entropy(lo, yb, reduction="sum")
            else:  # margin
                correct = lo.gather(1, yb[:, None]).squeeze(1)
                other = lo.clone()
                other.scatter_(1, yb[:, None], float("-inf"))
                obj = (correct - other.max(1).values).sum()
            obj.backward()
            lg.flush_step(sample_indices=range(s, s + len(xb)))
        lg.detach()
        return store.get_projected_matrix(), targets

    # ---- 1) train target model on full data, collect grads ----
    print(f"[exp27] training target model on {n_train} examples ...")
    target_model = train(new_model(), np.arange(n_train), args.epochs, args.lr)
    acc = float((logits(target_model, Xte_t).argmax(1) == yte_t).float().mean())
    print(f"[exp27] target test accuracy: {acc:.4f}")

    grads = {}  # method -> (train_grads, test_grads)
    # Loss gradient for both train and test: <grad_loss(z), grad_loss(j)> is
    # positive for helpful same-class examples, which increase the test margin,
    # so the influence score correlates *positively* with the retrained margin.
    for name, track in [("last", args.track), ("all", 0)]:
        gtr, _ = collect_grads(target_model, Xtr_t, ytr_t, track, "loss")
        gte, _ = collect_grads(target_model, Xte_t, yte_t, track, "loss")
        grads[name] = (gtr, gte)

    # ---- 2) ground-truth LDS margins from subset retraining ----
    print(f"[exp27] retraining {args.n_subsets} subsets (frac={args.subset_frac}) ...")
    rng = np.random.default_rng(args.seed)
    masks = np.zeros((args.n_subsets, n_train), dtype=np.float32)
    margins = np.zeros((args.n_subsets, n_test), dtype=np.float32)
    k = int(args.subset_frac * n_train)
    for m in range(args.n_subsets):
        sub = rng.choice(n_train, size=k, replace=False)
        masks[m, sub] = 1.0
        model_m = train(new_model(), sub, args.epochs, args.lr)
        margins[m] = test_margins(model_m)
        if (m + 1) % max(1, args.n_subsets // 10) == 0:
            print(f"  subset {m+1}/{args.n_subsets}")

    # ---- 3) attribution scores + LDS per method ----
    def lds_for(attr):  # attr: (n_test, n_train)
        pred = masks @ attr.T  # (n_subsets, n_test)
        rs = [spearmanr(pred[:, i], margins[:, i]).correlation for i in range(n_test)]
        rs = [r for r in rs if not np.isnan(r)]
        return float(np.mean(rs)), float(np.std(rs))

    def dot_scores(gtr, gte):
        return gte @ gtr.T

    def trak_scores(gtr, gte, lam=None):
        d = gtr.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(gtr.T @ gtr) / d
        H = gtr.T @ gtr + lam * np.eye(d, dtype=np.float32)
        return gte @ np.linalg.solve(H, gtr.T)

    results = {}
    gtr_l, gte_l = grads["last"]
    gtr_a, gte_a = grads["all"]
    # dot and TRAK estimator for BOTH last-block (Traceprop) and all-layers (TRAK)
    # → the parity comparison is controlled: only the tracked layers differ.
    results["last_block_dot"] = lds_for(dot_scores(gtr_l, gte_l))
    results["last_block_trak"] = lds_for(trak_scores(gtr_l, gte_l))
    results["all_layers_dot"] = lds_for(dot_scores(gtr_a, gte_a))
    results["all_layers_trak"] = lds_for(trak_scores(gtr_a, gte_a))
    rng2 = np.random.default_rng(0)
    results["random"] = lds_for(rng2.standard_normal((n_test, n_train)).astype(np.float32))

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
    print("\n=== LDS (mean ± std over test examples) ===")
    for k, v in out["lds"].items():
        print(f"  {k:<22} {v['mean']:+.4f} ± {v['std']:.4f}")
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    fn = f"results/exp27_{args.backend}_{out['model'].replace('/', '_')}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
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
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
