"""Fast diagnosis of why an HF LoRA classifier may sit at chance accuracy.

Trains ONLY the target model (no retrains) and prints the four things that
distinguish the likely causes:
  1. trainable parameter names   -> is the classification head (score) actually
     trainable under PEFT modules_to_save, or is a random head frozen?
  2. pad_token_id (config + tokenizer) and padding side -> GPTNeoX/decoder
     sequence-classification pools the LAST NON-PAD token; a wrong pad id reads
     the wrong position and every input looks identical -> chance.
  3. train loss per epoch        -> is the model learning at all?
  4. test prediction distribution -> collapsed to one class?

Usage:
  python diag_hf_classifier.py --model EleutherAI/pythia-160m --data sst2 \
     --n 500 --epochs 5 --seq 32 --lr 1e-3 --device cuda
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from exp27_lds_quality import build_hf_classifier, load_sst2, synthetic_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--data", choices=["sst2", "synthetic"], default="sst2")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(0); np.random.seed(0)

    # ---- data ----
    if args.data == "sst2":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        print(f"[tok] pad_token={tok.pad_token!r} pad_token_id={tok.pad_token_id} "
              f"eos_token_id={tok.eos_token_id} padding_side={tok.padding_side}")
        Xtr, ytr, Xte, yte, vocab = load_sst2(args.n, args.n_test, args.seq, args.model, 0)
    else:
        vocab = 1000
        Xtr, ytr = synthetic_data(args.n, args.seq, vocab, 0)
        Xte, yte = synthetic_data(args.n_test, args.seq, vocab, 1)
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    print(f"[data] {args.data}: n_train={len(Xtr)} n_test={len(Xte)} "
          f"label balance train={np.bincount(ytr).tolist()} test={np.bincount(yte).tolist()}")

    # ---- model ----
    torch.manual_seed(1234); np.random.seed(1234)
    model = build_hf_classifier(args.model, r=args.rank).to(dev)
    print(f"[model] config.pad_token_id={model.config.pad_token_id} "
          f"dtype={next(model.parameters()).dtype}")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    head_trainable = [n for n in trainable if "score" in n or "classifier" in n]
    lora_trainable = [n for n in trainable if "lora_" in n]
    print(f"[trainable] total tensors={len(trainable)}  lora={len(lora_trainable)}  "
          f"head={len(head_trainable)}")
    print(f"[trainable] HEAD params: {head_trainable if head_trainable else '*** NONE — head is FROZEN (random head -> chance) ***'}")
    print(f"[trainable] sample lora: {lora_trainable[:2]}")

    # ---- train, print loss/epoch ----
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        tot = 0.0; nb = 0
        for s in range(0, len(Xtr), args.batch):
            b = perm[s:s + args.batch]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad(set_to_none=True)
            out = model(xb); logits = out.logits
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[train] epoch {ep+1}/{args.epochs}  mean loss={tot/max(nb,1):.4f}")

    # ---- eval ----
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).logits.argmax(1)
    acc = float((pred == yte_t).float().mean())
    dist = torch.bincount(pred, minlength=2).tolist()
    print(f"[eval] test acc={acc:.4f}  pred distribution (argmax)={dist}  "
          f"{'*** COLLAPSED to one class ***' if 0 in dist else ''}")
    print(f"[verdict] {'OK - model learns, signal viable' if acc >= 0.75 else 'NOT LEARNING - fix before any LDS run'}")


if __name__ == "__main__":
    main()
