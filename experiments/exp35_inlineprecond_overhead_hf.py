"""Run B: in-TRAINING overhead of factored logging + inline K-FAC covariance on a
real HF model (e.g. Pythia-1B) on GPU, head-to-head against LogIX's total
attribution cost (its logging pass + its SEPARATE covariance pass).

Traceprop configs, measured in a real LoRA training loop (fwd/bwd/opt.step),
interleaved, warmup dropped, cuda-synchronized:
  (a) train only                                   -> baseline
  (b) train + factored logging (inline_precond off)
  (c) train + factored logging + inline covariance -> the full Traceprop path

Reports medians + a one-sided Mann-Whitney U test (same rigor as the speed
results): total (c vs a) and covariance-only (c vs b).

With --with_logix, also times LogIX on the SAME model/batches:
  - LogIX logging pass wall-clock (its per-example gradient logging)
  - LogIX covariance pass wall-clock (the separate sweep it needs for preconditioning)
  LogIX total attribution = logging pass + covariance pass (both EXTRA passes over
  the data), vs Traceprop folding everything into the single training pass.

Colab A100:
  python exp35_inlineprecond_overhead_hf.py --model EleutherAI/pythia-1b --device cuda \
     --steps 200 --batch 8 --seq 128 --repeats 10 --with_logix
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp27_lds_quality import build_hf_classifier
from traceprop.attribution.gradient_store import GradientStore
from traceprop.llm import LoRAGradientLogger, select_lora_linears


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build(model_name, rank, device):
    torch.manual_seed(0)
    return build_hf_classifier(model_name, r=rank).to(device)


def tp_loop(model, batches, mode, kfac, track, device):
    """Traceprop timing. mode in {'none','log','log_cov'}. Returns seconds."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    lg = None
    if mode in ("log", "log_cov"):
        patterns = ("lora_A", "lora_B", "score", "classifier")
        last_n = None if track <= 0 else track
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        store = GradientStore(proj_dim=512, seed=42)
        lg = LoRAGradientLogger(store, targets, proj_dim=512, factored=True,
                                kfac=kfac, inline_precond=(mode == "log_cov"))
    _sync(device)
    t0 = time.perf_counter()
    for xb, yb in batches:
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        logits = out if not hasattr(out, "logits") else out.logits
        F.cross_entropy(logits, yb, reduction="mean").backward()
        if lg is not None:
            lg.flush_step(buffer=True)
        opt.step()
    if lg is not None:
        lg.drain()
    _sync(device)
    dt = time.perf_counter() - t0
    if lg is not None:
        lg.detach()
    return dt


def mwu_one_sided_greater(a, b):
    """P-value that distribution a is stochastically greater than b (a=with, b=without)."""
    from scipy.stats import mannwhitneyu
    try:
        return float(mannwhitneyu(a, b, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def logix_arm(model_name, rank, kfac, track, batches, device):
    """Time LogIX FAIRLY: its per-example gradient logging runs INLINE in a real
    training loop (with opt.step, exactly like Traceprop and like exp31's ~3.2%),
    NOT as a separate pass -- so we don't inflate its cost. Only the covariance
    pass (which LogIX genuinely needs as an extra sweep for K-FAC preconditioning)
    is counted as extra work. Returns seconds for: baseline training on the same
    watched model, training WITH inline logging, and the separate covariance pass.
    """
    import logix
    from logix_strict import install_strict_warnings, patch_loralinear_weight_proxy
    install_strict_warnings(); patch_loralinear_weight_proxy()

    lx = build(model_name, rank, device)   # same PEFT model / tracked LoRA linears as Traceprop
    tracked = [n for n, m in lx.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n)]
    if track > 0:
        import re
        def bidx(n):
            m = re.search(r"(?:^|\.)(?:h|layers)\.(\d+)\.", n)
            return int(m.group(1)) if m else None
        idxs = sorted({bidx(n) for n in tracked if bidx(n) is not None})
        keep = set(idxs[-track:])
        tracked = [n for n in tracked if bidx(n) in keep]

    run_ = logix.LogIX(project=f"ovh_{os.getpid()}", config="exp31_config.yaml")
    run_.watch(lx, name_filter=tracked, type_filter=[nn.Linear])
    trainable = [p for p in lx.parameters() if p.requires_grad]

    ids = {"n": 0}
    def dids(bs):
        out = [str(ids["n"] + i) for i in range(bs)]; ids["n"] += bs; return out

    def train_loop(log_active):
        """One training pass (fwd/bwd/opt.step). log_active -> LogIX logs inline."""
        opt = torch.optim.Adam(trainable, lr=1e-4)
        if log_active:
            run_.setup({"grad": ["log"]}); run_.save(True)
        _sync(device); t0 = time.perf_counter()
        for xb, yb in batches:
            ids["n"] = 0
            opt.zero_grad(set_to_none=True)
            if log_active:
                with run_(data_id=dids(len(xb))):
                    out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
                    F.cross_entropy(logits, yb, reduction="sum").backward()
            else:
                out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
                F.cross_entropy(logits, yb, reduction="sum").backward()
            opt.step()
        if log_active:
            run_.finalize()
        _sync(device)
        return time.perf_counter() - t0

    none_s = train_loop(False)   # baseline on the watched model, logging OFF
    log_s = train_loop(True)     # same loop, LogIX logging ON (inline, like Traceprop)

    # covariance pass -- the ONLY genuinely extra pass LogIX needs (for K-FAC precond)
    run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
    _sync(device); t0 = time.perf_counter()
    for xb, yb in batches:
        ids["n"] = 0
        with run_(data_id=dids(len(xb))):
            lx.zero_grad(set_to_none=True)
            out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
            F.cross_entropy(logits, yb, reduction="sum").backward()
    _sync(device)
    cov_s = time.perf_counter() - t0
    return {"logix_none_s": none_s, "logix_log_s": log_s, "logix_cov_s": cov_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--kfac", type=int, default=8)
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--with_logix", action="store_true")
    ap.add_argument("--out", default="results/exp35_inlineprecond_overhead_hf.json")
    args = ap.parse_args()

    device = args.device
    model = build(args.model, args.rank, device)
    n_tracked = len(select_lora_linears(
        model, ("lora_A", "lora_B", "score", "classifier"),
        last_n_blocks=(None if args.track <= 0 else args.track)))
    g = torch.Generator().manual_seed(0)
    batches = [(torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device),
                torch.randint(0, 2, (args.batch,), generator=g).to(device))
               for _ in range(args.steps)]

    print(f"[overhead-hf] {args.model} on {device}, {n_tracked} tracked layers, kfac={args.kfac}, "
          f"{args.steps} steps x b{args.batch} x s{args.seq}, {args.repeats} repeats")

    a, b, c = [], [], []
    for r in range(args.repeats + 1):
        da = tp_loop(model, batches, "none", args.kfac, args.track, device)
        db = tp_loop(model, batches, "log", args.kfac, args.track, device)
        dc = tp_loop(model, batches, "log_cov", args.kfac, args.track, device)
        if r == 0:
            continue
        a.append(da); b.append(db); c.append(dc)
        print(f"  r{r}: none={da:.3f} log={db:.3f} log_cov={dc:.3f}")
    a, b, c = map(np.array, (a, b, c))

    res = {
        "model": args.model, "device": device, "n_tracked_layers": n_tracked,
        "kfac": args.kfac, "track": args.track, "steps": args.steps,
        "batch": args.batch, "seq": args.seq, "repeats": args.repeats,
        "none_s_median": round(float(np.median(a)), 4),
        "log_s_median": round(float(np.median(b)), 4),
        "log_cov_s_median": round(float(np.median(c)), 4),
        "total_overhead_pct": round(float(np.median((c - a) / a) * 100), 3),
        "logging_only_pct": round(float(np.median((b - a) / a) * 100), 3),
        "covariance_only_pct": round(float(np.median((c - b) / b) * 100), 3),
        "mwu_p_total_gt_plain": round(mwu_one_sided_greater(c, a), 5),
        "mwu_p_cov_gt_log": round(mwu_one_sided_greater(c, b), 5),
    }

    if args.with_logix:
        lg_none, lg_log, lg_cov = [], [], []
        for r in range(max(3, args.repeats // 3) + 1):
            d = logix_arm(args.model, args.rank, args.kfac, args.track, batches, device)
            if r == 0:
                continue
            lg_none.append(d["logix_none_s"]); lg_log.append(d["logix_log_s"]); lg_cov.append(d["logix_cov_s"])
            print(f"  logix r{r}: none={d['logix_none_s']:.3f} log_inline={d['logix_log_s']:.3f} "
                  f"cov_pass={d['logix_cov_s']:.3f}")
        lg_none, lg_log, lg_cov = np.array(lg_none), np.array(lg_log), np.array(lg_cov)
        # FAIR accounting: LogIX logging runs INLINE during training (like Traceprop and
        # like exp31's ~3.2%), so its overhead is (log - none)/none, NOT a full extra pass.
        # ONLY the covariance pass is genuinely extra work.
        inline_pct = float(np.median((lg_log - lg_none) / lg_none) * 100)
        cov_pass_pct = float(np.median(lg_cov / lg_none) * 100)
        res["logix"] = {
            "logix_none_s_median": round(float(np.median(lg_none)), 4),
            "logix_log_s_median": round(float(np.median(lg_log)), 4),
            "logix_cov_pass_s_median": round(float(np.median(lg_cov)), 4),
            "logix_inline_logging_pct": round(inline_pct, 3),
            "logix_covariance_pass_pct_of_training": round(cov_pass_pct, 3),
            "logix_total_attribution_pct": round(inline_pct + cov_pass_pct, 3),
            "note": "FAIR: LogIX logging is measured INLINE in the training loop "
                    "(logix_inline_logging_pct, comparable to exp31 ~3.2%); only the covariance "
                    "pass is extra (logix_covariance_pass_pct_of_training). LogIX total attribution "
                    f"= sum of the two. Traceprop total is +{res['total_overhead_pct']}% in ONE "
                    "pass (no separate covariance pass).",
        }

    print(json.dumps(res, indent=2))
    print(f"\n[overhead-hf] Traceprop TOTAL +{res['total_overhead_pct']}% "
          f"(cov-only +{res['covariance_only_pct']}%, MWU p_total={res['mwu_p_total_gt_plain']}, "
          f"p_cov={res['mwu_p_cov_gt_log']})")
    if args.with_logix:
        lx = res["logix"]
        print(f"[overhead-hf] LogIX total attribution +{lx['logix_total_attribution_pct']}% "
              f"= inline logging +{lx['logix_inline_logging_pct']}% + covariance pass "
              f"+{lx['logix_covariance_pass_pct_of_training']}% (of a training pass)  "
              f"vs Traceprop +{res['total_overhead_pct']}% single pass")
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
