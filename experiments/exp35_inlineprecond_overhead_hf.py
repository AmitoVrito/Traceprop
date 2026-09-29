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

Colab A100, 1B:
  python exp35_inlineprecond_overhead_hf.py --model EleutherAI/pythia-1b --device cuda \
     --steps 200 --batch 8 --seq 128 --repeats 10 --with_logix

Colab A100, 7B (dtype auto-detects bf16 on 40GB / fp32 on 80GB; skips the LogIX-PCA
arm since full-dim covariance is prohibitive at this scale and PCA is dominated by
LogIX-random on both speed and quality at 1B already; skips Traceprop's logging-only
arm to keep the run short):
  python exp35_inlineprecond_overhead_hf.py --model EleutherAI/pythia-6.9b --device cuda \
     --track 0 --kfac 8 --dtype auto --steps 50 --batch 8 --seq 128 --repeats 5 \
     --with_logix --skip_logix_pca --skip_log_only \
     --out results/exp35_inlineprecond_overhead_hf_pythia7b.json
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


def resolve_dtype(requested, device):
    """--dtype auto: bf16 on <=40GB GPUs (7B in fp32 wouldn't fit), fp32 above that
    (matches the 1B run's methodology when there's headroom). Never silently picks a
    dtype different from what's recorded in the output JSON."""
    if requested != "auto":
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[requested]
    if device != "cuda":
        return torch.float32
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return torch.bfloat16 if total_gb <= 40 else torch.float32


def build(model_name, rank, device, dtype=torch.float32):
    torch.manual_seed(0)
    # build_hf_classifier (exp27_lds_quality.py) forces fp32 internally (transformers
    # 5.x honours config torch_dtype, often fp16, which destabilises LoRA retraining);
    # cast AFTER building so every arm (plain/Traceprop/LogIX) sees the SAME requested
    # dtype instead of silently diverging from it.
    return build_hf_classifier(model_name, r=rank).to(device=device, dtype=dtype)


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


def plain_fwd_bwd(model, batches, device):
    """Sanity reference: one fwd+bwd pass over the batches, NO LogIX, NO opt.step.
    The LogIX covariance pass should be close to this; a covariance pass many times
    a plain fwd+bwd would signal a misconfiguration (e.g. full-dim covariance)."""
    _sync(device); t0 = time.perf_counter()
    for xb, yb in batches:
        model.zero_grad(set_to_none=True)
        out = model(xb); logits = out if not hasattr(out, "logits") else out.logits
        F.cross_entropy(logits, yb, reduction="sum").backward()
    _sync(device)
    return time.perf_counter() - t0


def logix_arm(model_name, rank, kfac, track, batches, device, init_strategy="pca",
              dtype=torch.float32):
    """Time LogIX FAIRLY: its per-example gradient logging runs INLINE in a real
    training loop (with opt.step, exactly like Traceprop and like exp31's ~3.2%),
    NOT as a separate pass -- so we don't inflate its cost. Only the covariance
    pass (which LogIX genuinely needs as an extra sweep for K-FAC preconditioning)
    is counted as extra work. Returns seconds for: baseline training on the same
    watched model, training WITH inline logging, and the separate covariance pass.
    """
    import logix
    from logix_strict import (install_strict_warnings, patch_loralinear_weight_proxy,
                              assert_pca_init_took_effect)
    install_strict_warnings(); patch_loralinear_weight_proxy()

    # VETTED exp31 LogIX setup: watch + restore_trainable (watch() freezes all
    # non-tracked params, which silently makes the LogIX model train less than the
    # plain baseline -> the exp31 negative-overhead bug); PCA covariance pass;
    # add_lora() for storage-matched rank compression; restore_trainable AGAIN
    # (add_lora calls watch() internally, re-freezing). ALL setup is outside the
    # timed regions -- only the data-pass loops are timed.
    lx = build(model_name, rank, device, dtype=dtype)
    trainable = [p for p in lx.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable}
    def restore_trainable():
        for p in lx.parameters():
            if id(p) in trainable_ids:
                p.requires_grad = True

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
    run_.config.lora.init = init_strategy
    run_.watch(lx, name_filter=tracked, type_filter=[nn.Linear])
    restore_trainable()

    ids = {"n": 0}
    def dids(bs):
        out = [str(ids["n"] + i) for i in range(bs)]; ids["n"] += bs; return out

    def fwd_bwd_pass():
        """One fwd+bwd pass over the batches (no opt.step). Timed region only."""
        _sync(device); t0 = time.perf_counter()
        for xb, yb in batches:
            ids["n"] = 0
            lx.zero_grad(set_to_none=True)
            with run_(data_id=dids(len(xb))):
                out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
                F.cross_entropy(logits, yb, reduction="sum").backward()
        _sync(device)
        return time.perf_counter() - t0

    # --- covariance pass (PCA init needs it BEFORE add_lora); timed, setup excluded
    run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
    cov_s = fwd_bwd_pass()
    run_.finalize()

    # --- add_lora (storage-matched rank compression) + restore ---
    run_.add_lora()
    # add_lora() creates its OWN compression modules (logix_lora_A/B/C) that default
    # to fp32 regardless of the base model's dtype -- re-cast the whole (now-wrapped)
    # model so LogIX's own added parameters match dtype too, or a bf16 activation
    # hitting an fp32 logix_lora_* weight raises a dtype-mismatch RuntimeError.
    lx = lx.to(dtype=dtype)
    restore_trainable()
    assert_pca_init_took_effect(run_, init_strategy)

    # --- inline logging pass (add_lora'd model + grad log), timed, setup excluded
    run_.setup({"grad": ["log"]})
    run_.save(True)
    opt = torch.optim.Adam([p for p in lx.parameters() if p.requires_grad], lr=1e-4)
    _sync(device); t0 = time.perf_counter()
    for xb, yb in batches:
        ids["n"] = 0
        opt.zero_grad(set_to_none=True)
        with run_(data_id=dids(len(xb))):
            out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
            F.cross_entropy(logits, yb, reduction="sum").backward()
        opt.step()
    run_.finalize()
    _sync(device)
    log_s = time.perf_counter() - t0
    return {"logix_log_s": log_s, "logix_cov_s": cov_s}


def logix_random_onepass_s(model_name, rank, track, batches, device, dtype=torch.float32):
    """LogIX RANDOM-init single pass: covariance + grad logging TOGETHER
    (setup forward/backward covariance + grad log), post-add_lora. Random init
    needs no pre-add_lora PCA covariance pass, so this is a genuine single pass.
    Verified on tiny (exp35_logix_onepass_equiv.py) that one-pass logging matches
    two-pass exactly (dot corr 1.0); preconditioned scores differ marginally
    (finalize/normalization nuance). Returns the single-pass wall-clock."""
    import logix
    from logix_strict import install_strict_warnings, patch_loralinear_weight_proxy
    install_strict_warnings(); patch_loralinear_weight_proxy()
    lx = build(model_name, rank, device, dtype=dtype)
    trainable_ids = {id(p) for p in lx.parameters() if p.requires_grad}
    def restore():
        for p in lx.parameters():
            if id(p) in trainable_ids:
                p.requires_grad = True
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
    run_ = logix.LogIX(project=f"ovh_rand_{os.getpid()}", config="exp31_config.yaml")
    run_.config.lora.init = "random"
    run_.watch(lx, name_filter=tracked, type_filter=[nn.Linear]); restore()
    run_.add_lora()
    # add_lora()'s own compression modules (logix_lora_A/B/C) default to fp32
    # regardless of the base model's dtype -- re-cast the whole wrapped model so a
    # bf16 activation doesn't hit an fp32 LogIX weight (dtype-mismatch RuntimeError).
    lx = lx.to(dtype=dtype)
    restore()   # random init: no prior covariance pass needed
    run_.setup({"forward": ["covariance"], "backward": ["covariance"], "grad": ["log"]})
    run_.save(True)
    ids = {"n": 0}
    def dids(bs):
        out = [str(ids["n"] + i) for i in range(bs)]; ids["n"] += bs; return out
    opt = torch.optim.Adam([p for p in lx.parameters() if p.requires_grad], lr=1e-4)
    _sync(device); t0 = time.perf_counter()
    for xb, yb in batches:
        ids["n"] = 0
        opt.zero_grad(set_to_none=True)
        with run_(data_id=dids(len(xb))):
            out = lx(xb); logits = out if not hasattr(out, "logits") else out.logits
            F.cross_entropy(logits, yb, reduction="sum").backward()
        opt.step()
    run_.finalize()
    _sync(device)
    return time.perf_counter() - t0


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
    ap.add_argument("--dtype", choices=["fp32", "bf16", "auto"], default="fp32",
                    help="fp32 keeps existing 1B results' methodology unchanged (the "
                         "default). 'auto' picks bf16 on <=40GB GPUs, fp32 above that "
                         "(for large models like 7B where fp32 may not fit on a 40GB "
                         "A100); the SAME resolved dtype is used for every arm so the "
                         "overhead ratios stay apples-to-apples, and it's recorded in "
                         "the output JSON, never silently chosen.")
    ap.add_argument("--skip_logix_pca", action="store_true",
                    help="skip the LogIX-PCA arm (full-dim covariance is prohibitive "
                         "at large scale, and PCA-init is dominated by random-init on "
                         "both speed and quality anyway -- see fairrand)")
    ap.add_argument("--skip_log_only", action="store_true",
                    help="skip Traceprop's logging-only (no covariance) arm; only "
                         "measure plain training and the full log+cov total")
    ap.add_argument("--out", default="results/exp35_inlineprecond_overhead_hf.json")
    args = ap.parse_args()

    device = args.device
    dtype = resolve_dtype(args.dtype, device)
    dtype_name = {torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
    model = build(args.model, args.rank, device, dtype=dtype)
    n_tracked = len(select_lora_linears(
        model, ("lora_A", "lora_B", "score", "classifier"),
        last_n_blocks=(None if args.track <= 0 else args.track)))
    g = torch.Generator().manual_seed(0)
    batches = [(torch.randint(0, args.vocab, (args.batch, args.seq), generator=g).to(device),
                torch.randint(0, 2, (args.batch,), generator=g).to(device))
               for _ in range(args.steps)]

    print(f"[overhead-hf] {args.model} on {device}, dtype={dtype_name} (requested="
          f"{args.dtype}), {n_tracked} tracked layers, kfac={args.kfac}, "
          f"{args.steps} steps x b{args.batch} x s{args.seq}, {args.repeats} repeats")

    a, b, c = [], [], []
    for r in range(args.repeats + 1):
        da = tp_loop(model, batches, "none", args.kfac, args.track, device)
        db = None if args.skip_log_only else tp_loop(model, batches, "log", args.kfac, args.track, device)
        dc = tp_loop(model, batches, "log_cov", args.kfac, args.track, device)
        if r == 0:
            continue
        a.append(da)
        if not args.skip_log_only:
            b.append(db)
        c.append(dc)
        print(f"  r{r}: none={da:.3f} "
              f"{'log=' + format(db, '.3f') + ' ' if not args.skip_log_only else ''}"
              f"log_cov={dc:.3f}")
    a, c = np.array(a), np.array(c)
    b = np.array(b) if not args.skip_log_only else None

    res = {
        "model": args.model, "device": device, "dtype": dtype_name,
        "dtype_requested": args.dtype, "n_tracked_layers": n_tracked,
        "kfac": args.kfac, "track": args.track, "steps": args.steps,
        "batch": args.batch, "seq": args.seq, "repeats": args.repeats,
        "none_s_median": round(float(np.median(a)), 4),
        "log_cov_s_median": round(float(np.median(c)), 4),
        "total_overhead_pct": round(float(np.median((c - a) / a) * 100), 3),
        "mwu_p_total_gt_plain": round(mwu_one_sided_greater(c, a), 5),
    }
    if not args.skip_log_only:
        res.update({
            "log_s_median": round(float(np.median(b)), 4),
            "logging_only_pct": round(float(np.median((b - a) / a) * 100), 3),
            "covariance_only_pct": round(float(np.median((c - b) / b) * 100), 3),
            "mwu_p_cov_gt_log": round(mwu_one_sided_greater(c, b), 5),
        })

    if args.with_logix:
        plain_train = res["none_s_median"]
        plain_fwdbwd = None
        if not args.skip_logix_pca:
            # sanity reference: plain fwd+bwd (no LogIX) on the SAME batches/model class.
            pfb_model = build(args.model, args.rank, device, dtype=dtype)
            plain_fwdbwd = np.median([plain_fwd_bwd(pfb_model, batches, device)
                                      for _ in range(max(3, args.repeats // 3))])
            lg_log, lg_cov = [], []
            for r in range(max(3, args.repeats // 3) + 1):
                d = logix_arm(args.model, args.rank, args.kfac, args.track, batches, device,
                              dtype=dtype)
                if r == 0:
                    continue
                lg_log.append(d["logix_log_s"]); lg_cov.append(d["logix_cov_s"])
                print(f"  logix r{r}: log_inline={d['logix_log_s']:.3f} cov_pass={d['logix_cov_s']:.3f}")
            lg_log, lg_cov = np.array(lg_log), np.array(lg_cov)
            # baseline = the SAME plain-training median used for Traceprop (res none), so both
            # methods' overhead is expressed against identical plain training.
            inline_pct = float((np.median(lg_log) - plain_train) / plain_train * 100)
            cov_pass_pct = float(np.median(lg_cov) / plain_train * 100)
            cov_over_fwdbwd = float(np.median(lg_cov) / plain_fwdbwd)  # sanity: should be ~1
        # LogIX RANDOM-init single-pass (covariance+log together) overhead, for the
        # trade-off table. NOTE: fairrand (tiny-backend LDS) showed random-init is LogIX's
        # BEST quality config, not "lower quality" -- PCA-init is dominated (worse quality
        # AND needs a second pass). Uses the SAME repeat count as Traceprop (not repeats//3)
        # and keeps every raw sample so a real Mann-Whitney test against Traceprop's total
        # is possible, not just a median-vs-median comparison.
        rand_samples = np.array([
            logix_random_onepass_s(args.model, args.rank, args.track, batches, device, dtype=dtype)
            for _ in range(args.repeats)
        ])
        lg_rand = float(np.median(rand_samples))
        rand_onepass_pct_samples = (rand_samples - plain_train) / plain_train * 100
        rand_onepass_pct = float(np.median(rand_onepass_pct_samples))
        # Traceprop's own overhead-pct samples (c = log_cov total time, a = plain baseline),
        # computed the same way, so both sides of the MWU test are apples-to-apples percentages
        # rather than mixing raw seconds against percentages.
        tp_total_pct_samples = (c - a) / a * 100
        mwu_p_tp_lt_logix_random = mwu_one_sided_greater(rand_onepass_pct_samples, tp_total_pct_samples)
        res["logix"] = {
            "logix_random_onepass_s_median": round(float(lg_rand), 4),
            "logix_random_onepass_overhead_pct": round(rand_onepass_pct, 3),
            "logix_random_onepass_repeats": int(args.repeats),
            "logix_random_onepass_overhead_pct_samples": [round(float(x), 3) for x in rand_onepass_pct_samples],
            "traceprop_total_overhead_pct_samples": [round(float(x), 3) for x in tp_total_pct_samples],
            "traceprop_repeats": int(args.repeats),
            "mwu_p_logix_random_gt_traceprop": round(mwu_p_tp_lt_logix_random, 5),
            "note": "VETTED exp31 setup (add_lora + watch/restore, storage-matched). "
                    "LogIX-RANDOM = 1 pass, LogIX's best config on both speed and quality "
                    "(fairrand beats PCA-init on quality too, so PCA is not a stronger "
                    "reference point despite being 2 passes). Traceprop = 1 pass, "
                    f"+{res['total_overhead_pct']}%. mwu_p_logix_random_gt_traceprop: one-sided "
                    "Mann-Whitney, LogIX-random overhead-% samples > Traceprop overhead-% "
                    "samples, both arms at the SAME repeat count (traceprop_repeats == "
                    "logix_random_onepass_repeats == --repeats). All setup EXCLUDED from "
                    "timed regions." + ("" if args.skip_logix_pca else
                    " LogIX-PCA (2 passes: inline logging + separate PCA covariance pass) "
                    "also measured -- see logix_pca_* fields."),
        }
        if not args.skip_logix_pca:
            res["logix"].update({
                "logix_cov_pass_s_median": round(float(np.median(lg_cov)), 4),
                "logix_pca_inline_logging_pct": round(inline_pct, 3),
                "logix_pca_covariance_pass_pct_of_training": round(cov_pass_pct, 3),
                "logix_pca_total_attribution_pct": round(inline_pct + cov_pass_pct, 3),
                "cov_pass_over_plain_fwd_bwd": round(cov_over_fwdbwd, 3),
            })

    print(json.dumps(res, indent=2))
    summary = f"\n[overhead-hf] Traceprop TOTAL +{res['total_overhead_pct']}% (MWU p_total={res['mwu_p_total_gt_plain']}"
    if not args.skip_log_only:
        summary += f", cov-only +{res['covariance_only_pct']}%, p_cov={res['mwu_p_cov_gt_log']}"
    print(summary + ")")
    if args.with_logix:
        lx = res["logix"]
        if not args.skip_logix_pca:
            print(f"[overhead-hf] LogIX-PCA (2-pass) total +{lx['logix_pca_total_attribution_pct']}% "
                  f"= logging +{lx['logix_pca_inline_logging_pct']}% + PCA cov pass "
                  f"+{lx['logix_pca_covariance_pass_pct_of_training']}%  "
                  f"[SANITY cov/fwd_bwd={lx['cov_pass_over_plain_fwd_bwd']}x]")
        print(f"[overhead-hf] LogIX-RANDOM (1-pass) overhead +{lx['logix_random_onepass_overhead_pct']}%  "
              f"vs Traceprop (1-pass) +{res['total_overhead_pct']}%  "
              f"[n={lx['logix_random_onepass_repeats']} vs n={lx['traceprop_repeats']}, "
              f"MWU p(LogIX-random > Traceprop)={lx['mwu_p_logix_random_gt_traceprop']}]")
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
