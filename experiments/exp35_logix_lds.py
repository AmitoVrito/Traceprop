"""exp35 -- LDS quality of LogIX's own compressed gradients (item 1, refinement).

exp31 measures LogIX's *speed* at its natural storage footprint. This
measures its *attribution quality* at that same footprint, using LogIX's own
official compute_influence_all() API (not a hand-extraction of its internal
tensors -- less implementation risk, and it's the API a real user would
call).

Storage matching: LogIX ALWAYS runs at its own natural settings (no rank
override -- an earlier version solved analytically for a rank meant to hit
Traceprop's proj_dim budget, which was wrong: LoraLinear clamps
rank=min(requested, in_features, out_features), so the analytical formula
used the requested rank, not the actual, possibly much smaller, clamped
one). Real bytes/example are measured directly from the on-disk log after
the real training-set logging pass, and Traceprop's own gradients are then
ALSO recomputed at that matched proj_dim (traceprop_dot_matched /
traceprop_trak_matched) alongside the default-proj_dim reference
(traceprop_dot / traceprop_trak), for a genuine storage-matched comparison
in both directions -- Traceprop is grown or shrunk to LogIX, never the
reverse.

Two LogIX scoring conditions:
  dot            compute_influence_all(mode="dot", precondition=False) --
                 directly analogous to our own dot_scores().
  preconditioned compute_influence_all(mode="dot", precondition=True,
                 hessian="raw") -- LogIX's Gauss-Newton-style correction,
                 analogous in spirit (not identical math) to our TRAK
                 estimator. Requires an extra full pass over the training set
                 to accumulate covariance statistics before the logging pass
                 -- Traceprop's TRAK correction needs no such extra pass, so
                 report this cost honestly if it shows up as a fairness point.
                 Best-effort: if covariance accumulation doesn't leave the
                 state LogIX expects (its own precondition() bails out with a
                 log warning rather than raising), this falls back to the
                 unconditioned dot score and the run is flagged as such.

Both LogIX conditions are scored against the SAME subset-retraining ground
truth (masks/margins) used for Traceprop's own dot/trak numbers elsewhere in
the repo (exp27/exp29's lds_for), so all four -- Traceprop-dot,
Traceprop-trak, LogIX-dot, LogIX-preconditioned -- are directly comparable
LDS numbers, not scores from different harnesses.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from exp27_lds_quality import (
    synthetic_data, load_sst2, build_tiny_classifier, build_hf_classifier,
)
from logix_strict import (
    install_strict_warnings, assert_pca_init_took_effect,
    patch_loralinear_weight_proxy, validate_logix_gradients,
)


def run(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.stats import spearmanr
    import logix

    install_strict_warnings()
    patch_loralinear_weight_proxy()

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
            m = build_tiny_classifier(vocab, seq=args.seq, r=args.rank, n_blocks=args.n_blocks)
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
    scope_patterns = ("lora_A", "lora_B") + HEAD
    last_n = None if args.track <= 0 else args.track

    def train_final(idx, epochs, lr, model=None):
        model = model if model is not None else new_model()
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

    def collect_grads_posthoc(model, X, y, patterns, last_n, proj_dim=None):
        proj_dim = proj_dim if proj_dim is not None else args.proj_dim
        store = GradientStore(proj_dim=proj_dim, seed=42)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=proj_dim)
        n = len(X)
        for s in range(0, n, args.batch):
            xb, yb = X[s:s + args.batch], y[s:s + args.batch]
            model.zero_grad(set_to_none=True)
            lo = logits(model, xb)
            F.cross_entropy(lo, yb, reduction="sum").backward()
            lg.flush_step(sample_indices=range(s, s + len(xb)))
        lg.detach()
        return store.get_projected_matrix()

    print(f"[exp35] training target model ({n_train} examples) ...")
    target_model = train_final(np.arange(n_train), args.epochs, args.lr)
    acc = float((logits(target_model, Xte_t).argmax(1) == yte_t).float().mean())
    print(f"[exp35] target test accuracy: {acc:.4f}")

    # ---- Traceprop's own post-hoc gradients at this same final checkpoint ----
    G_train = collect_grads_posthoc(target_model, Xtr_t, ytr_t, scope_patterns, last_n)
    G_test = collect_grads_posthoc(target_model, Xte_t, yte_t, scope_patterns, last_n)

    # ---- LogIX setup: same tracked-module scope as exp31 ----
    # LogIX operates on a SEPARATE deep copy of the trained model, not
    # target_model itself. add_lora() permanently rewrites the wrapped
    # modules' forward (result = _linear(x) + compression_path(x)), and
    # running Traceprop's OWN hook-based LoRAGradientLogger afterward on
    # that same wrapped model raised a RuntimeError from PyTorch's autograd
    # (view+inplace conflict between LogIX's backward hooks and Traceprop's)
    # -- confirmed by hitting it when the matched-proj_dim collect_grads_posthoc
    # call below first ran on the already-wrapped target_model. Two
    # independent, unwrapped-vs-wrapped copies of the SAME trained weights
    # avoids this entirely and keeps Traceprop's own collection (both the
    # default and the storage-matched proj_dim) untouched by LogIX either way.
    import copy
    logix_model = copy.deepcopy(target_model)

    if args.backend == "tiny":
        # respect --track properly (last N blocks, or all if track<=0) --
        # previously hardcoded to "score + last block only" regardless of
        # --track, which silently ignored the flag on this backend and made
        # a track={1,N,0} scope sweep impossible.
        tracked_names = [
            n for n, m in logix_model.named_modules()
            if isinstance(m, nn.Linear) and (n == "score" or "lora_A" in n or "lora_B" in n)
        ]
        if args.track > 0:
            import re as _re
            def _block_idx(name):
                m = _re.search(r"blocks\.(\d+)\.", name)
                return int(m.group(1)) if m else None
            idxs = sorted({_block_idx(n) for n in tracked_names if _block_idx(n) is not None})
            keep = set(idxs[-args.track:])
            tracked_names = [n for n in tracked_names if n == "score" or _block_idx(n) in keep]
    else:
        tracked_names = [
            n for n, m in logix_model.named_modules()
            if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n)
        ]
        if args.track > 0:
            import re
            def block_idx(name):
                m = re.search(r"(?:^|\.)(?:h|layers)\.(\d+)\.", name)
                return int(m.group(1)) if m else None
            idxs = sorted({block_idx(n) for n in tracked_names if block_idx(n) is not None})
            keep = set(idxs[-args.track:])
            tracked_names = [n for n in tracked_names if block_idx(n) in keep]

    n_tracked_layers = len(tracked_names)

    id_gen_counter = {"n": 0}

    def data_ids(batch_size):
        ids = [str(id_gen_counter["n"] + i) for i in range(batch_size)]
        id_gen_counter["n"] += batch_size
        return ids

    # LogIX runs at its own NATURAL settings -- no rank override. An earlier
    # version solved analytically for a rank meant to hit our proj_dim
    # budget; this was wrong the same way exp31's was (LoraLinear clamps
    # rank=min(requested, in_features, out_features), so the analytical
    # formula used the requested rank, not the actual, possibly much
    # smaller, clamped one). Real bytes/example are measured directly from
    # the on-disk log after the real training-set logging pass below
    # (measure_logix_bytes_per_example()), and Traceprop's own gradients are
    # then ALSO recomputed at that matched proj_dim for a genuine
    # apples-to-apples storage-matched LDS comparison.
    run_ = logix.LogIX(project=f"exp35_{os.getpid()}", config="exp31_config.yaml")
    run_.config.lora.init = args.lora_init
    run_.watch(logix_model, name_filter=tracked_names, type_filter=[nn.Linear])

    # --- Hessian/covariance accumulation pass (needed for precondition=True
    # scoring below, AND for add_lora()'s PCA init if --lora_init=pca --
    # LoRAHandler.add_lora() reads this SAME per-module covariance state, so
    # one pass over the training set serves both purposes). Covariance is
    # accumulated from forward activations / backward error signals
    # (KFAC-style), NOT from "grad" -- that's why this needs its own setup()
    # call, separate from the "grad": ["log"] mode used for the actual
    # per-example logging pass below.
    hessian_ok = True
    try:
        run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
        for s in range(0, n_train, args.batch):
            xb, yb = Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch]
            with run_(data_id=data_ids(len(xb))):
                logix_model.zero_grad(set_to_none=True)
                F.cross_entropy(logits(logix_model, xb), yb, reduction="sum").backward()
        run_.finalize()
    except Exception as e:
        print(f"[exp35] WARNING: Hessian/covariance accumulation pass failed ({e!r}); "
              f"preconditioned scores will fall back to unconditioned dot, and PCA "
              f"init (if requested) will fall back to random.")
        hessian_ok = False

    # CRITICAL (same bug as exp31, fixed here for the same reason): watch()
    # alone does NOT apply LogIX's compression -- is_lora(model) (which gates
    # the compressed logging path) only becomes true after add_lora() inserts
    # its own "logix_lora_*" wrapper. Without this call, LogIX logs raw,
    # uncompressed gradients of our own PEFT adapters, and the rank/storage
    # numbers above describe a knob that was never actually engaged.
    run_.add_lora()
    effective_init = args.lora_init if hessian_ok else "random"
    assert_pca_init_took_effect(run_, effective_init)

    # --- Logging pass over the TRAIN set: persist raw per-example grad to disk ---
    # save(True) is required -- build_log_dataloader() reads the log dataset
    # back from disk (LogDataset(log_dir=...)), so without it the loader is
    # silently empty and compute_influence_all() sees zero train batches.
    # NOTE: eval() internally calls save(False) -- it must NOT be called here,
    # or it undoes save(True) and the train pass silently logs nothing.
    run_.setup({"grad": ["log"]})

    # Gradient validation MUST run BEFORE save(True) below -- LogIX's default
    # _save state is False, so a check here stays in-memory only (get_log()
    # reads self.binfo.log regardless of save state). Running it after
    # save(True) instead persisted the check batch's entries into the SAME
    # on-disk log directory build_log_dataloader() reads later, silently
    # inflating n_train (caught: 40 became 44, off by exactly n_chk).
    gradient_validation = None
    if not getattr(args, "skip_gradient_validation", False):
        def per_example_loss_fn(xb, yb):
            raw = F.cross_entropy(logits(logix_model, xb), yb, reduction="none")
            return [raw[i] for i in range(raw.shape[0])]

        n_chk = min(n_train, args.batch, 4)
        # distinct id namespace (2e9+) so this check's log entries -- if this
        # were ever run with save(True) active -- could never collide with
        # train (0+) or test (1e9+) data_ids
        check_ids = [str(2 * 10 ** 9 + i) for i in range(n_chk)]
        gradient_validation = validate_logix_gradients(
            run_, logix_model, tracked_names, Xtr_t[:n_chk], ytr_t[:n_chk],
            per_example_loss_fn, data_id=check_ids,
        )
        print(f"[exp35] gradient validation OK: worst_cosine="
              f"{gradient_validation['worst_cosine']:.6f} over "
              f"{gradient_validation['n_checks']} checks, scale_ratio "
              f"mean/min/max={gradient_validation['scale_ratio_mean']:.3f}/"
              f"{gradient_validation['scale_ratio_min']:.3f}/"
              f"{gradient_validation['scale_ratio_max']:.3f}")

    run_.save(True)

    id_gen_counter["n"] = 0
    for s in range(0, n_train, args.batch):
        xb, yb = Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch]
        with run_(data_id=data_ids(len(xb))):
            logix_model.zero_grad(set_to_none=True)
            F.cross_entropy(logits(logix_model, xb), yb, reduction="sum").backward()

    # finalize() flushes the LogSaver's remaining buffer to the on-disk mmap
    # chunks LogDataset reads -- without this the log dataloader is silently
    # empty (build_log_dataset() just finds zero chunk files).
    run_.finalize()

    from logix_strict import measure_logix_bytes_per_example
    logix_bytes_per_example = measure_logix_bytes_per_example(run_, n_train)
    traceprop_proj_dim_to_match = max(1, round(logix_bytes_per_example / 4))
    print(f"[exp35] LogIX natural settings: measured {logix_bytes_per_example:.1f}B/example "
          f"on disk for {n_tracked_layers} tracked layers over the real {n_train}-example "
          f"train logging pass (real serialized size, not an analytical estimate). "
          f"Matching Traceprop proj_dim = {traceprop_proj_dim_to_match}.")

    log_loader = run_.build_log_dataloader(batch_size=args.batch, flatten=False)

    # --- TEST set: query points only, kept in-memory via get_log(), not
    # persisted to the train log database (eval() -> save(False)) ---
    run_.eval()
    id_gen_counter["n"] = 10 ** 9  # disjoint id space from train
    test_logs = []
    for s in range(0, n_test, args.batch):
        xb, yb = Xte_t[s:s + args.batch], yte_t[s:s + args.batch]
        with run_(data_id=data_ids(len(xb))):
            logix_model.zero_grad(set_to_none=True)
            F.cross_entropy(logits(logix_model, xb), yb, reduction="sum").backward()
        test_logs.append(run_.get_log(copy=True))

    def influence_matrix(precondition, hessian):
        """Assemble the full (n_test, n_train) score matrix by querying each
        test batch's log against the whole train log_loader and concatenating
        along the test axis -- compute_influence_all's src_log is a single
        batch's log, so we call it once per test batch."""
        rows = []
        for data_id, log in test_logs:
            res = run_.compute_influence_all(
                src_log=(data_id, log), loader=log_loader,
                mode="dot", precondition=precondition, hessian=hessian,
            )
            rows.append(res["influence"].numpy())
        return np.concatenate(rows, axis=0)  # (n_test, n_train)

    print("[exp35] scoring LogIX dot (precondition=False) ...")
    logix_dot = influence_matrix(precondition=False, hessian="raw")

    if hessian_ok:
        print("[exp35] scoring LogIX preconditioned (precondition=True, hessian=kfac) ...")
        try:
            logix_precond = influence_matrix(precondition=True, hessian="kfac")
            precond_note = "computed"
        except Exception as e:
            print(f"[exp35] WARNING: preconditioned scoring failed ({e!r}); using dot as fallback.")
            logix_precond = logix_dot
            precond_note = f"fallback_to_dot: {e!r}"
    else:
        logix_precond = logix_dot
        precond_note = "fallback_to_dot: hessian accumulation pass failed"

    # ---- Traceprop's own gradients at the storage-matched proj_dim, for a
    # genuine apples-to-apples comparison against LogIX's natural footprint
    # (not the reverse -- LogIX is never shrunk to match us) ----
    G_train_matched = collect_grads_posthoc(
        target_model, Xtr_t, ytr_t, scope_patterns, last_n, proj_dim=traceprop_proj_dim_to_match)
    G_test_matched = collect_grads_posthoc(
        target_model, Xte_t, yte_t, scope_patterns, last_n, proj_dim=traceprop_proj_dim_to_match)

    # ---- ground-truth LDS margins from subset retraining ----
    print(f"[exp35] retraining {args.n_subsets} subsets (frac={args.subset_frac}) ...")
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
        """attr: (n_test, n_train) score matrix."""
        pred = masks @ attr.T
        rs = [spearmanr(pred[:, i], margins[:, i]).correlation for i in range(n_test)]
        rs_arr = np.array(rs, dtype=np.float64)
        valid = ~np.isnan(rs_arr)
        return float(np.mean(rs_arr[valid])), float(np.std(rs_arr[valid])), rs_arr

    def dot_scores(gtr, gte):
        return gte @ gtr.T

    def trak_scores(gtr, gte, lam=None):
        d = gtr.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(gtr.T @ gtr) / d
        H = gtr.T @ gtr + lam * np.eye(d, dtype=np.float32)
        return gte @ np.linalg.solve(H, gtr.T)

    results, per_example_r = {}, {}
    for name, mat in (
        ("traceprop_dot", dot_scores(G_train, G_test)),
        ("traceprop_trak", trak_scores(G_train, G_test)),
        ("traceprop_dot_matched", dot_scores(G_train_matched, G_test_matched)),
        ("traceprop_trak_matched", trak_scores(G_train_matched, G_test_matched)),
        ("logix_dot", logix_dot),
        ("logix_preconditioned", logix_precond),
    ):
        mean, std, rs_arr = lds_for(mat)
        results[name] = (mean, std)
        per_example_r[name] = rs_arr
    rng2 = np.random.default_rng(0)
    mean, std, rs_arr = lds_for(rng2.standard_normal((n_test, n_train)).astype(np.float32))
    results["random"] = (mean, std)
    per_example_r["random"] = rs_arr

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "lora_init": args.lora_init,
        "lora_init_effective": effective_init,
        "n_train": n_train, "n_test": n_test, "n_subsets": args.n_subsets,
        "subset_frac": args.subset_frac, "epochs": args.epochs,
        "proj_dim": args.proj_dim, "track_last_n_blocks": args.track,
        "target_test_acc": round(acc, 4),
        "storage_matching": {
            "logix_bytes_per_example_measured": round(logix_bytes_per_example, 2),
            "traceprop_proj_dim_default": args.proj_dim,
            "traceprop_proj_dim_matched": traceprop_proj_dim_to_match,
            "note": "LogIX runs at its own natural settings (no rank override); "
                    "logix_bytes_per_example_measured is the REAL on-disk serialized size "
                    "from the actual train logging pass, not an analytical estimate. "
                    "traceprop_dot/traceprop_trak use proj_dim (the default reference, "
                    "512 floats/2KB elsewhere in the paper); "
                    "traceprop_dot_matched/traceprop_trak_matched use "
                    "traceprop_proj_dim_matched, grown or shrunk to LogIX's own measured "
                    "footprint for a genuine storage-matched comparison.",
        },
        "logix_preconditioned_note": precond_note,
        "gradient_validation": gradient_validation,
        "lds": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in results.items()},
    }
    print("\n=== LDS: Traceprop vs LogIX (own compute_influence_all API) ===")
    for k, v in out["lds"].items():
        print(f"  {k:<22} {v['mean']:+.4f} +/- {v['std']:.4f}")
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"{args.backend}_{out['model'].replace('/', '_')}_{args.lora_init}init"
    fn = getattr(args, "out", None) or f"results/exp35_{tag}.json"
    npz_fn = (fn[:-5] if fn.endswith(".json") else fn) + "_raw.npz" if getattr(args, "out", None) \
        else f"results/exp35_{tag}_raw.npz"
    if (os.path.exists(fn) or os.path.exists(npz_fn)) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {fn} or {npz_fn}. Pass --out <path> for a "
            f"different filename, or --force to overwrite."
        )
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")
    np.savez(npz_fn, masks=masks, margins=margins,
             **{f"r_{k}": v for k, v in per_example_r.items()})
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
    ap.add_argument("--n_blocks", type=int, default=2, help="tiny backend model depth")
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_init", choices=["random", "pca"], default="pca",
                    help="LogIX's LoRA init strategy for add_lora(). 'pca' is its authors' "
                         "recommended setting for LoRA (needs the covariance pass above, "
                         "timed via hessian_ok/covariance already in this script); 'random' "
                         "is LogIX's literal default (no extra pass needed). Run both to "
                         "report LDS for each, matching exp31's two overhead configs.")
    ap.add_argument("--out", default=None, help="output path override (default: auto)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    ap.add_argument("--skip_gradient_validation", action="store_true",
                    help="skip the one-time cosine-vs-autograd check of LogIX's logged "
                         "gradients (cheap, catches wiring bugs -- see logix_strict.py)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
