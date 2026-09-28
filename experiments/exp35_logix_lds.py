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
    from traceprop.llm import (
        LoRAGradientLogger, select_lora_linears, apply_kfac_precondition,
    )

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

    # Held-out split of the TEST examples used ONLY to pick the inline-precond
    # damping lambda (never the reported/eval set). Deterministic in args.seed so
    # the split is reproducible and can be re-applied downstream (saved to npz).
    _perm = np.random.default_rng(args.seed + 777).permutation(n_test)
    _n_val = max(5, int(round(getattr(args, "precond_val_frac", 0.3) * n_test)))
    precond_val_idx = np.sort(_perm[:_n_val])
    precond_eval_idx = np.sort(_perm[_n_val:])

    # Shared RELATIVE damping grid: used to tune BOTH our inline whitening and
    # LogIX's preconditioning, on the same val split -- the fairness requirement.
    damping_grid = [float(x) for x in
                    getattr(args, "precond_damping_grid",
                            "1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1").split(",")]

    # Target-model init seed is 1234+args.seed so different --seed values give
    # genuinely different target models (not just different data/subsets). seed=0
    # -> 1234, reproducing the pre-fairness-check run exactly. Within one run all
    # subset-retrained models share this init (standard LDS practice).
    _model_seed = 1234 + args.seed

    def new_model(seed=None):
        s = _model_seed if seed is None else seed
        torch.manual_seed(s)
        np.random.seed(s)
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

    def train_final(idx, epochs, lr, model=None, train_seed=None):
        # train_seed varies BOTH the model init and the batch order -- used only by
        # the noise-ceiling check (retrain the SAME subset twice differently). The
        # main LDS retraining leaves it None so every subset shares the same init
        # (only the data differs), as LDS requires.
        model = model if model is not None else new_model(train_seed)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        idx = np.asarray(idx)
        # ONE rng created before the loop so each epoch gets a DIFFERENT (but fully
        # deterministic, hence LDS-reproducible) batch order. Re-seeding rng(0) inside
        # the loop reused the same order every epoch and generalised noticeably worse.
        perm_rng = np.random.default_rng(0 if train_seed is None else train_seed)
        for _ in range(epochs):
            perm = perm_rng.permutation(len(idx))
            for s in range(0, len(idx), args.batch):
                b = idx[perm[s:s + args.batch]]
                xb, yb = Xtr_t[b], ytr_t[b]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(logits(model, xb), yb).backward()
                opt.step()
        return model

    # collect_grads_posthoc stashes the wall-clock of the logging pass and, when
    # inline_precond=True, the inline-accumulated K-FAC covariance here, so the
    # caller can read them without changing the (matrix) return type.
    _collect_info = {"elapsed": None, "cov": None}

    def collect_grads_posthoc(model, X, y, patterns, last_n, proj_dim=None,
                               factored=False, kfac=None, inline_precond=False):
        import time
        proj_dim = proj_dim if proj_dim is not None else args.proj_dim
        store = GradientStore(proj_dim=proj_dim, seed=42)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=proj_dim,
                                 factored=factored, kfac=kfac or 16,
                                 inline_precond=inline_precond)
        n = len(X)
        t0 = time.perf_counter()
        for s in range(0, n, args.batch):
            xb, yb = X[s:s + args.batch], y[s:s + args.batch]
            model.zero_grad(set_to_none=True)
            lo = logits(model, xb)
            F.cross_entropy(lo, yb, reduction="sum").backward()
            lg.flush_step(sample_indices=range(s, s + len(xb)))
        _collect_info["elapsed"] = time.perf_counter() - t0
        _collect_info["cov"] = lg.kfac_covariances() if inline_precond else None
        lg.detach()
        return store.get_projected_matrix()

    def validate_factored_gradients(model, patterns, last_n, kfac, X, y, n_checks=4, seed=42):
        """Cosine-vs-autograd check for the factored (Kronecker) sketch, the
        same rigor standard the dense path is held to elsewhere in this
        script (validate_logix_gradients / gradient_validation above), which
        the factored path has never had until now.

        For a single example, _factored_sketch()'s hook-based computation is
        Sum_t (P g_t) ⊗ (Q a_t) using the activation/output-grad the SAME
        hooks capture for the dense path (already implicitly trusted).
        Sum_t (P g_t) ⊗ (Q a_t) = P (Sum_t g_t ⊗ a_t) Q^T algebraically, so an
        INDEPENDENT ground truth built from torch.autograd.grad (a fresh
        forward/backward, not reusing the hook-captured tensors at all) run
        through that same P, G_true, Q^T formula must match the logger's own
        sketch to numerical precision if -- and only if -- both the hook
        capture and the sketch algebra are correct. This does not test
        whether the sketch is a good LOW-DIMENSIONAL approximation (that's
        what LDS separately measures) -- only that the sketch the logger
        produces is the sketch the math says it should produce."""
        import torch
        model = model.eval()
        store = GradientStore(proj_dim=64, seed=seed)
        targets = select_lora_linears(model, patterns, last_n_blocks=last_n)
        lg = LoRAGradientLogger(store, targets, proj_dim=64, factored=True,
                                 kfac=kfac, seed=seed)
        cosines = []
        n_checks = min(n_checks, len(X))
        for i in range(n_checks):
            xb, yb = X[i:i + 1], y[i:i + 1]

            model.zero_grad(set_to_none=True)
            lo = logits(model, xb)
            F.cross_entropy(lo, yb, reduction="sum").backward()
            sketch_hook = lg._factored_sketch()  # populates lg._fac_PQ on first call
            if sketch_hook is None:
                continue
            sketch_hook = sketch_hook[0].detach().cpu().numpy()

            manual_parts = []
            for name, module in targets:
                # Only modules that actually fired a hook are in the sketch (and in
                # _fac_PQ). PEFT wraps modules_to_save (e.g. the classifier head) in
                # a frozen `original_module` copy that never runs in forward and has
                # requires_grad=False -- it is absent from the hook sketch, so skip
                # it here too (otherwise autograd.grad raises "does not require grad"
                # and the manual concatenation would misalign with the hook sketch).
                if name not in lg._fac_PQ or not module.weight.requires_grad:
                    continue
                model.zero_grad(set_to_none=True)
                lo2 = logits(model, xb)
                loss2 = F.cross_entropy(lo2, yb, reduction="sum")
                (g_true,) = torch.autograd.grad(loss2, module.weight, retain_graph=False)
                P, Q = lg._fac_PQ[name]
                # all in float32 -- the hook sketch upcasts too, and g_true/P/Q may
                # differ in dtype (e.g. an fp16 HF model gives a Half g_true).
                manual = (P.float() @ g_true.float() @ Q.float().T)
                manual_parts.append(manual.reshape(-1).detach().cpu().numpy())
            sketch_manual = np.concatenate(manual_parts)

            num = float(np.dot(sketch_hook, sketch_manual))
            den = float(np.linalg.norm(sketch_hook) * np.linalg.norm(sketch_manual))
            cosines.append(num / den if den > 0 else float("nan"))
        lg.detach()
        model.train()
        cosines = np.array(cosines, dtype=np.float64)
        return {
            "n_checks": int(len(cosines)),
            "worst_cosine": float(np.nanmin(cosines)) if len(cosines) else None,
            "mean_cosine": float(np.nanmean(cosines)) if len(cosines) else None,
        }

    print(f"[exp35] training target model ({n_train} examples) ...")
    target_model = train_final(np.arange(n_train), args.epochs, args.lr)
    acc = float((logits(target_model, Xte_t).argmax(1) == yte_t).float().mean())
    print(f"[exp35] target test accuracy: {acc:.4f}")

    # Fast sanity gate: a model at (near-)chance has no learnable signal, so its
    # subset margins are noise and any LDS number is meaningless. Abort BEFORE the
    # expensive retraining loop instead of burning it on a dead model.
    _min_acc = getattr(args, "min_target_acc", 0.0)
    if _min_acc > 0.0 and acc < _min_acc:
        with torch.no_grad():
            pred = logits(target_model, Xte_t).argmax(1)
            dist = torch.bincount(pred, minlength=int(yte_t.max()) + 1).tolist()
        raise SystemExit(
            f"[exp35] ABORT: target test acc {acc:.4f} < --min_target_acc {_min_acc} "
            f"(pred class distribution {dist}). The model is not learning the task -- "
            f"fix the SETUP (data/epochs/lr/head) before any LDS run. See diag_hf_classifier.py."
        )

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

    # --- SECOND covariance pass, AFTER add_lora(), on the WRAPPED module
    # names. The pass above (pre-add_lora) covers PCA init, which needs
    # covariance keyed by the ORIGINAL module names (LoRAHandler.add_lora()
    # reads it that way). But compute_influence_all's precondition() step at
    # scoring time looks up covariance by whatever module names are IN THE
    # LOGGED DATA -- which, after add_lora(), are the WRAPPED
    # "...logix_lora_B" names, not the originals. Reusing the pre-add_lora
    # covariance state for that lookup silently fails: LogIX's own
    # precondition() finds a key mismatch, logs "Not all covariances have
    # been computed" as a WARNING (not an exception), and returns src_log
    # UNCHANGED -- confirmed empirically by comparing logix_dot and
    # logix_preconditioned score arrays byte-for-byte after a real run: they
    # were IDENTICAL, meaning "preconditioning" had silently been a no-op the
    # entire time despite hessian_ok being True and precond_note reading
    # "computed". Re-accumulate covariance here, post-wrap, so the keys the
    # later lookup needs actually exist.
    precondition_hessian_ok = True
    try:
        run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
        for s in range(0, n_train, args.batch):
            xb, yb = Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch]
            with run_(data_id=data_ids(len(xb))):
                logix_model.zero_grad(set_to_none=True)
                F.cross_entropy(logits(logix_model, xb), yb, reduction="sum").backward()
        run_.finalize()
    except Exception as e:
        print(f"[exp35] WARNING: post-add_lora covariance pass failed ({e!r}); "
              f"preconditioned scores will fall back to unconditioned dot.")
        precondition_hessian_ok = False

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

    def influence_matrix(precondition, hessian, damping=None):
        """Assemble the full (n_test, n_train) score matrix by querying each
        test batch's log against the whole train log_loader and concatenating
        along the test axis -- compute_influence_all's src_log is a single
        batch's log, so we call it once per test batch. ``damping`` is passed
        straight to LogIX (absolute, added to the K-FAC eigenvalues); None uses
        LogIX's own default (0.1*mean eigval)."""
        rows = []
        for data_id, log in test_logs:
            res = run_.compute_influence_all(
                src_log=(data_id, log), loader=log_loader,
                mode="dot", precondition=precondition, hessian=hessian,
                damping=damping,
            )
            rows.append(res["influence"].numpy())
        return np.concatenate(rows, axis=0)  # (n_test, n_train)

    print("[exp35] scoring LogIX dot (precondition=False) ...")
    logix_dot = influence_matrix(precondition=False, hessian="raw")

    if precondition_hessian_ok:
        print("[exp35] scoring LogIX preconditioned (precondition=True, hessian=kfac) ...")
        try:
            logix_precond = influence_matrix(precondition=True, hessian="kfac")
            # LogIX's own precondition() can fail SILENTLY -- it catches its own
            # key-mismatch case internally, logs a WARNING (not an exception),
            # and returns the src_log UNCHANGED, so compute_influence_all runs
            # to completion and returns scores identical to precondition=False.
            # Our own try/except only catches actual exceptions, so it reported
            # "computed" for exactly this silent-fallback case before this check
            # was added -- confirmed by finding logix_dot and logix_precond
            # byte-identical in a real run despite precond_note saying
            # "computed". Detect it directly instead of trusting the absence
            # of an exception.
            if np.array_equal(logix_precond, logix_dot):
                precond_note = ("fallback_to_dot: LogIX's own precondition() silently "
                                 "returned unconditioned scores (covariance key mismatch or "
                                 "similar internal bail-out) -- scores are byte-identical to "
                                 "precondition=False, not a coincidence")
            else:
                precond_note = "computed"
        except Exception as e:
            print(f"[exp35] WARNING: preconditioned scoring failed ({e!r}); using dot as fallback.")
            logix_precond = logix_dot
            precond_note = f"fallback_to_dot: {e!r}"
    else:
        logix_precond = logix_dot
        precond_note = "fallback_to_dot: post-add_lora covariance accumulation pass failed"

    # ---- Probe: dump LogIX's covariance/eigval state shapes to verify what space
    # LogIX actually preconditions in (full d x d, or its own add_lora-projected
    # rank-r space), and test whether repeated preconditioned calls mutate state
    # (the `full_eigval += damping` aliasing concern). Exits before retraining. ----
    if getattr(args, "dump_cov_shapes", False):
        ev_state, evec_state = run_.state.get_covariance_svd_state()
        info = {"n_tracked_layers": n_tracked_layers, "lora_rank": args.rank,
                "lora_init": effective_init, "modules": {}}
        for i, (mod, ev) in enumerate(ev_state.items()):
            vec = evec_state.get(mod, {})
            entry = {}
            if isinstance(ev, dict):
                entry["eigval_kind"] = "dict(forward,backward)"
                entry["fwd_eigval_shape"] = list(ev["forward"].shape)
                entry["bwd_eigval_shape"] = list(ev["backward"].shape)
            else:
                entry["eigval_kind"] = "tensor"
                entry["eigval_shape"] = list(ev.shape)
            if isinstance(vec, dict):
                if "forward" in vec:
                    entry["fwd_eigvec_shape"] = list(vec["forward"].shape)
                if "backward" in vec:
                    entry["bwd_eigvec_shape"] = list(vec["backward"].shape)
            info["modules"][mod] = entry
            if i >= 4:
                break
        # mutation test: identical damping twice -> identical scores iff no state mutation
        m1 = influence_matrix(precondition=True, hessian="kfac", damping=1e-7)
        m2 = influence_matrix(precondition=True, hessian="kfac", damping=1e-7)
        info["repeat_same_damping_identical"] = bool(np.allclose(m1, m2))
        info["state_mutated_across_calls"] = not info["repeat_same_damping_identical"]
        # default-bug test: does damping=None leak module-0's damping to all modules?
        # (compare default None-run against an explicit per-module-uniform absolute)
        info["default_run_note"] = ("logix_preconditioned above used damping=None; LogIX's "
                                    "precondition_kfac sets damping inside the module loop, so "
                                    "modules after the first reuse module-0's absolute damping.")
        os.makedirs("results", exist_ok=True)
        with open("results/exp35_cov_shapes.json", "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info, indent=2))
        raise SystemExit(0)

    # ---- Check 1 (fairness): give LogIX the SAME damping-tuning opportunity we
    # give ourselves. LogIX's damping is ABSOLUTE (added to K-FAC eigenvalues);
    # its default (None) = 0.1 * mean(eigval) = relative 0.1. To sweep it on the
    # SAME relative grid as our inline whitening, convert each lambda_rel to
    # absolute via LogIX's own global mean K-FAC eigenvalue. Score the whole grid
    # here; selection on the val split happens after retraining (with margins).
    logix_precond_grid = {}       # lambda_rel -> (n_test, n_train) score matrix
    logix_tuned_info = {}
    if getattr(args, "tune_logix", False) and precondition_hessian_ok \
            and precond_note == "computed":
        ev_state, _ = run_.state.get_covariance_svd_state()
        eig_means = []
        for _mod, ev in ev_state.items():
            if isinstance(ev, dict):
                full = torch.outer(ev["backward"].float().flatten(),
                                    ev["forward"].float().flatten())
            else:
                full = ev.float()
            eig_means.append(float(full.mean()))
        global_mean_eig = float(np.mean(eig_means)) if eig_means else None

        # Check 2 (item): give LogIX PER-MODULE relative damping, matching what our
        # inline whitening does (lambda_rel * that module's mean eigval, per module),
        # instead of a single global absolute damping. Monkeypatch precondition_kfac
        # so the passed `damping` is interpreted as lambda_rel and scaled per module.
        # Never uses LogIX's damping=None path (which leaks module-0's damping).
        permodule = getattr(args, "logix_permodule_damping", False)
        default_bug = {}
        if permodule:
            import logix.analysis.influence_function as _lif
            from logix_strict import precondition_kfac_permodule
            _lif.precondition_kfac = precondition_kfac_permodule

            # Default-bug probe: does LogIX's damping=None run (module-0 leak, the
            # UNPATCHED logix_precond above) differ from a correct per-module 0.1?
            correct01 = influence_matrix(precondition=True, hessian="kfac", damping=0.1)
            default_bug = {
                "logix_default_differs_from_correct_permodule_0.1":
                    bool(not np.allclose(logix_precond, correct01)),
                "max_abs_diff": float(np.abs(logix_precond - correct01).max()),
                "interpretation": "if True, LogIX's damping=None default leaks module-0's "
                                  "absolute damping to all later modules (a real LogIX bug).",
            }

        if global_mean_eig and global_mean_eig > 0:
            logix_tuned_info = {
                "global_mean_kfac_eigval": global_mean_eig,
                "logix_default_damping_rule": "0.1 * mean(kfac_eigval) per module (relative 0.1)",
                "logix_default_damping_abs_nominal": 0.1 * global_mean_eig,
                "damping_granularity": ("per_module_relative" if permodule
                                        else "global_absolute"),
                "kfac_space": "logix add_lora rank-%d projected (eigval per factor ~rank-sized), "
                              "NOT full d x d -- confirmed via cov shape probe" % args.rank,
                "default_damping_bug": default_bug,
                "note": ("LogIX preconditions in its own add_lora-projected rank space. "
                         "per_module_relative: damping = lambda_rel * that module's mean eigval "
                         "(matches Traceprop). global_absolute: lambda_rel * global mean eigval."),
            }
            for lam_rel in damping_grid:
                d_pass = lam_rel if permodule else lam_rel * global_mean_eig
                mat = influence_matrix(precondition=True, hessian="kfac", damping=d_pass)
                logix_precond_grid[lam_rel] = mat
            print(f"[exp35] LogIX damping sweep ({'per-module' if permodule else 'global'} "
                  f"relative): global mean K-FAC eigval={global_mean_eig:.3e}; scored "
                  f"{len(logix_precond_grid)} grid points. default_bug={default_bug.get('logix_default_differs_from_correct_permodule_0.1')}")
        else:
            print("[exp35] WARNING: could not extract LogIX mean eigenvalue; skipping LogIX tuning.")

    # ---- Traceprop's own gradients at the storage-matched proj_dim, for a
    # genuine apples-to-apples comparison against LogIX's natural footprint
    # (not the reverse -- LogIX is never shrunk to match us) ----
    G_train_matched = collect_grads_posthoc(
        target_model, Xtr_t, ytr_t, scope_patterns, last_n, proj_dim=traceprop_proj_dim_to_match)
    G_test_matched = collect_grads_posthoc(
        target_model, Xte_t, yte_t, scope_patterns, last_n, proj_dim=traceprop_proj_dim_to_match)

    # ---- Traceprop's FACTORED (Kronecker) sketch, storage-BRACKETED via kfac
    # instead of proj_dim -- exp25/exp31 already confirmed this path beats
    # LogIX on SPEED at every tracked scope, but no LDS number for it has
    # ever been measured; the dense path above is the only one validated for
    # attribution quality so far. kfac must be an integer, so an exact byte
    # match only happens when logix_bytes/4/n_tracked_layers is a perfect
    # square (true for pythia-1b's uniform-rank scopes, NOT guaranteed here).
    # Rather than pick one side, run BOTH bracketing candidates: floor(ideal)
    # uses LESS storage than LogIX (the conservative direction -- Traceprop
    # under a smaller budget still matching LogIX would be the stronger
    # claim), ceil(ideal) uses MORE (favors Traceprop, weaker claim if quality
    # only holds there). Each gets both dot and TRAK scoring, matching the
    # dense path's parity.
    factored_variants = {}  # label -> {"G_train":..., "G_test":..., "bytes":..., "rel_error":...}
    factored_gradient_validation = {}
    if getattr(args, "factored", False):
        # bytes_per_element=4 (fp32, the actual stored dtype -- what's really
        # on disk today) and, if --fp16, ALSO 2 (fp16 storage would allow a
        # larger kfac in the same byte budget; the resulting sketch is
        # round-tripped through np.float16 below to simulate the real
        # precision loss, not just given a bigger kfac for free).
        bpe_variants = [(4, "")] + ([(2, "_fp16")] if getattr(args, "fp16", False) else [])
        for bytes_per_element, suffix in bpe_variants:
            ideal_kfac = (logix_bytes_per_example / bytes_per_element / n_tracked_layers) ** 0.5
            kfac_candidates = sorted({max(1, int(np.floor(ideal_kfac))),
                                       max(1, int(np.ceil(ideal_kfac)))})
            for kfac in kfac_candidates:
                label = f"{kfac}{suffix}"
                bytes_achieved = n_tracked_layers * kfac ** 2 * bytes_per_element
                rel_error = (bytes_achieved - logix_bytes_per_example) / logix_bytes_per_example
                direction = "LESS" if rel_error < 0 else "MORE"
                dtype_note = "fp16" if suffix else "fp32"
                print(f"[exp35] factored path ({dtype_note}): kfac={kfac} -> "
                      f"{bytes_achieved}B/example vs LogIX's {logix_bytes_per_example:.1f}"
                      f"B/example measured over {n_tracked_layers} tracked layers "
                      f"({rel_error:+.1%}, {direction} storage than LogIX).")

                print(f"[exp35] validating factored sketch kfac={kfac} ({dtype_note}) (cosine "
                      f"vs. independent autograd) ...")
                fgv = validate_factored_gradients(
                    target_model, scope_patterns, last_n, kfac, Xtr_t, ytr_t)
                factored_gradient_validation[label] = fgv
                print(f"[exp35] factored gradient validation kfac={kfac} ({dtype_note}): "
                      f"worst_cosine={fgv['worst_cosine']:.6f} over {fgv['n_checks']} checks")

                G_train_f = collect_grads_posthoc(
                    target_model, Xtr_t, ytr_t, scope_patterns, last_n, factored=True, kfac=kfac)
                t_base = _collect_info["elapsed"]
                G_test_f = collect_grads_posthoc(
                    target_model, Xte_t, yte_t, scope_patterns, last_n, factored=True, kfac=kfac)

                # Inline K-FAC: re-log the TRAIN pass with covariance accumulation
                # on (same sketches, plus the k x k covariance of the projected
                # factors), and measure the extra wall-clock vs the plain factored
                # pass above -- the honest cost of "no second covariance pass".
                cov_bundle = None
                inline_overhead_pct = None
                if getattr(args, "inline_precond", False):
                    G_train_f = collect_grads_posthoc(
                        target_model, Xtr_t, ytr_t, scope_patterns, last_n,
                        factored=True, kfac=kfac, inline_precond=True)
                    t_cov = _collect_info["elapsed"]
                    cov_bundle = _collect_info["cov"]
                    inline_overhead_pct = (
                        (t_cov - t_base) / t_base * 100.0 if t_base else None)
                    print(f"[exp35] inline-precond covariance accumulation overhead "
                          f"(kfac={kfac}, {dtype_note}): {t_base*1e3:.1f}ms plain vs "
                          f"{t_cov*1e3:.1f}ms with covariance "
                          f"({inline_overhead_pct:+.2f}% of the logging pass) -- no "
                          f"second pass over the data.")

                if suffix:  # simulate fp16 storage precision loss, not just a free bigger kfac
                    G_train_f = G_train_f.astype(np.float16).astype(np.float32)
                    G_test_f = G_test_f.astype(np.float16).astype(np.float32)
                factored_variants[label] = {
                    "G_train": G_train_f, "G_test": G_test_f,
                    "bytes_achieved": bytes_achieved, "rel_error": rel_error,
                    "dtype": dtype_note, "kfac": kfac,
                    "cov_bundle": cov_bundle,
                    "inline_overhead_pct": inline_overhead_pct,
                }

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

    # ---- noise ceiling: retrain N subsets TWICE with different seeds; the mean
    # per-example Spearman between the two margin vectors (across those subsets)
    # is the maximum LDS any attribution method could reach given retraining noise.
    # A pilot LDS well below this ceiling but well above 0 is real signal. ----
    noise_ceiling = None
    n_nc = getattr(args, "noise_ceiling", 0)
    if n_nc > 0:
        print(f"[exp35] noise ceiling: {n_nc} subsets x2 retrains (different seeds) ...")
        nc_rng = np.random.default_rng(args.seed + 4242)
        ma = np.zeros((n_nc, n_test), dtype=np.float32)
        mb = np.zeros((n_nc, n_test), dtype=np.float32)
        for m in range(n_nc):
            sub = nc_rng.choice(n_train, size=k, replace=False)
            ma[m] = test_margins(train_final(sub, args.epochs, args.lr, train_seed=1000 + m))
            mb[m] = test_margins(train_final(sub, args.epochs, args.lr, train_seed=5000 + m))
        rs = [spearmanr(ma[:, i], mb[:, i]).correlation for i in range(n_test)]
        rs = np.array(rs, dtype=np.float64); rs = rs[~np.isnan(rs)]
        noise_ceiling = {"mean": round(float(np.mean(rs)), 4),
                         "std": round(float(np.std(rs)), 4), "n_subsets": int(n_nc)}
        print(f"[exp35] noise ceiling (max achievable LDS): "
              f"{noise_ceiling['mean']:+.4f} +/- {noise_ceiling['std']:.4f}")

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

    score_list = [
        ("traceprop_dot", dot_scores(G_train, G_test)),
        ("traceprop_trak", trak_scores(G_train, G_test)),
        ("traceprop_dot_matched", dot_scores(G_train_matched, G_test_matched)),
        ("traceprop_trak_matched", trak_scores(G_train_matched, G_test_matched)),
        ("logix_dot", logix_dot),
        ("logix_preconditioned", logix_precond),
    ]
    def lds_on_subset(attr, idx):
        """Mean Spearman LDS over a SUBSET of test examples (for held-out
        damping selection -- never touches the reported eval examples)."""
        pred = masks @ attr.T  # (n_subsets, n_test)
        rs = [spearmanr(pred[:, i], margins[:, i]).correlation for i in idx]
        rs = np.array(rs, dtype=np.float64)
        return float(np.nanmean(rs[~np.isnan(rs)]))

    # ---- Check 1: select LogIX's best damping on the SAME val split, report as
    # logix_preconditioned_tuned (tuned-vs-tuned fairness). ----
    if logix_precond_grid:
        best_lg = None  # (lambda_rel, val_lds, mat)
        lg_grid_val = {}
        for lam_rel, mat in logix_precond_grid.items():
            v_lds = lds_on_subset(mat, precond_val_idx)
            lg_grid_val[f"{lam_rel:g}"] = round(v_lds, 4)
            if best_lg is None or v_lds > best_lg[1]:
                best_lg = (lam_rel, v_lds, mat)
        score_list.append(("logix_preconditioned_tuned", best_lg[2]))
        logix_tuned_info.update({
            "chosen_lambda_rel": best_lg[0],
            "chosen_damping_abs": best_lg[0] * logix_tuned_info["global_mean_kfac_eigval"],
            "chosen_val_lds": round(best_lg[1], 4),
            "grid_val_lds": lg_grid_val,
        })
        print(f"[exp35] LogIX tuned: chose lambda_rel={best_lg[0]:g} "
              f"(val LDS={best_lg[1]:+.4f}); grid={lg_grid_val}")

    inline_precond_selection = {}  # label -> {chosen_damping, grid_val_lds, ...}

    for label, v in factored_variants.items():
        score_list.append((f"traceprop_factored_kfac{label}_dot",
                            dot_scores(v["G_train"], v["G_test"])))
        score_list.append((f"traceprop_factored_kfac{label}_trak",
                            trak_scores(v["G_train"], v["G_test"])))

        # ---- inline K-FAC preconditioning: whiten both train & test factored
        # sketches with the TRAIN covariance accumulated inline, pick damping on
        # the held-out VAL split, score with the chosen damping. Same-pass
        # covariance = no second pass, unlike LogIX's separate covariance sweep.
        if v.get("cov_bundle") is not None:
            layout, cov_G, cov_A = v["cov_bundle"]
            grid_val_lds = {}
            best = None  # (damping, val_lds, attr)
            for damp in damping_grid:
                Wtr = apply_kfac_precondition(v["G_train"], layout, cov_G, cov_A, damp)
                Wte = apply_kfac_precondition(v["G_test"], layout, cov_G, cov_A, damp)
                attr = dot_scores(Wtr, Wte)
                val_lds = lds_on_subset(attr, precond_val_idx)
                grid_val_lds[f"{damp:g}"] = round(val_lds, 4)
                if best is None or val_lds > best[1]:
                    best = (damp, val_lds, attr)
            chosen_damp, chosen_val_lds, best_attr = best
            score_list.append(
                (f"traceprop_factored_kfac{label}_inlineprecond", best_attr))
            inline_precond_selection[label] = {
                "chosen_damping": chosen_damp,
                "chosen_val_lds": round(chosen_val_lds, 4),
                "grid_val_lds": grid_val_lds,
                "n_val": int(len(precond_val_idx)),
                "n_eval": int(len(precond_eval_idx)),
                "inline_overhead_pct": (round(v["inline_overhead_pct"], 3)
                                        if v["inline_overhead_pct"] is not None else None),
            }
            print(f"[exp35] inline-precond kfac={label}: chose damping={chosen_damp:g} "
                  f"(val LDS={chosen_val_lds:+.4f} over {len(precond_val_idx)} held-out "
                  f"val examples); grid={grid_val_lds}")

    results, per_example_r, score_matrices = {}, {}, {}
    for name, mat in score_list:
        mean, std, rs_arr = lds_for(mat)
        results[name] = (mean, std)
        per_example_r[name] = rs_arr
        score_matrices[name] = mat
    rng2 = np.random.default_rng(0)
    mean, std, rs_arr = lds_for(rng2.standard_normal((n_test, n_train)).astype(np.float32))
    results["random"] = (mean, std)
    per_example_r["random"] = rs_arr

    # Held-out eval-only LDS: mean over the eval split ONLY (disjoint from the
    # val examples used to pick the inline-precond damping), so the reported
    # inline-precond number has zero tuning leakage. Computed from the already-
    # stored per-example Spearman r, no rescoring.
    def _heldout(rs_arr):
        sub = rs_arr[precond_eval_idx]
        sub = sub[~np.isnan(sub)]
        return {"mean": round(float(np.mean(sub)), 4),
                "std": round(float(np.std(sub)), 4)}
    lds_heldout_eval = {name: _heldout(per_example_r[name]) for name in per_example_r}

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "lora_init": args.lora_init,
        "lora_init_effective": effective_init,
        "n_train": n_train, "n_test": n_test, "n_subsets": args.n_subsets,
        "subset_frac": args.subset_frac, "epochs": args.epochs,
        "noise_ceiling": noise_ceiling,
        "proj_dim": args.proj_dim, "track_last_n_blocks": args.track,
        "target_test_acc": round(acc, 4),
        "storage_matching": {
            "logix_bytes_per_example_measured": round(logix_bytes_per_example, 2),
            "traceprop_proj_dim_default": args.proj_dim,
            "traceprop_proj_dim_matched": traceprop_proj_dim_to_match,
            "traceprop_factored_kfac_bracket": {
                label: {
                    "kfac": v["kfac"],
                    "dtype": v["dtype"],
                    "bytes_achieved": v["bytes_achieved"],
                    "rel_error_vs_logix": round(v["rel_error"], 4),
                    "direction": "less_storage_than_logix" if v["rel_error"] < 0
                                 else "more_storage_than_logix",
                }
                for label, v in factored_variants.items()
            },
            "note": "LogIX runs at its own natural settings (no rank override); "
                    "logix_bytes_per_example_measured is the REAL on-disk serialized size "
                    "from the actual train logging pass, not an analytical estimate. "
                    "traceprop_dot/traceprop_trak use proj_dim (the default reference, "
                    "512 floats/2KB elsewhere in the paper); "
                    "traceprop_dot_matched/traceprop_trak_matched use "
                    "traceprop_proj_dim_matched, grown or shrunk to LogIX's own measured "
                    "footprint for a genuine storage-matched comparison. traceprop_factored_* "
                    "(if --factored was passed) brackets BOTH achievable integer kfac values "
                    "around LogIX's measured footprint -- an exact byte match only happens "
                    "when bytes/4/n_tracked_layers is a perfect square, which is not "
                    "guaranteed on this backend; see traceprop_factored_kfac_bracket for the "
                    "signed relative error of each. Neither direction is inherently "
                    "'conservative' for Traceprop -- less storage is the stronger claim if "
                    "quality still holds, more storage is the weaker one.",
        },
        "logix_preconditioned_note": precond_note,
        "gradient_validation": gradient_validation,
        "factored_gradient_validation": factored_gradient_validation,
        "inline_precond_selection": inline_precond_selection,
        "logix_tuned_selection": logix_tuned_info,
        "precond_val_eval_split": {
            "val_idx": [int(i) for i in precond_val_idx],
            "eval_idx": [int(i) for i in precond_eval_idx],
            "note": "inline-precond damping is chosen on val_idx only; lds_heldout_eval "
                    "is reported on eval_idx (disjoint) so there is no tuning leakage. "
                    "Downstream two-way bootstrap must restrict to eval_idx.",
        },
        "lds": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)} for k, v in results.items()},
        "lds_heldout_eval": lds_heldout_eval,
    }
    print("\n=== LDS: Traceprop vs LogIX (own compute_influence_all API) ===")
    if noise_ceiling is not None:
        print(f"    NOISE CEILING (max achievable LDS): {noise_ceiling['mean']:+.4f} "
              f"+/- {noise_ceiling['std']:.4f}  ({noise_ceiling['n_subsets']} subsets x2)")
    print(f"    [full n_test={n_test}]      [held-out eval n={len(precond_eval_idx)}]")
    for k, v in out["lds"].items():
        he = out["lds_heldout_eval"].get(k, {})
        print(f"  {k:<38} {v['mean']:+.4f} +/- {v['std']:.4f}   "
              f"{he.get('mean', float('nan')):+.4f} +/- {he.get('std', float('nan')):.4f}")
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
             precond_val_idx=precond_val_idx, precond_eval_idx=precond_eval_idx,
             **{f"r_{k}": v for k, v in per_example_r.items()},
             **{f"attr_{k}": v for k, v in score_matrices.items()})
    print(f"saved -> {npz_fn} (per-example scores + raw (n_test, n_train) attribution "
          f"matrices -- enables a two-way bootstrap over both test examples AND subsets, "
          f"not just the already-reduced per-example Spearman r)")
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
    ap.add_argument("--factored", action="store_true",
                    help="also score Traceprop's Kronecker-factored sketch, bracketed at the "
                         "two achievable integer kfac values around LogIX's measured "
                         "bytes/example, validated via cosine-vs-independent-autograd")
    ap.add_argument("--fp16", action="store_true",
                    help="with --factored, ALSO run a fp16-storage bracket (2 bytes/element "
                         "instead of 4, allowing a larger kfac at the same byte budget); the "
                         "sketch is round-tripped through np.float16 to simulate real storage "
                         "precision loss, not just given a bigger kfac for free")
    ap.add_argument("--inline_precond", action="store_true",
                    help="with --factored, accumulate K-FAC covariance of the projected "
                         "factors INLINE during the (same) logging pass and apply damped "
                         "whitening at query time -- the 'no second covariance pass' path. "
                         "Adds traceprop_factored_kfac*_inlineprecond scores.")
    ap.add_argument("--precond_damping_grid",
                    default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1",
                    help="comma-separated relative-damping candidates, used for BOTH inline-"
                         "precond AND (with --tune_logix) LogIX's preconditioning; the best on "
                         "the held-out val split is chosen and reported on the disjoint eval "
                         "split (lds_heldout_eval). Grid extends below 1e-3 to confirm the peak.")
    ap.add_argument("--dump_cov_shapes", action="store_true",
                    help="print LogIX covariance/eigval state shapes (to verify the "
                         "preconditioning space) + a state-mutation test, then exit before "
                         "retraining")
    ap.add_argument("--logix_permodule_damping", action="store_true",
                    help="with --tune_logix, give LogIX PER-MODULE relative damping (lambda_rel * "
                         "that module's mean K-FAC eigval, monkeypatching precondition_kfac) to "
                         "match Traceprop's per-module scheme, instead of one global absolute "
                         "damping. Also probes whether LogIX's damping=None default-bug fires.")
    ap.add_argument("--tune_logix", action="store_true",
                    help="sweep LogIX's preconditioning damping over the SAME relative grid on "
                         "the SAME val split (converted to LogIX's absolute scale via its own "
                         "mean K-FAC eigenvalue) and report the best as logix_preconditioned_"
                         "tuned -- the tuned-vs-tuned fairness comparison")
    ap.add_argument("--precond_val_frac", type=float, default=0.3,
                    help="fraction of test examples held out to pick the inline-precond "
                         "damping (never used in the reported eval-split LDS)")
    ap.add_argument("--min_target_acc", type=float, default=0.0,
                    help="abort before the retraining loop if the target model's test "
                         "accuracy is below this (guards against running LDS on a "
                         "non-learning model at chance). 0 disables.")
    ap.add_argument("--noise_ceiling", type=int, default=0,
                    help="retrain this many subsets TWICE (different seeds) and report the "
                         "mean per-example Spearman between the two margin sets -- the max LDS "
                         "achievable given retraining noise. 0 disables.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
