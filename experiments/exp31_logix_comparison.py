"""exp31 — Head-to-head vs. LogIX (Choe et al., github.com/logix-project/logix).

Reviewer item 12: getting per-sample gradients from forward/backward hooks is
the same trick LogIX and Opacus use. This measures LogIX's own per-step
training overhead, hooked into the SAME training loop and SAME tracked-module
scope (last transformer block's LoRA adapters) used by exp25/exp30's
LoRAGradientLogger measurement, so the two overhead numbers are directly
comparable rather than each being quoted from a different paper's setup.

Two backends:
  --backend tiny  Small from-scratch classifier (exp27's tiny-clf, CPU). Ran
                  already: LogIX 12.9% +/- 19.3%, LoRAGradientLogger 25.9% +/-
                  14.7% (n=100 repeats each). Held pending this backend's GPU
                  number before writing anything into the paper -- that CPU
                  result is the opposite regime from the paper's ~1% claim.
  --backend hf    GPT-2 / Pythia + PEFT LoRA, matching exp25 exactly (same
                  model, same tracked scope, same batch/seq). This is the
                  number that actually matters for the paper.

Requires logix-ai, which caps at python<3.11:
    pip install logix-ai   (or: pip install git+https://github.com/logix-project/logix.git)
Run with a Python 3.10 (or earlier) interpreter. On Colab, use
!pip -q install "logix-ai" (Colab's default Python is 3.10/3.11 -- check
`python --version`; if it's 3.11, this will fail the same way it does here).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from exp25_llm_inline_overhead import build_tiny_model, build_hf_model, hf_batch, tiny_batch
from exp27_lds_quality import build_tiny_classifier, synthetic_data
from logix_strict import (
    install_strict_warnings, assert_pca_init_took_effect,
    patch_loralinear_weight_proxy, validate_logix_gradients,
)


def run(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import logix

    install_strict_warnings()
    patch_loralinear_weight_proxy()

    device = args.device
    if device == "cuda" and not getattr(args, "skip_gpu_check", False):
        want = getattr(args, "gpu_check", "L4")
        got = torch.cuda.get_device_name(0)
        if want and want not in got:
            raise SystemExit(
                f"expected a GPU containing '{want}' but got '{got}' -- refusing to run "
                f"(pass --gpu_check '' to disable, or --gpu_check <substring> to expect something else)."
            )

    # add_lora() mutates the model IN PLACE, wrapping each tracked module in a
    # LoraLinear(encoder/bottleneck/decoder/_linear) wrapper -- calling it a
    # second time on an already-wrapped model wraps the wrapper (confirmed
    # empirically: tracked module count and shapes go completely wrong,
    # "logix_lora_B.logix_lora_B: Linear(8,8)" etc). Since this script builds
    # two LogIX configs (logix_default_disk, matched_buffering), each needs
    # its OWN freshly-built, never-wrapped model -- hence build_model() as a
    # factory instead of a single shared `model` variable.
    if args.backend == "tiny":
        def build_model():
            torch.manual_seed(1234)
            np.random.seed(1234)
            return build_tiny_classifier(args.vocab, seq=args.seq, r=args.rank).to(device)

        Xtr, ytr = synthetic_data(args.n_train, args.seq, args.vocab, seed=0)
        Xtr_t = torch.tensor(Xtr, device=device)
        ytr_t = torch.tensor(ytr, device=device)

        _template = build_model()
        n_blocks = len(_template.blocks)
        last_block_idx = n_blocks - 1
        tracked_names = [
            n for n, m in _template.named_modules()
            if isinstance(m, nn.Linear)
            and (n == "score" or (f"blocks.{last_block_idx}." in n and ("lora_A" in n or "lora_B" in n)))
        ]
        del _template

        def make_loss_fn(model):
            return lambda xb, yb: F.cross_entropy(model(xb), yb, reduction="sum")

        def batch(step):
            s = (step * args.batch) % (args.n_train - args.batch)
            return Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch], s

    else:  # hf: exact same model/scope as exp25
        def build_model():
            return build_hf_model(args.model, r=args.rank).to(device)

        x = hf_batch(args.model, args.seq, args.batch, device)
        last_n = None if args.track <= 0 else args.track
        _template = build_model()
        tracked_names = [
            n for n, m in _template.named_modules()
            if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n)
        ]
        if args.track > 0:
            # keep only last-N-block adapters, matching exp25's select_lora_linears(last_n_blocks=...)
            import re
            def block_idx(name):
                m = re.search(r"(?:^|\.)(?:h|layers)\.(\d+)\.", name)
                return int(m.group(1)) if m else None
            idxs = sorted({block_idx(n) for n in tracked_names if block_idx(n) is not None})
            keep = set(idxs[-args.track:])
            tracked_names = [n for n in tracked_names if block_idx(n) in keep]
        del _template

        def make_loss_fn(model):
            def loss_fn(xb, yb=None):
                logits = model(xb).logits
                return F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    xb[:, 1:].reshape(-1),
                )
            return loss_fn

        def batch(step):
            return x, None, step * args.batch

    print(f"[exp31] backend={args.backend} tracking {len(tracked_names)} modules")

    # --- Storage matching -------------------------------------------------
    # Ours: a single dense sparse-JL projection to proj_dim floats, total
    # args.proj_dim * 4 bytes/example regardless of tracked-layer count.
    # LogIX: a rank x rank core matrix PER tracked layer (LoraLinear inserts
    # A: in->rank, B: rank->rank, C: rank->out; the per-example-varying part
    # is the B core), so its per-example storage is
    # n_layers * rank^2 * 4 bytes and grows with both rank AND layer count.
    # Its own default (rank=64) with our 6-layer GPT-2 scope would store
    # ~96KB/example -- ~48x our 2KB budget -- which would make "LogIX is
    # slower" partly a "LogIX is doing more work by default" result, not a
    # mechanism result. Two ways to remove that confound, both run here:
    #   rank_mode=matched        solve LogIX's rank down to OUR proj_dim budget
    #   rank_mode=logix_default  leave LogIX at its own default rank (no
    #                            override); pairs with a SEPARATE exp25 run at
    #                            --proj_dim reverse_proj_dim (printed below) so
    #                            Traceprop is grown UP to LogIX's budget instead
    n_tracked_layers = len(tracked_names)
    our_bytes_per_example = args.proj_dim * 4
    if args.rank_mode == "matched":
        matched_rank = max(1, int((our_bytes_per_example / (4 * n_tracked_layers)) ** 0.5))
        logix_bytes_per_example = n_tracked_layers * (matched_rank ** 2) * 4
        print(f"[exp31] rank_mode=matched: ours={our_bytes_per_example}B/example "
              f"({args.proj_dim} floats), LogIX rank set to {matched_rank} for "
              f"{n_tracked_layers} tracked layers -> {logix_bytes_per_example}B/example "
              f"(analytical -- not yet verified against LogIX's actual serialized log size)")
    else:  # logix_default: don't touch LogIX's own rank; report the reverse-match point instead
        matched_rank = None
        default_rank = args.logix_default_rank
        logix_bytes_per_example = n_tracked_layers * (default_rank ** 2) * 4
        reverse_proj_dim = max(1, logix_bytes_per_example // 4)
        print(f"[exp31] rank_mode=logix_default: LogIX left at its own default rank="
              f"{default_rank} for {n_tracked_layers} tracked layers -> "
              f"{logix_bytes_per_example}B/example. For the reverse-match comparison point, "
              f"rerun exp25/exp26 with --proj_dim {reverse_proj_dim} "
              f"({reverse_proj_dim * 4}B/example) to grow Traceprop up to this same budget.")

    validation_stats = []

    def build_run(save_to_disk, init_strategy):
        """Fresh model + fresh LogIX instance, wrapped with add_lora() exactly
        once -- see the note above build_model() on why this can't reuse a
        shared model across configs. Returns (run_, model, loss_fn, opt,
        covariance_pass_s) -- the last is 0.0 for init_strategy='random',
        and the real wall-clock of the covariance-accumulation pass for
        'pca' (must be counted as part of LogIX's cost for that config, not
        hidden -- it's a genuine extra pass over the data LogIX's authors
        recommend for LoRA, the same "second pass" shape this paper argues
        against elsewhere, so it has to be reported, not omitted)."""
        model = build_model()
        loss_fn = make_loss_fn(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(trainable, lr=1e-3)

        # logix.init() is a process-level singleton; use LogIX(...) directly so we
        # can build a second instance for the second (matched-buffering) config.
        run_ = logix.LogIX(project=f"exp31_{os.getpid()}_{save_to_disk}_{init_strategy}",
                            config="exp31_config.yaml")
        run_.config.lora.init = init_strategy
        if matched_rank is not None:
            run_.config.lora.rank = matched_rank  # storage-matched, see above
        # else: leave LogIX's own default rank (LoRAConfig.rank=64) untouched
        run_.watch(model, name_filter=tracked_names, type_filter=[nn.Linear])

        covariance_pass_s = 0.0
        if init_strategy == "pca":
            # PCA init needs per-module forward/backward covariance BEFORE
            # add_lora() runs (LoRAHandler.add_lora() reads
            # self._state.get_covariance_state(), keyed by the ORIGINAL
            # pre-wrap module names) -- confirmed by reading logix/lora/lora.py.
            # Without this pass, add_lora() finds an empty covariance state
            # and silently falls back to random init (see logix_strict.py);
            # assert_pca_init_took_effect() below catches that if it happens.
            run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(args.pca_cov_steps):
                xb, yb, s = batch(step)
                opt.zero_grad(set_to_none=True)
                with run_(data_id=[str(s + i) for i in range(args.batch)]):
                    loss_fn(xb, yb).backward()
                opt.step()
            run_.finalize()
            if device == "cuda":
                torch.cuda.synchronize()
            covariance_pass_s = time.perf_counter() - t0

        # CRITICAL: watch() alone does NOT apply LogIX's low-rank compression --
        # it just hooks whatever module it's given at that module's own shape.
        # is_lora(model) (which gates the compressed logging path) checks for
        # LogIX's own "logix_lora_B" naming, which only exists after add_lora()
        # inserts its encoder/bottleneck/decoder wrapper. Earlier runs of this
        # script set run_.config.lora.rank but never called add_lora() -- so
        # the rank setting had NO EFFECT and LogIX was logging raw, uncompressed
        # gradients of our own (already rank-8) PEFT adapters the whole time.
        # Confirmed by inspecting is_lora()/add_lora() source and empirically
        # verifying the tracked module list before/after this call: without
        # add_lora(), tracked modules are our own "lora_A"/"lora_B" at their
        # native shape; with it, they become "logix_lora_B: Linear(rank, rank)".
        # Calling add_lora() a SECOND time on an already-wrapped model wraps
        # the wrapper (confirmed empirically) -- hence a brand-new model here.
        run_.add_lora()
        assert_pca_init_took_effect(run_, init_strategy)
        # {"grad": ["log"]} only -- deliberately NOT requesting "covariance" or
        # "hessian" statistics during the TIMED logging pass below. LogIX's
        # Hessian/EK-FAC machinery is opt-in via those keys; omitting them
        # here means no additional covariance accumulation happens during the
        # timed comparison itself (the covariance pass above, when it runs,
        # is timed and reported separately, not hidden inside this cost).
        run_.setup({"grad": ["log"]})
        run_.save(save_to_disk)

        if not getattr(args, "skip_gradient_validation", False):
            def per_example_loss_fn(xb, yb):
                if args.backend == "tiny":
                    raw = F.cross_entropy(model(xb), yb, reduction="none")
                else:
                    logits = model(xb).logits
                    raw = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        xb[:, 1:].reshape(-1), reduction="none",
                    ).reshape(xb.shape[0], -1).sum(dim=1)
                return [raw[i] for i in range(raw.shape[0])]

            check_xb, check_yb, check_s = batch(0)
            n_chk = min(len(check_xb) if args.backend == "tiny" else check_xb.shape[0], 4)
            check_xb, check_yb = check_xb[:n_chk], (check_yb[:n_chk] if check_yb is not None else None)
            stats = validate_logix_gradients(
                run_, model, tracked_names, check_xb, check_yb, per_example_loss_fn,
                data_id=[str(check_s + i) for i in range(n_chk)],
            )
            print(f"[exp31] gradient validation ({init_strategy}, save_to_disk={save_to_disk}) "
                  f"OK: worst_cosine={stats['worst_cosine']:.6f} over {stats['n_checks']} checks, "
                  f"scale_ratio mean/min/max={stats['scale_ratio_mean']:.3f}/"
                  f"{stats['scale_ratio_min']:.3f}/{stats['scale_ratio_max']:.3f}")
            validation_stats.append({"init_strategy": init_strategy, "save_to_disk": save_to_disk, **stats})

        return run_, model, loss_fn, opt, covariance_pass_s

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def reset_peak_mem():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_mem_mb():
        if device != "cuda":
            return None
        return torch.cuda.max_memory_allocated() / 1024 ** 2

    def make_step_fn(loss_fn, opt, lrun=None):
        """lrun=None -> plain training step, no LogIX involvement at all (used
        for the true baseline, on a model add_lora() never touched -- see the
        note in build_run() on why the baseline can't reuse the wrapped
        model: LoraLinear.forward() unconditionally runs its
        encoder/bottleneck/decoder matmuls even outside the logging context,
        since the wrapping is architectural, not hook-conditional)."""
        def step(step_idx):
            xb, yb, s = batch(step_idx)
            opt.zero_grad(set_to_none=True)
            if lrun is not None:
                with lrun(data_id=[str(s + i) for i in range(args.batch)]):
                    loss_fn(xb, yb).backward()
            else:
                loss_fn(xb, yb).backward()
            opt.step()
        return step

    def warmup(step_fn):
        for step in range(args.warmup):
            step_fn(step)
        sync()

    def block(step_fn):
        sync()
        t0 = time.perf_counter()
        for step in range(args.steps):
            step_fn(step)
        sync()
        return time.perf_counter() - t0

    def measure(save_to_disk, init_strategy):
        """One full interleaved measurement (baseline vs LogIX), matched
        warmup/reps/sync policy, at a given LogIX disk-flush setting and init
        strategy. Baseline runs on a freshly-built, never-add_lora()'d model;
        the LogIX side runs on its own freshly-built, add_lora()-wrapped
        model -- two separate models, not one shared one (see build_run()'s
        docstring)."""
        base_model = build_model()
        base_loss_fn = make_loss_fn(base_model)
        base_opt = torch.optim.SGD(
            [p for p in base_model.parameters() if p.requires_grad], lr=1e-3)
        base_step = make_step_fn(base_loss_fn, base_opt, lrun=None)

        lrun, model, loss_fn, opt, covariance_pass_s = build_run(save_to_disk, init_strategy)
        logix_step = make_step_fn(loss_fn, opt, lrun=lrun)

        warmup(base_step)
        warmup(logix_step)

        overheads, base_times = [], []
        base_peak_mb, logix_peak_mb = [], []
        for _ in range(args.repeats):
            reset_peak_mem()
            b = block(base_step)
            base_peak_mb.append(peak_mem_mb())
            reset_peak_mem()
            i = block(logix_step)
            logix_peak_mb.append(peak_mem_mb())
            overheads.append((i - b) / b * 100.0)
            base_times.append(b)

        med = statistics.median(overheads)
        std = statistics.pstdev(overheads) if len(overheads) > 1 else 0.0
        base_step_ms = statistics.median(base_times) / args.steps * 1e3
        base_total_s = sum(base_times)
        return {
            "save_to_disk": save_to_disk,
            "init_strategy": init_strategy,
            "base_step_ms": round(base_step_ms, 4),
            "overhead_pct_median": round(med, 3),
            "overhead_pct_std": round(std, 3),
            "overhead_samples": [round(x, 3) for x in overheads],
            "covariance_pass_s": round(covariance_pass_s, 4),
            # covariance pass is a ONE-TIME cost (PCA init only), not per-repeat --
            # reported both as raw seconds and as a % of this config's total
            # measured baseline time, so it isn't silently invisible next to the
            # per-step overhead numbers above.
            "covariance_pass_pct_of_measured_baseline": (
                round(covariance_pass_s / base_total_s * 100, 2) if base_total_s > 0 else 0.0
            ),
            "peak_mem_mb_baseline": round(statistics.median([m for m in base_peak_mb if m]), 2) if device == "cuda" else None,
            "peak_mem_mb_logix": round(statistics.median([m for m in logix_peak_mb if m]), 2) if device == "cuda" else None,
        }

    # Configurations, per the fairness protocol: LogIX's own literal default
    # (random init, writes to disk as it goes) and its authors' recommended
    # setting for LoRA (PCA init, which needs the extra covariance pass timed
    # above -- a genuine second pass over the data, reported honestly rather
    # than hidden). Both at matched-buffering disk policy (in-memory, no
    # per-step host round-trip) so the disk-vs-init axes aren't conflated.
    configs = {}
    print("[exp31] measuring LogIX default (random init, matched-buffering) ...")
    configs["default_random_init"] = measure(save_to_disk=False, init_strategy="random")
    print("[exp31] measuring LogIX recommended (pca init, matched-buffering, "
          "covariance pass timed) ...")
    configs["recommended_pca_init"] = measure(save_to_disk=False, init_strategy="pca")

    out = {
        "tool": "logix (logix-project/logix)",
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "tracked_modules_count": len(tracked_names),
        "track_last_n_blocks": args.track,
        "steps": args.steps, "repeats": args.repeats, "warmup": args.warmup,
        "rank_mode": args.rank_mode,
        "storage_matching": {
            "ours_bytes_per_example": our_bytes_per_example,
            "ours_proj_dim": args.proj_dim,
            "logix_rank_used": matched_rank if matched_rank is not None else args.logix_default_rank,
            "logix_rank_overridden": matched_rank is not None,
            "logix_bytes_per_example_analytical": logix_bytes_per_example,
            "reverse_match_proj_dim": None if args.rank_mode == "matched" else
                max(1, logix_bytes_per_example // 4),
            "note": "analytical (rank^2 * 4 bytes * n_layers), not yet verified against "
                    "LogIX's actual serialized log file size -- check this on the next run. "
                    "rank_mode=matched shrinks LogIX to our budget; rank_mode=logix_default "
                    "leaves LogIX at its own default and reports the proj_dim needed to grow "
                    "Traceprop up to LogIX's budget instead (the reverse-match point).",
        },
        "gradient_validation": validation_stats,
        "configs": configs,
        "note": "directly comparable to exp25/exp30's LoRAGradientLogger overhead numbers "
                "(same model, scope, batch/seq, interleaved-block timing methodology). "
                "default_random_init requests no covariance (LogIX's literal default); "
                "recommended_pca_init requests forward/backward covariance for PCA init "
                "(its authors' recommended LoRA setting) and times that pass separately "
                "in covariance_pass_s -- see logix_preconditioned in exp35 for the "
                "corresponding LDS-side comparison.",
    }
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    fn = getattr(args, "out", None) or \
        f"results/exp31_logix_{args.backend}_{out['model'].replace('/', '_')}_{args.rank_mode}.json"
    if os.path.exists(fn) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {fn}. Pass --out <path> for a different "
            f"filename, or --force to overwrite."
        )
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
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--proj_dim", type=int, default=512,
                    help="reference for storage matching -- our proj_dim elsewhere "
                         "(exp25/exp30 default 512 = 2KB/example); used to solve LogIX's "
                         "rank down when rank_mode=matched, ignored (informational only) "
                         "when rank_mode=logix_default")
    ap.add_argument("--rank_mode", choices=["matched", "logix_default"], default="matched",
                    help="matched: shrink LogIX's rank to our proj_dim budget (item 1's "
                         "primary comparison). logix_default: leave LogIX at its own "
                         "default rank and report the proj_dim needed to grow Traceprop "
                         "up to LogIX's budget instead (the reverse-match point) -- rerun "
                         "exp25/exp26 with that --proj_dim separately to get Traceprop's "
                         "own overhead number at that budget")
    ap.add_argument("--logix_default_rank", type=int, default=64,
                    help="LogIX's own default LoRA rank (logix/config.py's LoRAConfig); "
                         "used only for the logix_bytes_per_example print/report when "
                         "rank_mode=logix_default (the actual run uses LogIX's real default, "
                         "this value is just for the printed math to match)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pca_cov_steps", type=int, default=20,
                    help="steps used to accumulate forward/backward covariance for LogIX's "
                         "recommended PCA init (recommended_pca_init config only); this pass "
                         "is timed and reported separately (covariance_pass_s), not hidden "
                         "inside the per-step overhead numbers")
    ap.add_argument("--out", default=None, help="output path override (default: auto from backend/model)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    ap.add_argument("--skip_gradient_validation", action="store_true",
                    help="skip the one-time cosine-vs-autograd check of LogIX's logged "
                         "gradients (runs once per config by default -- cheap, catches "
                         "wiring bugs like the add_lora()/PEFT .weight incompatibility)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
