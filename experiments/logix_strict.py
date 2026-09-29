"""Shared strict-mode helpers for LogIX experiment scripts (exp31, exp35).

Several silent failures already happened in this codebase because a real
problem only ever showed up as a log line nobody read, or a check that
looked like it passed but wasn't actually exercising the risky path: LogIX's
PCA init quietly falling back to random init ("Hessian state not provided"),
add_lora() silently having no effect before it was ever called at all, and
(caught only via an explicit cosine-vs-autograd check) confirming the
.weight compatibility proxy for PEFT doesn't silently corrupt LogIX's logged
gradients. These helpers turn the ones we know how to check into hard
failures or hard asserts.

Fan-in/fan-out warning handling: PEFT emits exactly one UserWarning on every
normal successful hf run --
  "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting
  fan_in_fan_out to True."
-- auto-correcting a real GPT-2-specific config mismatch (Conv1D vs Linear).
That's benign and expected; turning it into an error would break every run,
not catch a bug. install_strict_warnings() explicitly ignores THAT EXACT
message text (not the whole peft warning category) before escalating any
OTHER UserWarning from peft or logix to a hard error, so an unexpected new
peft warning still fails loudly instead of being silently swallowed by a
category-wide exclusion.
"""
from __future__ import annotations

import warnings

_PATCHED = False
_BF16_PATCHED = False
_FAN_IN_FAN_OUT_MSG = (
    r"^fan_in_fan_out is set to False but the target module is `Conv1D`\. "
    r"Setting fan_in_fan_out to True\.$"
)


def patch_to_numpy_bf16():
    """LogIX's own logix.utils.to_numpy() does `tensor.cpu().detach().numpy()`
    unconditionally, which raises `TypeError: Got unsupported ScalarType
    BFloat16` -- NumPy has no native bf16 type, so PyTorch's .numpy() refuses
    on a bf16 tensor. Only surfaces when the watched model runs in bf16 (e.g.
    --dtype bf16/auto for large models that don't fit fp32); fp32/fp16 both
    convert fine, which is why this was never hit before. Upcast to fp32
    before the numpy conversion -- the value being logged is a small,
    already-projected sketch, not the full gradient, so this is a negligible
    cost, not a shortcut that changes what's being measured.

    logix.logging.log_saver does `from logix.utils import to_numpy`, a direct
    name import that copies the reference into its own module namespace at
    import time -- patching logix.utils.to_numpy alone does NOT affect that
    already-bound reference, so both call sites are patched explicitly.
    Idempotent -- safe to call multiple times."""
    global _BF16_PATCHED
    if _BF16_PATCHED:
        return
    import numpy as np
    import torch
    import logix.utils as _logix_utils
    import logix.logging.log_saver as _log_saver

    def to_numpy_bf16_safe(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            return tensor.cpu().detach().numpy()
        raise ValueError("Unsupported tensor type. Supported libraries: NumPy, PyTorch")

    _logix_utils.to_numpy = to_numpy_bf16_safe
    _log_saver.to_numpy = to_numpy_bf16_safe
    _BF16_PATCHED = True


def patch_loralinear_weight_proxy():
    """PEFT's own LoRA forward (peft/tuners/lora/layer.py) reaches directly
    into `self.lora_A[adapter].weight.dtype` for a dtype cast -- it does NOT
    go through that submodule's forward()/__call__. Once LogIX's
    add_lora() replaces that submodule with its own LoraLinear wrapper
    (which stores the original under ._linear and has no .weight of its
    own), that direct attribute access raises AttributeError and the whole
    forward pass breaks. Only reproduces on the hf/PEFT backend (exp31/
    exp35's tiny backend uses hand-rolled nn.Linear lora_A/lora_B whose
    forward always goes through proper module calls, never direct .weight
    access, so it never hit this). Fix: proxy .weight to the wrapped
    original -- PEFT only reads it, never assigns, so a read-only property
    is enough. Idempotent -- safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return
    from logix.lora.modules import LoraLinear
    LoraLinear.weight = property(lambda self: self._linear.weight)
    _PATCHED = True


def install_strict_warnings():
    """Any UserWarning genuinely raised (via warnings.warn) from logix's OR
    peft's code becomes a hard error, except the one specific, known-benign
    peft fan_in_fan_out message (matched by exact text, not by module/category
    -- see module docstring). Does NOT cover logix's logging.warning() calls
    (e.g. the PCA/Hessian fallback) -- those go through Python's `logging`
    module with propagate=False, which warnings.filterwarnings cannot catch;
    use assert_pca_init_took_effect() for that specific, known risk instead.

    Filter order: warnings.filterwarnings() INSERTS EACH NEW FILTER AT THE
    FRONT of the list, and matching is first-match-wins -- so the "ignore"
    rule must be registered LAST (ending up first) for it to actually
    exclude the fan_in_fan_out message before the broader "error" rules see
    it. Registering "ignore" first (intuitive but wrong) puts it BEHIND the
    "error" rules, which then match first and the exclusion never fires --
    confirmed empirically with a standalone repro before fixing this order."""
    warnings.filterwarnings("error", category=UserWarning, module=r".*peft.*")
    warnings.filterwarnings("error", category=UserWarning, module=r".*logix.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=_FAN_IN_FAN_OUT_MSG)


def assert_pca_init_took_effect(run_, requested_init: str):
    """Call right after run_.add_lora(). LoRAHandler.add_lora() silently
    rewrites self.init_strategy from "pca" to "random" in place (with only a
    logging.warning(), not an exception) if no covariance state exists yet at
    that point -- confirmed by reading logix/lora/lora.py. If we asked for
    pca and didn't get it, that's exactly the kind of thing that must not
    pass silently a second time."""
    if requested_init != "pca":
        return
    actual = getattr(run_.lora_handler, "init_strategy", None)
    if actual != "pca":
        raise RuntimeError(
            f"requested LogIX init='pca' but LoRAHandler fell back to "
            f"'{actual}' -- covariance state was empty when add_lora() ran. "
            f"Run the covariance-accumulation pass (setup({{'forward': "
            f"['covariance'], 'backward': ['covariance']}}) + a pass over "
            f"data + finalize()) BEFORE calling add_lora(), not after."
        )


def measure_logix_bytes_per_example(run_, n_examples):
    """Actual on-disk bytes/example, NOT an analytical rank^2*4*n_layers
    estimate. The earlier analytical formula was wrong in a way that only
    showed up when cross-checked: LoraLinear clamps
    rank=min(requested_rank, in_features, out_features), and the
    REQUESTED rank was being reported/used for the "matched storage" math,
    not the actual (possibly-clamped) rank LogIX applied -- e.g. requesting
    rank=16 against a scope where every tracked module's smaller dimension
    is 8 actually yields an 8x8 core everywhere, not 16x16, silently making
    every downstream "storage-matched" proj_dim wrong. Measuring the real
    serialized log size sidesteps needing to know the effective rank at all.

    Call AFTER run_.finalize() (flushes the LogSaver's buffer to disk) and
    AFTER the real logging pass over n_examples has completed -- reads
    every "log_*.mmap" chunk file's size directly from run_.log_dir. Returns
    bytes/example as a float (whole-directory .mmap total / n_examples);
    metadata JSON files are excluded (small, fixed per-chunk overhead, not
    part of the per-example gradient payload)."""
    import glob
    import os

    mmap_files = glob.glob(os.path.join(run_.log_dir, "log_*.mmap"))
    if not mmap_files:
        raise RuntimeError(
            f"measure_logix_bytes_per_example: no log_*.mmap files found in "
            f"{run_.log_dir} -- was save(True) active and finalize() called "
            f"before this, over a real logging pass?"
        )
    total_bytes = sum(os.path.getsize(f) for f in mmap_files)
    return total_bytes / n_examples


def validate_logix_gradients(run_, model, tracked_names, xb, yb, loss_fn,
                              data_id, min_cosine=0.9999, check_examples=None,
                              max_scale_nonuniformity=1.01):
    """Hard assert: LogIX's logged per-example gradients must match direct
    autograd on the SAME forward pass, to the same cosine >= 0.9999 standard
    Traceprop's own per-sample gradients are held to
    (tests/unit/test_lora_logging.py's test_per_sample_grads_match_autograd).

    Ground truth MUST come from the same forward realization LogIX logged
    (per-example loss + torch.autograd.grad(loss_i, params, retain_graph=True)
    on ONE shared forward pass), not a separate re-run forward pass per
    example -- a separate forward draws fresh dropout masks and compares a
    genuinely different quantity, which looks like a correctness bug (cosine
    as low as 0.34 was observed) but is a harness artifact, not a real one;
    confirmed by rerunning with dropout off (model.eval()), which made the
    separate-forward and same-forward methods agree, and independently by
    using the same-forward method from the start (cosine=1.000000 on every
    example/module tested, hf AND tiny backends).

    Must be called AFTER run_.add_lora() and the {"grad": ["log"]} + save()
    setup, and covers ONLY the batch passed in -- this is a one-time
    correctness certification of the wired-up setup (add_lora(), the .weight
    proxy, tracked-module scope), not something to run every repeat.

    Does NOT assert the per-example gradient's absolute scale matches --
    only direction (cosine). A scale factor was observed between LogIX's
    logged gradients and direct autograd (consistently near 2x in spot
    checks, on both hf and tiny backends, ruling out the .weight proxy as
    the cause since tiny never uses it) -- root cause not fully pinned down.
    A UNIFORM global scale is harmless for LDS (rank-correlation based,
    scale-invariant) and for concatenated-across-modules dot products (a
    single scalar factors out of the inner product). A NON-uniform,
    per-module scale would NOT be harmless -- it would reweight modules
    relative to each other in the concatenated per-example vector even
    though each module's own cosine is 1.0, silently corrupting downstream
    attribution scores. So max_scale_nonuniformity (default 1.01, i.e. the
    largest and smallest observed per-example scale ratio across ALL
    checked examples and modules must be within 1%) is asserted as a hard
    failure condition, on the same footing as the cosine check -- not just
    reported."""
    import torch
    import torch.nn.functional as F

    mods = {n: dict(model.named_modules())[n] for n in tracked_names}
    params = [mods[n].logix_lora_B.weight for n in tracked_names]

    with run_(data_id=data_id):
        model.zero_grad(set_to_none=True)
        loss_per_example = loss_fn(xb, yb)  # must return a list/tuple of per-example scalar losses
        total = sum(loss_per_example)
        # retain_graph=True: the graph must survive this backward() so the
        # per-example torch.autograd.grad() calls below can reuse it -- a
        # second, separate forward pass would draw fresh dropout masks and
        # compare a different quantity (see docstring).
        total.backward(retain_graph=True)

    logged_data_id, log = run_.get_log()

    n_examples = len(loss_per_example)
    examples = range(n_examples) if check_examples is None else check_examples
    worst_cosine = float("inf")
    ratios = []
    ratios_by_module = {n: [] for n in tracked_names}
    for i in examples:
        grads = torch.autograd.grad(loss_per_example[i], params, retain_graph=True)
        for n, g in zip(tracked_names, grads):
            key = n + ".logix_lora_B"
            logged_grad = log[key]["grad"][i].flatten().float()
            true_grad = g.flatten().float()
            tn, ln = true_grad.norm().item(), logged_grad.norm().item()
            if tn < 1e-9 and ln < 1e-9:
                continue  # both genuinely ~0 (e.g. lora_A at fresh LoRA init, B=0) -- not a failure
            if tn < 1e-9 or ln < 1e-9:
                raise RuntimeError(
                    f"validate_logix_gradients: module {n} example {i} -- one of "
                    f"true/logged gradient is ~0 and the other isn't "
                    f"(true_norm={tn:.6g}, logged_norm={ln:.6g}); can't compute a "
                    f"meaningful cosine, and this itself looks like a real mismatch."
                )
            cos = (true_grad @ logged_grad).item() / (tn * ln)
            worst_cosine = min(worst_cosine, cos)
            ratio = ln / tn
            ratios.append(ratio)
            ratios_by_module[n].append(ratio)
            if cos < min_cosine:
                raise RuntimeError(
                    f"validate_logix_gradients FAILED: module {n} example {i} "
                    f"cosine={cos:.6f} < {min_cosine} against direct autograd on "
                    f"the same forward pass -- LogIX's logged gradient does not "
                    f"match ground truth. Do not trust LogIX numbers from this "
                    f"setup until this is root-caused."
                )

    if ratios:
        nonuniformity = max(ratios) / min(ratios)
        if nonuniformity > max_scale_nonuniformity:
            per_module_report = {
                n: (round(min(r), 6), round(max(r), 6)) for n, r in ratios_by_module.items() if r
            }
            raise RuntimeError(
                f"validate_logix_gradients FAILED: LogIX/autograd scale ratio is NOT "
                f"uniform across modules/examples -- max/min={nonuniformity:.4f} > "
                f"{max_scale_nonuniformity}. A non-uniform per-module scale silently "
                f"reweights modules in the concatenated per-example attribution vector "
                f"even though each module's own cosine is 1.0 -- do not treat this as "
                f"a harmless global scale. Per-module (min, max) ratio: {per_module_report}"
            )

    return {
        "n_checks": len(ratios),
        "worst_cosine": worst_cosine,
        "scale_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "scale_ratio_min": min(ratios) if ratios else None,
        "scale_ratio_max": max(ratios) if ratios else None,
        "scale_nonuniformity": max(ratios) / min(ratios) if ratios else None,
    }


def precondition_kfac_permodule(src, state, damping=None):
    """Drop-in replacement for logix's precondition_kfac with PER-MODULE relative
    damping. The passed ``damping`` is interpreted as a RELATIVE strength lambda_rel;
    each module's absolute damping is ``lambda_rel * mean(that module's K-FAC
    eigenvalues)`` -- matching Traceprop's inline per-module scheme.

    The rotation/division math is a faithful copy of logix's own precondition_kfac
    (einops einsum in the covariance eigenbasis); the ONLY change is how the damping
    scalar is derived per module. Two safety differences from the original: the
    eigenvalue tensor is updated OUT OF PLACE (the original's ``full_eigval +=``
    can alias state), and the ``damping is None`` module-0 leak is avoided by
    always computing a per-module value. A unit test asserts that, given a UNIFORM
    absolute damping, this reproduces logix's original output exactly, so any LDS
    difference comes from the damping VALUE, not a reimplementation artifact.
    """
    import torch
    from einops import einsum
    from logix.utils import nested_dict

    preconditioned = nested_dict()
    cov_eigval, cov_eigvec = state.get_covariance_svd_state()
    lam_rel = 0.1 if damping is None else damping
    for module_name in src.keys():
        src_grad = src[module_name]["grad"]
        device = src_grad.device
        module_eigvec = cov_eigvec[module_name]
        fwd_eigvec = module_eigvec["forward"].to(device=device)
        bwd_eigvec = module_eigvec["backward"].to(device=device)
        module_eigval = cov_eigval[module_name]
        if isinstance(module_eigval, torch.Tensor):
            full_eigval = module_eigval.to(device=device)
        else:
            fwd_eigval = module_eigval["forward"]
            bwd_eigval = module_eigval["backward"]
            full_eigval = torch.outer(bwd_eigval, fwd_eigval).to(device=device)
        full_eigval = full_eigval + lam_rel * torch.mean(full_eigval)  # per-module, out-of-place
        rotated_grad = einsum(
            bwd_eigvec.t(), src_grad, fwd_eigvec,
            "a b, batch b c, c d -> batch a d",
        )
        prec_rotated_grad = rotated_grad / full_eigval
        preconditioned[module_name]["grad"] = einsum(
            bwd_eigvec, prec_rotated_grad, fwd_eigvec.t(),
            "a b, batch b c, c d -> batch a d",
        )
    return preconditioned
