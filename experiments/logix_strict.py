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
_FAN_IN_FAN_OUT_MSG = (
    r"^fan_in_fan_out is set to False but the target module is `Conv1D`\. "
    r"Setting fan_in_fan_out to True\.$"
)


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


def validate_logix_gradients(run_, model, tracked_names, xb, yb, loss_fn,
                              data_id, min_cosine=0.9999, check_examples=None):
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
    only direction (cosine). A uniform global scale factor was observed
    between LogIX's logged gradients and direct autograd (consistently near
    2x in spot checks, on both hf and tiny backends, ruling out the .weight
    proxy as the cause since tiny never uses it) -- root cause not fully
    pinned down, but it doesn't affect LDS (rank-correlation based) or the
    validity of this correctness check, since cosine similarity is
    scale-invariant. Flagged in the return value so callers can report it."""
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
            ratios.append(ln / tn)
            if cos < min_cosine:
                raise RuntimeError(
                    f"validate_logix_gradients FAILED: module {n} example {i} "
                    f"cosine={cos:.6f} < {min_cosine} against direct autograd on "
                    f"the same forward pass -- LogIX's logged gradient does not "
                    f"match ground truth. Do not trust LogIX numbers from this "
                    f"setup until this is root-caused."
                )
    return {
        "n_checks": len(ratios),
        "worst_cosine": worst_cosine,
        "scale_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "scale_ratio_min": min(ratios) if ratios else None,
        "scale_ratio_max": max(ratios) if ratios else None,
    }
