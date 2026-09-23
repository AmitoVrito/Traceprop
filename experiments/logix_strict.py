"""Shared strict-mode helpers for LogIX experiment scripts (exp31, exp35).

Two silent failures already happened in this codebase because a real problem
only ever showed up as a log line nobody read: LogIX's PCA init quietly
falling back to random init ("Hessian state not provided"), and add_lora()
silently having no effect before it was ever called at all. These helpers
turn the ones we know how to check into hard failures.

NOT included: PEFT's "fan_in_fan_out is set to False but the target module is
`Conv1D`" warning. That one is PEFT auto-correcting a real GPT-2-specific
config mismatch (Conv1D vs Linear) and fires on every normal successful hf
run -- turning it into an error would break every run, not catch a bug.
Documented here rather than silently included in a blanket filter.
"""
from __future__ import annotations

import warnings

_PATCHED = False


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
    """Any UserWarning genuinely raised (via warnings.warn) from logix's own
    code becomes a hard error. Does NOT cover logix's logging.warning() calls
    (e.g. the PCA/Hessian fallback) -- those go through Python's `logging`
    module with propagate=False, which warnings.filterwarnings cannot catch;
    use assert_pca_init_took_effect() for that specific, known risk instead."""
    warnings.filterwarnings("error", category=UserWarning, module=r".*logix.*")


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
