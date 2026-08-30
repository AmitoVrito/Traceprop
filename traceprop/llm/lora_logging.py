"""Inline per-sample gradient logging for transformer + LoRA training.

The central systems claim of Traceprop-LLM: attribution should not cost a
second pass over training. Post-hoc methods (TRAK, LoGRA, LoRIF, EK-FAC)
recompute per-sample gradients *after* the run finishes, requiring stored
checkpoints and a full extra forward/backward sweep over the training set.

``LoRAGradientLogger`` instead captures per-sample gradients *during* the
normal training backward pass, using the factored identity for a linear
layer ``y = x Wᵀ``::

    ∂L/∂W  for sample b  =  Σ_t  grad_y[b, t] ⊗ x[b, t]

We never materialise the dense per-sample gradient of the whole model: only
the tracked (small) LoRA adapter linears contribute, the outer product is
formed per layer, concatenated, and compressed by the same sparse
Johnson–Lindenstrauss projection the rest of Traceprop uses. One flush per
optimizer step; no extra forward/backward.

This module deliberately depends only on ``torch`` (not ``transformers`` or
``peft``): it attaches to any ``nn.Linear`` selected by name, so it works
identically for a hand-built transformer block and for a HuggingFace
``PeftModel`` whose adapter exposes ``lora_A`` / ``lora_B`` linears.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from traceprop.attribution.gradient_store import GradientStore


def _is_linear(module: Any) -> bool:
    import torch.nn as nn
    return isinstance(module, nn.Linear)


def select_lora_linears(model: Any, patterns: Iterable[str]) -> "list[tuple[str, Any]]":
    """Return ``(name, module)`` for every ``nn.Linear`` whose qualified name
    contains any of ``patterns`` (e.g. ``("lora_A", "lora_B")`` for PEFT, or
    ``("lora",)`` / a specific layer name for a hand-built model)."""
    out = []
    for name, module in model.named_modules():
        if _is_linear(module) and any(p in name for p in patterns):
            out.append((name, module))
    return out


class LoRAGradientLogger:
    """Attach to selected linear layers and log per-sample gradients inline.

    Parameters
    ----------
    store:
        Destination :class:`GradientStore`. Its sparse JL projection is lazily
        sized to the concatenated per-sample gradient dimension on first flush.
    target_modules:
        List of ``(name, nn.Linear)`` to track. Use :func:`select_lora_linears`.
    source_id:
        Optional provenance tag stored on every entry (e.g. dataset name).

    Usage
    -----
    ::

        logger = LoRAGradientLogger(store, select_lora_linears(model, ("lora_A", "lora_B")))
        for step, (x, y, idx) in enumerate(loader):
            loss = loss_fn(model(x), y)
            loss.backward()
            logger.flush_step(sample_indices=idx)   # <-- inline, no 2nd pass
            optimizer.step(); optimizer.zero_grad()
        logger.detach()
    """

    def __init__(
        self,
        store: GradientStore,
        target_modules: "list[tuple[str, Any]]",
        source_id: Optional[str] = None,
        proj_dim: int = 512,
        seed: int = 42,
    ) -> None:
        if not target_modules:
            raise ValueError(
                "LoRAGradientLogger: no target modules. For a PEFT model pass "
                "select_lora_linears(model, ('lora_A', 'lora_B'))."
            )
        self.store = store
        self.source_id = source_id
        self._targets = target_modules
        self._names = [n for n, _ in target_modules]
        self._fwd_input: dict[str, Any] = {}
        self._grad_output: dict[str, Any] = {}
        self._handles: list[Any] = []
        self._sample_counter = 0
        self._proj_dim = proj_dim
        self._seed = seed
        self._proj_matrix = None  # (D, proj_dim) torch tensor, lazily built on-device
        self.grad_dim: Optional[int] = None  # concatenated per-sample gradient dim
        # Keep the store's declared proj_dim consistent with ours.
        self.store._proj_dim = proj_dim
        self._attach()

    # -- hook plumbing -----------------------------------------------------
    def _attach(self) -> None:
        for name, module in self._targets:
            self._handles.append(
                module.register_forward_hook(self._make_fwd_hook(name))
            )
            self._handles.append(
                module.register_full_backward_hook(self._make_bwd_hook(name))
            )

    def _make_fwd_hook(self, name: str):
        def hook(_module, inputs, _output):
            # inputs[0]: (B, T, d_in) or (B, d_in). Keep on-device; detach.
            self._fwd_input[name] = inputs[0].detach()
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            # grad_output[0]: (B, T, d_out) or (B, d_out); per-sample, unreduced.
            self._grad_output[name] = grad_output[0].detach()
        return hook

    # -- per-step logging --------------------------------------------------
    def _per_sample_grads(self):
        """Concatenated per-sample gradients for the last backward, as an
        on-device ``(B, D)`` torch tensor (or ``None`` if hooks did not fire)."""
        import torch

        per_layer = []
        for name in self._names:
            a = self._fwd_input.get(name)
            g = self._grad_output.get(name)
            if a is None or g is None:
                continue
            if a.dim() == 2:  # (B, d_in) -> (B, 1, d_in)
                a = a.unsqueeze(1)
                g = g.unsqueeze(1)
            # per-sample weight grad: sum over tokens of outer(g, a) -> (B, d_out, d_in)
            gw = torch.einsum("bto,bti->boi", g.float(), a.float())
            per_layer.append(gw.reshape(gw.shape[0], -1))

        if not per_layer:
            return None
        return torch.cat(per_layer, dim=1)  # (B, D), on device

    def _ensure_projection(self, dim: int, device, dtype):
        """Lazily build the sparse Johnson–Lindenstrauss matrix on-device.

        Matches Traceprop's CPU RandomProjection: entries in {-1,0,+1} with
        p={1/6, 2/3, 1/6}, scaled by sqrt(3/proj_dim). Built once, on the same
        device as training, so the projection never leaves the accelerator.
        """
        if self._proj_matrix is not None:
            return
        import torch
        self.grad_dim = dim

        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        probs = torch.tensor([1 / 6, 2 / 3, 1 / 6])
        choice = torch.multinomial(
            probs, dim * self._proj_dim, replacement=True, generator=gen
        )
        vals = torch.tensor([-1.0, 0.0, 1.0])[choice].reshape(dim, self._proj_dim)
        vals *= (3.0 / self._proj_dim) ** 0.5
        self._proj_matrix = vals.to(device=device, dtype=torch.float32)

    def flush_step(
        self,
        sample_indices: Optional[Iterable[int]] = None,
        source_ids: Optional[list] = None,
    ) -> int:
        """Project (on-device) and store per-sample gradients from the most
        recent ``backward()``. Call once per optimizer step, before
        ``zero_grad()``. Returns the number of samples logged this step."""
        import torch

        grads = self._per_sample_grads()  # (B, D) on device
        self._fwd_input.clear()
        self._grad_output.clear()
        if grads is None:
            return 0

        self._ensure_projection(grads.shape[1], grads.device, grads.dtype)
        with torch.no_grad():
            proj = grads @ self._proj_matrix  # (B, proj_dim), on device
        proj_np = proj.detach().cpu().numpy().astype(np.float32, copy=False)

        n = proj_np.shape[0]
        idx_list = list(sample_indices) if sample_indices is not None else \
            list(range(self._sample_counter, self._sample_counter + n))
        ids = self.store.add_projected_batch(
            proj_np, source_id=self.source_id, sample_index_offset=0
        )
        for eid, real_idx in zip(ids, idx_list):
            self.store._entries[eid].sample_index = int(real_idx)
        self._sample_counter += n
        return n

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "LoRAGradientLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()
