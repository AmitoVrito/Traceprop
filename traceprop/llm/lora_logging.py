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


def _block_index(name: str) -> Optional[int]:
    """Extract a transformer block index from a module's qualified name.

    Handles the common layouts: ``transformer.h.11.…`` (GPT-2),
    ``gpt_neox.layers.23.…`` (Pythia/NeoX), ``model.layers.7.…`` (Llama).
    Returns ``None`` if no block index is present.
    """
    import re
    m = re.search(r"(?:^|\.)(?:h|layers|blocks)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def select_lora_linears(
    model: Any,
    patterns: Iterable[str],
    last_n_blocks: Optional[int] = None,
) -> "list[tuple[str, Any]]":
    """Return ``(name, module)`` for every ``nn.Linear`` whose qualified name
    contains any of ``patterns`` (e.g. ``("lora_A", "lora_B")`` for PEFT).

    If ``last_n_blocks`` is set, keep only modules in the highest ``n`` block
    indices — i.e. last-layer / last-few-block attribution. This is the cheap,
    high-signal regime (Traceprop-LL): the projected-gradient dimension, and
    therefore the JL projection matrix and its per-step read cost, scale with
    the number of tracked parameters, so restricting to the final block(s)
    is what keeps inline logging in the sub-1% overhead regime.
    """
    patterns = tuple(patterns)
    matched = [
        (name, module)
        for name, module in model.named_modules()
        if _is_linear(module) and any(p in name for p in patterns)
    ]
    if last_n_blocks is None:
        return matched

    idxs = [i for i in (_block_index(n) for n, _ in matched) if i is not None]
    if not idxs:
        return matched  # no block structure detected; track everything matched
    cutoff = max(idxs) - last_n_blocks + 1
    return [
        (name, module)
        for name, module in matched
        if (_block_index(name) is None or _block_index(name) >= cutoff)
    ]


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
        factored: bool = False,
        kfac: int = 16,
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
        # Factored (Kronecker) sketching: project the two small gradient factors
        # (input activation, output grad) separately and combine, avoiding the
        # dense-matrix read. Cheaper per step, but token cross-terms add variance
        # so its attribution quality per stored dimension is worse than the dense
        # JL projection — OFF by default; dense + overlap/buffering is preferred.
        self.factored = factored
        self._kfac = kfac
        self._fac_PQ: dict[str, Any] = {}  # name -> (P, Q) per-layer sketch mats
        self._proj_matrix = None  # dense fallback (D, proj_dim), lazily built
        self.grad_dim: Optional[int] = None  # true concatenated per-sample grad dim
        self.sketch_dim: Optional[int] = None  # stored (projected) dim
        self._gpu_buffer: list = []  # deferred projected batches (on device)
        self._gpu_buffer_idx: list = []  # matching sample indices
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

    def _sparse_jl(self, k: int, d: int, device, salt: int):
        """A (k, d) sparse sign JL matrix: entries in {-1,0,+1} with
        p={1/6,2/3,1/6}, scaled by sqrt(3/k) so E[MᵀM] = I (inner-product
        preserving). Deterministic in (seed, salt)."""
        import torch
        gen = torch.Generator(device="cpu").manual_seed(self._seed + salt)
        choice = torch.multinomial(
            torch.tensor([1 / 6, 2 / 3, 1 / 6]), k * d, replacement=True, generator=gen
        )
        vals = torch.tensor([-1.0, 0.0, 1.0])[choice].reshape(k, d)
        vals *= (3.0 / k) ** 0.5
        return vals.to(device=device, dtype=torch.float32)

    def _factored_sketch(self):
        """Kronecker-factored sketch of the per-sample gradients from the last
        backward, as an on-device ``(B, sketch_dim)`` tensor (or ``None``).

        For a linear layer with input a:(B,T,d_in) and output-grad g:(B,T,d_out),
        the per-sample gradient is Σ_t g_t ⊗ a_t. We sketch it as
        Σ_t (P g_t) ⊗ (Q a_t) with small JL maps P:(k_out,d_out), Q:(k_in,d_in) —
        never materialising the d_out·d_in outer product or a large projection
        matrix. The dot product of two sketches is an unbiased estimate of the
        true gradient dot product (what attribution needs)."""
        import torch

        parts = []
        grad_dim_total = 0
        for salt, name in enumerate(self._names):
            a = self._fwd_input.get(name)
            g = self._grad_output.get(name)
            if a is None or g is None:
                continue
            if a.dim() == 2:
                a = a.unsqueeze(1)
                g = g.unsqueeze(1)
            d_in, d_out = a.shape[-1], g.shape[-1]
            grad_dim_total += d_out * d_in
            k_out, k_in = min(d_out, self._kfac), min(d_in, self._kfac)
            PQ = self._fac_PQ.get(name)
            if PQ is None:
                P = self._sparse_jl(k_out, d_out, a.device, salt * 2 + 1)
                Q = self._sparse_jl(k_in, d_in, a.device, salt * 2 + 2)
                self._fac_PQ[name] = PQ = (P, Q)
            P, Q = PQ
            with torch.no_grad():
                Pg = torch.einsum("bto,ko->btk", g.float(), P)   # (B,T,k_out)
                Qa = torch.einsum("bti,li->btl", a.float(), Q)   # (B,T,k_in)
                S = torch.einsum("btk,btl->bkl", Pg, Qa)          # (B,k_out,k_in)
            parts.append(S.reshape(S.shape[0], -1))

        if not parts:
            return None
        self.grad_dim = grad_dim_total
        sketch = torch.cat(parts, dim=1)
        self.sketch_dim = sketch.shape[1]
        return sketch

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
        buffer: bool = False,
    ) -> int:
        """Project (on-device) per-sample gradients from the most recent
        ``backward()``. Call once per optimizer step, before ``zero_grad()``.

        With ``buffer=True`` the projected ``(B, proj_dim)`` tensor is kept on
        the accelerator and *not* copied to host — the device→host transfer
        (and the sync it forces) is deferred to :meth:`drain`. This keeps the
        projection off the critical path so it overlaps with the training step,
        which is what makes inline logging sub-1% in a real loop. Otherwise the
        projected batch is moved to the store immediately.

        Returns the number of samples logged this step.
        """
        import torch

        if self.factored:
            proj = self._factored_sketch()  # (B, sketch_dim) on device
            self._fwd_input.clear()
            self._grad_output.clear()
            if proj is None:
                return 0
            self.store._proj_dim = proj.shape[1]
        else:
            grads = self._per_sample_grads()  # (B, D) on device
            self._fwd_input.clear()
            self._grad_output.clear()
            if grads is None:
                return 0
            self._ensure_projection(grads.shape[1], grads.device, grads.dtype)
            with torch.no_grad():
                proj = grads @ self._proj_matrix  # (B, proj_dim), on device

        n = proj.shape[0]
        idx_list = list(sample_indices) if sample_indices is not None else \
            list(range(self._sample_counter, self._sample_counter + n))
        self._sample_counter += n

        if buffer:
            self._gpu_buffer.append(proj)
            self._gpu_buffer_idx.extend(int(i) for i in idx_list)
            return n

        self._commit(proj.detach().cpu().numpy().astype(np.float32, copy=False), idx_list)
        return n

    def _commit(self, proj_np: np.ndarray, idx_list: list) -> None:
        ids = self.store.add_projected_batch(
            proj_np, source_id=self.source_id, sample_index_offset=0
        )
        for eid, real_idx in zip(ids, idx_list):
            self.store._entries[eid].sample_index = int(real_idx)

    def drain(self) -> int:
        """Move all buffered projected gradients to the store in one host
        transfer. Call at the end of training (or every K steps). Returns the
        number of samples committed."""
        if not self._gpu_buffer:
            return 0
        import torch
        allproj = torch.cat(self._gpu_buffer, dim=0).detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        self._commit(allproj, self._gpu_buffer_idx)
        n = allproj.shape[0]
        self._gpu_buffer.clear()
        self._gpu_buffer_idx.clear()
        return n

    def detach(self) -> None:
        self.drain()
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "LoRAGradientLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()
