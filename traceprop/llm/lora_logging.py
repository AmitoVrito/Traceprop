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
        inline_precond: bool = False,
        grouped_dispatch: bool = False,
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
        # Inline K-FAC preconditioning: accumulate the covariance of the two
        # PROJECTED gradient factors (P g, Q a) per module DURING the factored
        # logging pass -- reusing the same Pg/Qa the sketch already computes, so
        # no second (covariance) pass over the data. Stored sketches are left
        # untouched (never whitened at logging time, which would let noisy
        # early-training covariance corrupt records); the k x k covariance is
        # applied only at query time via apply_kfac_precondition(). Off by
        # default. See kfac_covariances() to read the accumulated statistics.
        self.inline_precond = inline_precond
        self._cov_G: dict[str, Any] = {}   # name -> (k_out, k_out) Sum Pg Pg^T
        self._cov_A: dict[str, Any] = {}   # name -> (k_in,  k_in ) Sum Qa Qa^T
        self._fac_layout: list = []        # [(name, k_out, k_in)] in sketch-concat order
        # Grouped batched dispatch (opt-in, off by default -- see
        # _factored_sketch_grouped for the full rationale). Modules sharing an
        # identical (d_out, d_in) shape, the common case across LoRA layers,
        # are projected and covariance-accumulated with ONE batched einsum/bmm
        # per shape group instead of one per module, cutting kernel-launch
        # count roughly by the group size. Mathematically identical to the
        # per-module loop (same P/Q, same per-module sums); only the number
        # of kernel launches changes, not the arithmetic.
        self.grouped_dispatch = grouped_dispatch
        self._fac_groups_built = False
        self._fac_groups: dict = {}        # (d_out,d_in,k_out,k_in) -> [name, ...]
        self._fac_P_stack: dict = {}       # group key -> (M,k_out,d_out)
        self._fac_Q_stack: dict = {}       # group key -> (M,k_in,d_in)
        self._cov_G_stack: dict = {}       # group key -> (M,k_out,k_out)
        self._cov_A_stack: dict = {}       # group key -> (M,k_in,k_in)
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
        true gradient dot product (what attribution needs).

        Dispatches to :meth:`_factored_sketch_grouped` when
        ``grouped_dispatch=True`` (opt-in; batches modules of identical shape
        into one einsum/bmm call per shape group instead of one per module —
        see that method's docstring), else the per-module loop below."""
        if self.grouped_dispatch:
            return self._factored_sketch_grouped()
        return self._factored_sketch_loop()

    def _factored_sketch_loop(self):
        """Per-module reference implementation of :meth:`_factored_sketch`.
        One einsum/addmm_ per tracked module per step. Simple and always
        correct; :meth:`_factored_sketch_grouped` is a batched, opt-in,
        numerically-equivalent alternative for reducing kernel-launch count
        when many modules share a shape (the common LoRA case)."""
        import torch

        parts = []
        layout = []
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
                if self.inline_precond:
                    # Preallocate the fp32 covariance accumulators ONCE, here in the
                    # setup path -- so the per-step hot path below is branch-free and
                    # allocation-free (just two in-place addmm_).
                    self._cov_G[name] = torch.zeros(k_out, k_out, device=a.device,
                                                    dtype=torch.float32)
                    self._cov_A[name] = torch.zeros(k_in, k_in, device=a.device,
                                                    dtype=torch.float32)
            P, Q = PQ
            with torch.no_grad():
                Pg = torch.einsum("bto,ko->btk", g.float(), P)   # (B,T,k_out)
                Qa = torch.einsum("bti,li->btl", a.float(), Q)   # (B,T,k_in)
                S = torch.einsum("btk,btl->bkl", Pg, Qa)          # (B,k_out,k_in)
                if self.inline_precond:
                    # Covariance of the PROJECTED factors over all tokens this batch:
                    # cov_G += (P g)^T (P g), cov_A += (Q a)^T (Q a). In-place fused
                    # addmm_ into preallocated buffers -- no per-step allocation, no
                    # dict-membership branch. Reuses Pg/Qa already computed above, so
                    # the marginal cost is two small (k x BT)@(BT x k) matmuls; no
                    # extra pass over the data.
                    pg2 = Pg.reshape(-1, k_out)  # (B*T, k_out)
                    qa2 = Qa.reshape(-1, k_in)   # (B*T, k_in)
                    self._cov_G[name].addmm_(pg2.transpose(0, 1), pg2)
                    self._cov_A[name].addmm_(qa2.transpose(0, 1), qa2)
            parts.append(S.reshape(S.shape[0], -1))
            layout.append((name, int(k_out), int(k_in)))

        if not parts:
            return None
        self.grad_dim = grad_dim_total
        self._fac_layout = layout
        sketch = torch.cat(parts, dim=1)
        self.sketch_dim = sketch.shape[1]
        return sketch

    def _build_groups(self):
        """One-time (per logger instance) construction of shape-based groups
        from the now-fully-populated ``self._fac_PQ``. Called once, after the
        first step's per-module P/Q have all been created by the loop-path
        fallback in :meth:`_factored_sketch_grouped` — so group membership is
        derived from real per-module shapes, not guessed in advance, and any
        covariance accumulated during that first fallback step is carried
        forward into the new stacked buffers rather than lost.

        Modules are grouped by ``(d_out, d_in, k_out, k_in)`` (the k's are a
        deterministic function of d_out/d_in and kfac, included in the key
        only for clarity). Within a group, modules keep their original
        ``self._names`` relative order, so ``kfac_covariances()`` and the
        final sketch concatenation can always recover the exact per-module
        layout the ungrouped loop path would have produced.
        """
        import torch

        groups: dict = {}
        for name in self._names:
            if name not in self._fac_PQ:
                continue
            P, Q = self._fac_PQ[name]
            k_out, d_out = P.shape
            k_in, d_in = Q.shape
            key = (d_out, d_in, k_out, k_in)
            groups.setdefault(key, []).append(name)
        self._fac_groups = groups

        for key, names in groups.items():
            k_out, d_out, k_in, d_in = key[2], key[0], key[3], key[1]
            Ps = torch.stack([self._fac_PQ[n][0] for n in names], dim=0)  # (M,k_out,d_out)
            Qs = torch.stack([self._fac_PQ[n][1] for n in names], dim=0)  # (M,k_in,d_in)
            self._fac_P_stack[key] = Ps
            self._fac_Q_stack[key] = Qs
            if self.inline_precond:
                device = Ps.device
                cov_G0 = torch.stack(
                    [self._cov_G.get(n, torch.zeros(k_out, k_out, device=device,
                                                      dtype=torch.float32))
                     for n in names], dim=0)
                cov_A0 = torch.stack(
                    [self._cov_A.get(n, torch.zeros(k_in, k_in, device=device,
                                                      dtype=torch.float32))
                     for n in names], dim=0)
                self._cov_G_stack[key] = cov_G0
                self._cov_A_stack[key] = cov_A0
        self._fac_groups_built = True

    def _factored_sketch_grouped(self):
        """Batched variant of :meth:`_factored_sketch_loop`: modules sharing
        an identical ``(d_out, d_in)`` shape — the common case for LoRA,
        where e.g. every ``lora_A`` across transformer layers has the same
        shape and every ``lora_B`` has another — are projected and
        covariance-accumulated with ONE batched einsum/bmm call per shape
        group instead of one per module, cutting kernel-launch count roughly
        by the group size. Batching over an independent module axis does not
        change any individual module's arithmetic (same per-module P/Q, same
        per-module sums over tokens), only how many kernel launches it costs
        — see ``tests/unit/test_lora_logging.py::test_grouped_dispatch_matches_loop``
        for the numerical-equivalence check this claim rests on.

        Falls back to the exact per-module loop on the first step, while
        per-module P/Q are still being lazily created and group membership is
        not yet known; the resulting per-module covariance from that
        fallback step is carried forward into the stacked buffers
        (:meth:`_build_groups`), not discarded. Group membership is then
        assumed fixed for the life of the logger — every tracked module must
        fire every step after that, matching how LoRA hooks actually behave
        in practice; a step where that assumption is violated raises rather
        than silently producing a mismatched layout.
        """
        import torch

        present = {name for name in self._names
                   if name in self._fwd_input and name in self._grad_output}
        if not present:
            return None

        if not self._fac_groups_built:
            sketch = self._factored_sketch_loop()
            if set(self._fac_PQ) >= present:
                self._build_groups()
            return sketch

        known = {n for names in self._fac_groups.values() for n in names}
        if present != known:
            raise RuntimeError(
                "grouped_dispatch=True requires the same set of tracked modules "
                "to fire every step after the first (group membership is fixed "
                f"once built); got {sorted(present)}, expected {sorted(known)}."
            )

        grad_dim_total = 0
        parts_by_name: dict = {}
        for key, names in self._fac_groups.items():
            d_out, d_in, k_out, k_in = key
            grad_dim_total += d_out * d_in * len(names)
            a_list, g_list = [], []
            for n in names:
                a = self._fwd_input[n]
                g = self._grad_output[n]
                if a.dim() == 2:
                    a = a.unsqueeze(1)
                    g = g.unsqueeze(1)
                a_list.append(a)
                g_list.append(g)
            a_stack = torch.stack(a_list, dim=0)  # (M,B,T,d_in)
            g_stack = torch.stack(g_list, dim=0)  # (M,B,T,d_out)
            P_stack = self._fac_P_stack[key]      # (M,k_out,d_out)
            Q_stack = self._fac_Q_stack[key]      # (M,k_in,d_in)
            with torch.no_grad():
                Pg = torch.einsum("mbto,mko->mbtk", g_stack.float(), P_stack)  # (M,B,T,k_out)
                Qa = torch.einsum("mbti,mli->mbtl", a_stack.float(), Q_stack)  # (M,B,T,k_in)
                S = torch.einsum("mbtk,mbtl->mbkl", Pg, Qa)                     # (M,B,k_out,k_in)
                if self.inline_precond:
                    M, B, T, _ = Pg.shape
                    pg2 = Pg.reshape(M, B * T, k_out)
                    qa2 = Qa.reshape(M, B * T, k_in)
                    self._cov_G_stack[key] += torch.bmm(pg2.transpose(1, 2), pg2)
                    self._cov_A_stack[key] += torch.bmm(qa2.transpose(1, 2), qa2)
            for i, name in enumerate(names):
                parts_by_name[name] = S[i].reshape(S.shape[1], -1)

        self.grad_dim = grad_dim_total
        parts = [parts_by_name[n] for n in self._names if n in parts_by_name]
        layout = []
        for key, names in self._fac_groups.items():
            for n in names:
                if n in parts_by_name:
                    layout.append((n, int(key[2]), int(key[3])))
        # layout must follow self._names order, not group order, to match parts
        layout_by_name = {n: (n, ko, ki) for n, ko, ki in layout}
        layout = [layout_by_name[n] for n in self._names if n in parts_by_name]
        self._fac_layout = layout
        sketch = torch.cat(parts, dim=1)
        self.sketch_dim = sketch.shape[1]
        return sketch

    def kfac_covariances(self):
        """Inline-accumulated K-FAC covariance of the projected factors.

        Returns ``(layout, cov_G, cov_A)`` where ``layout`` is the per-module
        ``[(name, k_out, k_in), ...]`` block order matching the concatenated
        factored sketch, and ``cov_G[name]`` / ``cov_A[name]`` are the
        ``(k_out, k_out)`` / ``(k_in, k_in)`` covariance matrices ``Sum Pg Pg^T``
        / ``Sum Qa Qa^T`` accumulated over the logging pass (numpy, float64).

        Only populated when the logger was built with ``inline_precond=True``.
        Feed straight into :func:`apply_kfac_precondition`.

        When ``grouped_dispatch=True`` and groups have been built, this reads
        the per-module slices straight out of the stacked accumulators
        (``self._cov_G_stack`` / ``self._cov_A_stack``) rather than
        ``self._cov_G`` / ``self._cov_A``, which stop being updated once
        grouped accumulation takes over (see :meth:`_factored_sketch_grouped`)."""
        import numpy as np

        if self.grouped_dispatch and self._fac_groups_built:
            for key, names in self._fac_groups.items():
                for i, name in enumerate(names):
                    self._cov_G[name] = self._cov_G_stack[key][i]
                    self._cov_A[name] = self._cov_A_stack[key][i]

        cov_G = {k: v.detach().cpu().numpy().astype(np.float64)
                 for k, v in self._cov_G.items()}
        cov_A = {k: v.detach().cpu().numpy().astype(np.float64)
                 for k, v in self._cov_A.items()}
        return list(self._fac_layout), cov_G, cov_A

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


def _inv_sqrt_damped(M: np.ndarray, damping: float) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix with relative damping.

    Damping is scaled by the mean eigenvalue (``trace(M)/k``) so the correction
    is invariant to the arbitrary overall scale of the accumulated covariance
    (which depends on token count and loss reduction). Returns ``W`` such that
    ``W @ W ≈ (M + λ I)^{-1}`` with ``λ = damping · trace(M)/k``."""
    M = 0.5 * (M + M.T)
    k = M.shape[0]
    lam = damping * (np.trace(M) / max(k, 1))
    w, V = np.linalg.eigh(M + lam * np.eye(k))
    w = np.clip(w, a_min=1e-12, a_max=None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def apply_kfac_precondition(sketch, layout, cov_G, cov_A, damping):
    """K-FAC-whiten a stored factored sketch matrix at query time.

    Each per-module block ``S`` (shape ``(k_out, k_in)``) of every row is
    replaced by ``W_G S W_A`` where ``W_G = (Ĝ + λ_G I)^{-1/2}`` and
    ``W_A = (Â + λ_A I)^{-1/2}`` are built from the inline-accumulated
    sketch-space covariances ``Ĝ = cov_G[name]`` and ``Â = cov_A[name]``.

    The Frobenius inner product of two whitened sketches then equals
    ``tr(S_jᵀ Ĝ⁻¹ S_i Â⁻¹)`` -- the Kronecker-factored (K-FAC) preconditioned
    influence, computed entirely in the low-dimensional sketch space. The SAME
    whitening (built from the training covariance) must be applied to both the
    train and the query/test sketches before taking dot products.

    Parameters
    ----------
    sketch : ndarray, shape (N, sketch_dim)
        Stored factored sketches, per-module blocks concatenated in ``layout``
        order (exactly what ``GradientStore.get_projected_matrix()`` returns for
        a factored logger).
    layout : list[(name, k_out, k_in)]
        Per-module block layout, from :meth:`LoRAGradientLogger.kfac_covariances`.
    cov_G, cov_A : dict[str, ndarray]
        Sketch-space covariances keyed by module name (same source).
    damping : float
        Relative damping strength (see :func:`_inv_sqrt_damped`). Choose it on a
        held-out split, never on the reported test set.

    Returns
    -------
    ndarray, shape (N, sketch_dim)
        A new whitened sketch matrix; the input is not modified.
    """
    sketch = np.asarray(sketch)
    N = sketch.shape[0]
    out = np.empty_like(sketch)
    off = 0
    for name, k_out, k_in in layout:
        sz = k_out * k_in
        block = sketch[:, off:off + sz].reshape(N, k_out, k_in)
        WG = _inv_sqrt_damped(cov_G[name], damping)   # (k_out, k_out)
        WA = _inv_sqrt_damped(cov_A[name], damping)   # (k_in,  k_in )
        white = np.einsum("op,npj,jq->noq", WG, block, WA)
        out[:, off:off + sz] = white.reshape(N, sz)
        off += sz
    if off != sketch.shape[1]:
        raise ValueError(
            f"layout blocks sum to {off} columns but sketch has {sketch.shape[1]}; "
            f"layout does not match this sketch matrix."
        )
    return out
