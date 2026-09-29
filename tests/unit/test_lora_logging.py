"""Correctness of inline per-sample LoRA gradient capture.

The MLSys systems claim rests on one identity: the per-sample gradient of a
linear layer captured from forward/backward hooks during a *single* batched
backward equals the gradient you would get by backpropagating each sample
individually. If that fails, every downstream attribution number is wrong.

We assert exact directional agreement (cosine = 1) between the hook path and
per-sample autograd. A global scale (the loss-reduction convention) is allowed
because it cancels in dot-product attribution ranking.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from traceprop.attribution.gradient_store import GradientStore
from traceprop.llm import LoRAGradientLogger, select_lora_linears


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4, alpha=8):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.normal_(self.lora_B.weight, std=0.02)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


class ToyNet(nn.Module):
    def __init__(self, d=32, vocab=50, r=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.l1 = LoRALinear(d, d, r)
        self.l2 = LoRALinear(d, vocab, r)

    def forward(self, idx):
        h = torch.relu(self.l1(self.emb(idx)))
        return self.l2(h)


def test_per_sample_grads_match_autograd():
    torch.manual_seed(0)
    model = ToyNet()
    B, T = 6, 5
    x = torch.randint(0, 50, (B, T))

    targets = select_lora_linears(model, ("lora_A", "lora_B"))
    assert len(targets) == 4  # 2 LoRA linears x {A, B}

    # ground truth: individual backward per sample
    truth = []
    for i in range(B):
        model.zero_grad()
        li = F.cross_entropy(
            model(x[i:i+1]).reshape(-1, 50), x[i:i+1].reshape(-1)
        )
        li.backward()
        truth.append(torch.cat([m.weight.grad.reshape(-1).clone() for _, m in targets]))
    truth = torch.stack(truth).numpy()

    # hook path: one batched backward (sum reduction keeps per-sample signal)
    store = GradientStore(proj_dim=8)
    lg = LoRAGradientLogger(store, targets, proj_dim=8)
    model.zero_grad()
    loss = F.cross_entropy(model(x).reshape(-1, 50), x.reshape(-1), reduction="sum")
    loss.backward()
    hook = lg._per_sample_grads().numpy()

    assert hook.shape == truth.shape
    for i in range(B):
        cos = (truth[i] @ hook[i]) / (
            np.linalg.norm(truth[i]) * np.linalg.norm(hook[i]) + 1e-12
        )
        assert cos > 0.9999, f"sample {i} cosine {cos}"


def test_flush_populates_store_and_projects_on_device():
    torch.manual_seed(1)
    model = ToyNet()
    x = torch.randint(0, 50, (4, 5))
    targets = select_lora_linears(model, ("lora_A", "lora_B"))
    store = GradientStore(proj_dim=8)
    lg = LoRAGradientLogger(store, targets, proj_dim=8)

    loss = F.cross_entropy(model(x).reshape(-1, 50), x.reshape(-1))
    loss.backward()
    n = lg.flush_step(sample_indices=[10, 11, 12, 13])
    lg.detach()

    assert n == 4
    assert len(store) == 4
    mat = store.get_projected_matrix()
    assert mat.shape == (4, 8)
    # explicit sample indices preserved
    idxs = sorted(e.sample_index for e in store._entries.values())
    assert idxs == [10, 11, 12, 13]


# --- Inline K-FAC preconditioning ------------------------------------------
# The systems claim for preconditioning: Traceprop accumulates the K-FAC
# covariance of the *projected* factors DURING the factored logging pass (no
# second covariance pass), then applies a damped whitening at query time. Two
# things must hold: (1) the inline-accumulated covariance equals a separate
# covariance pass to numerical tolerance, and (2) whitening two sketches and
# taking their Frobenius inner product reproduces the K-FAC preconditioned
# influence tr(Sj^T G^{-1} Si A^{-1}).

def test_inline_covariance_matches_separate_pass():
    from traceprop.llm.lora_logging import apply_kfac_precondition  # noqa: F401

    torch.manual_seed(3)
    model = ToyNet()
    B, T, kfac = 4, 5, 4
    batches = [torch.randint(0, 50, (B, T)) for _ in range(3)]

    targets = select_lora_linears(model, ("lora_A", "lora_B"))
    store = GradientStore(proj_dim=8)
    lg = LoRAGradientLogger(store, targets, proj_dim=8, factored=True,
                            kfac=kfac, inline_precond=True)
    for xb in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).reshape(-1, 50), xb.reshape(-1),
                        reduction="sum").backward()
        lg.flush_step()
    layout, cov_G, cov_A = lg.kfac_covariances()
    PQ = dict(lg._fac_PQ)  # per-module (P, Q) sparse-JL maps actually used
    lg.detach()

    assert layout, "no layout captured"
    assert set(cov_G) == set(name for name, _, _ in layout)

    # Independent path: accumulate the FULL-space covariance Sum g g^T and
    # Sum a a^T with fresh hooks, then project with the SAME P, Q. Algebraically
    # P (Sum g g^T) P^T == Sum (P g)(P g)^T, so a match validates the inline
    # accumulation without reusing the logger's accumulator at all.
    cap: dict = {}
    handles = []
    tmap = dict(targets)
    for name, module in targets:
        def fwd(_m, inp, _out, nm=name):
            cap.setdefault(nm, {})["a"] = inp[0].detach()
        def bwd(_m, _gi, go, nm=name):
            cap.setdefault(nm, {})["g"] = go[0].detach()
        handles.append(module.register_forward_hook(fwd))
        handles.append(module.register_full_backward_hook(bwd))

    Cg_full: dict = {}
    Ca_full: dict = {}
    for xb in batches:
        cap.clear()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).reshape(-1, 50), xb.reshape(-1),
                        reduction="sum").backward()
        for name in tmap:
            a = cap[name]["a"].reshape(-1, cap[name]["a"].shape[-1]).double()
            g = cap[name]["g"].reshape(-1, cap[name]["g"].shape[-1]).double()
            Cg_full[name] = Cg_full.get(name, 0.0) + (g.T @ g).numpy()
            Ca_full[name] = Ca_full.get(name, 0.0) + (a.T @ a).numpy()
    for h in handles:
        h.remove()

    for name, _, _ in layout:
        P, Q = (t.double().cpu().numpy() for t in PQ[name])
        exp_G = P @ Cg_full[name] @ P.T
        exp_A = Q @ Ca_full[name] @ Q.T
        assert np.allclose(cov_G[name], exp_G, rtol=1e-6, atol=1e-8), \
            f"cov_G[{name}] mismatch, max abs {np.abs(cov_G[name]-exp_G).max()}"
        assert np.allclose(cov_A[name], exp_A, rtol=1e-6, atol=1e-8), \
            f"cov_A[{name}] mismatch, max abs {np.abs(cov_A[name]-exp_A).max()}"


def test_apply_kfac_precondition_reproduces_preconditioned_influence():
    from traceprop.llm.lora_logging import apply_kfac_precondition, _inv_sqrt_damped

    rng = np.random.default_rng(7)
    k_out, k_in, N = 4, 3, 6
    layout = [("m", k_out, k_in)]
    # random PSD covariances
    Bg = rng.standard_normal((k_out, k_out + 2))
    Ba = rng.standard_normal((k_in, k_in + 2))
    covG = Bg @ Bg.T
    covA = Ba @ Ba.T
    sketch = rng.standard_normal((N, k_out * k_in))
    damping = 0.1

    white = apply_kfac_precondition(sketch, layout, {"m": covG}, {"m": covA}, damping)
    assert white.shape == sketch.shape

    WG = _inv_sqrt_damped(covG, damping)
    WA = _inv_sqrt_damped(covA, damping)
    # inv-sqrt squared is the damped inverse
    lamG = damping * np.trace(covG) / k_out
    assert np.allclose(WG @ WG, np.linalg.inv(covG + lamG * np.eye(k_out)), atol=1e-10)

    # block whitening: W_G S W_A
    for n in range(N):
        S = sketch[n].reshape(k_out, k_in)
        assert np.allclose(white[n].reshape(k_out, k_in), WG @ S @ WA, atol=1e-10)

    # whitened Frobenius inner product == tr(Sj^T G^{-1} Si A^{-1})
    Ginv = np.linalg.inv(covG + lamG * np.eye(k_out))
    lamA = damping * np.trace(covA) / k_in
    Ainv = np.linalg.inv(covA + lamA * np.eye(k_in))
    for i in range(N):
        for j in range(N):
            Si = sketch[i].reshape(k_out, k_in)
            Sj = sketch[j].reshape(k_out, k_in)
            lhs = float(white[i] @ white[j])
            rhs = float(np.trace(Sj.T @ Ginv @ Si @ Ainv))
            assert abs(lhs - rhs) < 1e-8, f"({i},{j}) {lhs} vs {rhs}"


# --- Per-module LogIX damping monkeypatch: reimplementation fidelity ----------
# The fairness comparison monkeypatches logix's precondition_kfac to use per-module
# relative damping. Reviewers could object that the win comes from our reimpl, not
# the damping change. This test proves the rotation/division math is IDENTICAL to
# logix's own: given the equivalent absolute damping per module, our per-module
# function reproduces logix's original precondition_kfac output exactly.

def test_permodule_precondition_matches_logix_original():
    pytest.importorskip("logix")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    from logix.analysis.influence_function_utils import precondition_kfac as logix_orig
    from logix_strict import precondition_kfac_permodule

    class FakeState:
        def __init__(self, ev, evec):
            self._ev, self._evec = ev, evec
        def get_covariance_svd_state(self):
            return self._ev, self._evec

    torch.manual_seed(0)
    # three modules with DIFFERENT k dims and different eigenvalue scales, so a
    # single global damping could never fake per-module agreement
    specs = {"m0": (8, 8, 1.0), "m1": (6, 8, 10.0), "m2": (8, 5, 0.1)}
    ev, evec, src = {}, {}, {}
    for name, (k_out, k_in, scale) in specs.items():
        fwd_val = torch.rand(k_in) * scale + 0.01
        bwd_val = torch.rand(k_out) * scale + 0.01
        fwd_vec = torch.linalg.qr(torch.randn(k_in, k_in))[0]
        bwd_vec = torch.linalg.qr(torch.randn(k_out, k_out))[0]
        ev[name] = {"forward": fwd_val, "backward": bwd_val}
        evec[name] = {"forward": fwd_vec, "backward": bwd_vec}
        src[name] = {"grad": torch.randn(4, k_out, k_in)}
    state = FakeState(ev, evec)

    lam_rel = 0.037
    mine = precondition_kfac_permodule(src, state, damping=lam_rel)
    for name, (k_out, k_in, _) in specs.items():
        full_eigval = torch.outer(ev[name]["backward"], ev[name]["forward"])
        abs_d = float(lam_rel * full_eigval.mean())
        # logix original applies a single scalar damping; call it per module with
        # exactly that module's equivalent absolute damping
        orig = logix_orig({name: src[name]}, state, damping=abs_d)
        assert torch.allclose(mine[name]["grad"], orig[name]["grad"], atol=1e-6), \
            f"per-module patch diverges from logix original on {name}"


# --- Grouped batched dispatch -----------------------------------------------
# The 7B overhead investigation (see docs/mlsys/main.tex Appendix "7B Overhead
# Investigation Detail") found flush_step CPU-bound at scale (many small
# per-module einsum/copy/cat calls). grouped_dispatch=True batches modules
# sharing an identical (d_out, d_in) shape -- e.g. every lora_A across
# transformer layers -- into one einsum/bmm call per shape group. Batching
# over an independent module axis must not change any individual module's
# arithmetic, only how many kernel launches it costs; this test is the
# numerical-equivalence check that claim rests on, run against ToyNet (whose
# l1.lora_A and l2.lora_A share shape (32,4), giving a real group of size 2,
# not just size-1 groups that would trivially pass).

def test_grouped_dispatch_matches_loop():
    torch.manual_seed(5)
    B, T, kfac = 4, 6, 4
    batches = [torch.randint(0, 50, (B, T)) for _ in range(4)]
    targets_names = None

    def run(grouped):
        torch.manual_seed(0)  # same model init for both runs
        model = ToyNet()
        targets = select_lora_linears(model, ("lora_A", "lora_B"))
        store = GradientStore(proj_dim=8)
        lg = LoRAGradientLogger(store, targets, proj_dim=8, factored=True,
                                kfac=kfac, inline_precond=True,
                                grouped_dispatch=grouped)
        sketches = []
        for xb in batches:
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb).reshape(-1, 50), xb.reshape(-1),
                            reduction="sum").backward()
            proj = lg._factored_sketch()
            sketches.append(proj.clone())
        layout, cov_G, cov_A = lg.kfac_covariances()
        lg.detach()
        return sketches, dict(layout=[(n, ko, ki) for n, ko, ki in layout]), cov_G, cov_A

    loop_sketches, loop_meta, loop_cov_G, loop_cov_A = run(grouped=False)
    grp_sketches, grp_meta, grp_cov_G, grp_cov_A = run(grouped=True)

    # a real group of size > 1 must actually have formed (l1.lora_A / l2.lora_A
    # share shape (in=32, out=4)) -- otherwise this test would trivially pass
    # without exercising the batched path at all.
    assert loop_meta["layout"] == grp_meta["layout"], "layout order must match exactly"

    for s_loop, s_grp in zip(loop_sketches, grp_sketches):
        assert torch.allclose(s_loop, s_grp, atol=1e-6, rtol=1e-6), \
            f"sketch mismatch: max abs diff {(s_loop - s_grp).abs().max()}"

    assert set(loop_cov_G) == set(grp_cov_G)
    for name in loop_cov_G:
        assert np.allclose(loop_cov_G[name], grp_cov_G[name], atol=1e-6, rtol=1e-6), \
            f"cov_G[{name}] mismatch: max abs diff " \
            f"{np.abs(loop_cov_G[name] - grp_cov_G[name]).max()}"
        assert np.allclose(loop_cov_A[name], grp_cov_A[name], atol=1e-6, rtol=1e-6), \
            f"cov_A[{name}] mismatch: max abs diff " \
            f"{np.abs(loop_cov_A[name] - grp_cov_A[name]).max()}"


def test_grouped_dispatch_forms_a_real_multi_module_group():
    """Guard against the equivalence test above passing vacuously (all groups
    size 1, batched path never actually exercised). ToyNet's l1.lora_A and
    l2.lora_A both have shape (in_features=32, r=4), so they must land in the
    same group once grouped_dispatch builds its shape map."""
    torch.manual_seed(0)
    model = ToyNet()
    targets = select_lora_linears(model, ("lora_A", "lora_B"))
    store = GradientStore(proj_dim=8)
    lg = LoRAGradientLogger(store, targets, proj_dim=8, factored=True,
                            kfac=4, inline_precond=True, grouped_dispatch=True)
    for _ in range(2):  # first step builds groups from the fallback loop path
        xb = torch.randint(0, 50, (4, 6))
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).reshape(-1, 50), xb.reshape(-1),
                        reduction="sum").backward()
        lg._factored_sketch()
    lg.detach()
    assert lg._fac_groups_built
    group_sizes = sorted(len(v) for v in lg._fac_groups.values())
    assert max(group_sizes) >= 2, (
        f"expected at least one shape group with >1 module (l1.lora_A/l2.lora_A "
        f"share shape), got group sizes {group_sizes} -- the equivalence test "
        f"would be vacuous if this fails")
