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
