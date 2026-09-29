"""Equivalence check: LogIX one-pass (covariance gathered DURING grad logging)
== two-pass (separate covariance pass then logging), at a FIXED checkpoint, with
RANDOM init add_lora. If the preconditioned influence scores match, then LogIX
with random init can be single-pass at no quality cost -- so "single pass" is a
trade-off (random=1 pass/lower quality; PCA=2 passes/best quality; Traceprop=1
pass/best quality), NOT a structural exclusive of Traceprop.

Tiny backend, CPU. Run: python exp35_logix_onepass_equiv.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logix

from exp27_lds_quality import build_tiny_classifier, synthetic_data
from logix_strict import install_strict_warnings, patch_loralinear_weight_proxy


def main():
    install_strict_warnings(); patch_loralinear_weight_proxy()
    dev = "cpu"; seq, vocab, r = 32, 50, 8
    Xtr, ytr = synthetic_data(200, seq, vocab, 0)
    Xte, yte = synthetic_data(40, seq, vocab, 1)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xte_t, yte_t = torch.tensor(Xte), torch.tensor(yte)

    # one fixed trained checkpoint, reused (deep-copied) for both LogIX runs
    torch.manual_seed(1234); np.random.seed(1234)
    base = build_tiny_classifier(vocab, seq=seq, r=r, n_blocks=2)
    opt = torch.optim.Adam([p for p in base.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(2):
        for s in range(0, len(Xtr_t), 16):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(base(Xtr_t[s:s+16]), ytr_t[s:s+16]).backward()
            opt.step()
    base.eval()

    import copy

    def logix_scores(one_pass):
        import os
        model = copy.deepcopy(base)
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        def restore():
            for p in model.parameters():
                if id(p) in trainable_ids:
                    p.requires_grad = True
        tracked = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and (n == "score" or "lora_A" in n or "lora_B" in n)]
        run_ = logix.LogIX(project=f"equiv_{one_pass}_{os.getpid()}", config="exp31_config.yaml")
        run_.config.lora.init = "random"
        run_.watch(model, name_filter=tracked, type_filter=[nn.Linear])
        restore()

        ids = {"n": 0}
        def dids(bs):
            out = [str(ids["n"] + i) for i in range(bs)]; ids["n"] += bs; return out

        def pass_over(train=True):
            X, y = (Xtr_t, ytr_t) if train else (Xte_t, yte_t)
            ids["n"] = 0 if train else 10 ** 9
            logs = []
            for s in range(0, len(X), 16):
                with run_(data_id=dids(len(X[s:s+16]))):
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(X[s:s+16]), y[s:s+16], reduction="sum").backward()
                if not train:
                    logs.append(run_.get_log(copy=True))
            return logs

        torch.manual_seed(777); np.random.seed(777)  # SAME random projection in both runs
        run_.add_lora()   # random init -> needs NO prior covariance pass
        restore()
        if one_pass:
            # covariance + grad logging TOGETHER in a single pass
            run_.setup({"forward": ["covariance"], "backward": ["covariance"], "grad": ["log"]})
            run_.save(True)
            pass_over(train=True)
            run_.finalize()
        else:
            # separate covariance pass, THEN logging pass
            run_.setup({"forward": ["covariance"], "backward": ["covariance"]})
            pass_over(train=True)
            run_.finalize()
            run_.setup({"grad": ["log"]})
            run_.save(True)
            pass_over(train=True)
            run_.finalize()

        # grab the raw covariance state (pre-SVD) for a direct one-pass vs two-pass compare
        cov_state = run_.state.get_covariance_state()
        cov_np = {}
        for mod, d in cov_state.items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if hasattr(v, "detach"):
                        cov_np[f"{mod}::{k}"] = v.detach().cpu().numpy().astype(np.float64)
            elif hasattr(d, "detach"):
                cov_np[mod] = d.detach().cpu().numpy().astype(np.float64)

        loader = run_.build_log_dataloader(batch_size=16, flatten=False)
        run_.eval()
        def score(precondition, damping):
            rows = []
            for data_id, log in pass_over(train=False):
                res = run_.compute_influence_all(src_log=(data_id, log), loader=loader,
                                                 mode="dot", precondition=precondition,
                                                 hessian="kfac", damping=damping)
                rows.append(res["influence"].numpy())
            return np.concatenate(rows, axis=0)
        # per-module RELATIVE damping (0.01, 0.1) via the vetted monkeypatch -> realistic,
        # well-conditioned (not the near-singular 1e-6 that amplifies summation-order noise)
        import logix.analysis.influence_function as _lif
        from logix_strict import precondition_kfac_permodule
        _lif.precondition_kfac = precondition_kfac_permodule
        return {"cov": cov_np,
                "dot": score(False, None),
                "precond_rel_0.01": score(True, 0.01),
                "precond_rel_0.1": score(True, 0.1)}

    two = logix_scores(one_pass=False)
    one = logix_scores(one_pass=True)

    # (1) covariance states, per module
    print("=== covariance state: one-pass vs two-pass (seeded projection) ===")
    worst = 0.0; ratios = []
    for k in sorted(one["cov"]):
        a, b = one["cov"][k], two["cov"][k]
        denom = np.abs(b).max() + 1e-30
        worst = max(worst, float(np.abs(a - b).max() / denom))
        m = np.abs(b) > denom * 1e-3
        if m.any():
            ratios.append(float(np.median(a[m] / b[m])))
    ratios = np.array(ratios)
    print(f"  {len(one['cov'])} covariance tensors; worst max-relative-diff = {worst:.3e}  "
          f"allclose(rtol=1e-5)={all(np.allclose(one['cov'][k], two['cov'][k], rtol=1e-5, atol=1e-8) for k in one['cov'])}")
    print(f"  per-tensor one/two ratio: median={np.median(ratios):.4f} min={ratios.min():.4f} "
          f"max={ratios.max():.4f}  (constant ratio -> normalization/count diff; varied -> real diff)")

    # (2) scores at realistic per-module relative damping
    for key in ("dot", "precond_rel_0.01", "precond_rel_0.1"):
        a, b = one[key], two[key]
        ok = np.allclose(a, b, rtol=1e-4, atol=1e-6)
        md = float(np.abs(a - b).max()); cr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print(f"[{key:<16}] one==two: allclose={ok}  max_abs_diff={md:.3e}  corr={cr:.6f}")
    print("\nVERDICT: if covariance worst-rel-diff ~float-eps AND precond_rel_* agree -> "
          "one-pass == two-pass (the earlier diff was the near-singular 1e-6 damping); state "
          "'one-pass ≡ two-pass' and fairrand IS the single-pass LogIX-random quality.")


if __name__ == "__main__":
    main()
