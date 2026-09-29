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
        return {"dot": score(False, None),                 # no covariance -> tests logged-grad equiv
                "precond_default": score(True, None),      # default damping (has module-0 leak bug)
                "precond_fixed": score(True, 1e-6)}        # explicit damping -> removes leak bug

    two = logix_scores(one_pass=False)
    one = logix_scores(one_pass=True)
    for key in ("dot", "precond_default", "precond_fixed"):
        a, b = one[key], two[key]
        ok = np.allclose(a, b, rtol=1e-4, atol=1e-6)
        md = float(np.abs(a - b).max())
        cr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print(f"[{key:<16}] one==two: allclose={ok}  max_abs_diff={md:.3e}  corr={cr:.6f}")
    print("\nINTERPRETATION: dot identical -> logged grads match (one-pass logging is correct); "
          "precond_fixed identical -> covariance matches too (difference was only the default-damping "
          "leak bug); precond_default differing while precond_fixed matches confirms it's the bug, "
          "not a real one-pass/two-pass covariance difference.")


if __name__ == "__main__":
    main()
