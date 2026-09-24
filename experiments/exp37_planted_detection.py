"""exp37 -- Planted-example detection via inline attribution.

Stage 2, part 2 (alongside the scope sweep in exp31/exp35).

PRIMARY experiment: backdoor detection. Plants a trigger into K training
examples with their label forced to a fixed target class, trains with
Traceprop's inline capture (TracIn-style trajectory-sum gradients, matching
exp29's methodology), then for TEST inputs that ALSO contain the trigger,
attributes the (backdoored) prediction back to the training set and checks
whether the planted examples rank at the top (AUC, precision@k).

Hardened against four objections a reviewer would raise on the first
version of this experiment:
  1. Token matching / representation similarity alone can't explain a good
     score: K "distractor" examples ALSO get the trigger inserted but KEEP
     their correct label. A model that memorized "trigger present" the
     naive way can't tell poisoned from distractor; attribution should,
     because only poisoned examples pushed the model toward the target
     label. A representation-similarity baseline (cosine similarity of
     hidden states feeding the classifier head) is reported alongside
     attribution for exactly this comparison.
  2. The "backdoor learned" claim needs a control: target-label rate on
     CLEAN (untriggered) held-out inputs is reported next to the triggered
     rate. Without this, "94% of triggered inputs predict the target
     label" is meaningless if the model predicts that label 90% of the
     time regardless of input.
  3. Trigger insertion is padding-aware: naively overwriting the last
     array positions can silently land IN padding for short sequences
     (confirmed: GPT-2's tokenizer right-pads by default, and
     GPT2ForSequenceClassification's own pooling finds the first
     pad-token position to locate the true sequence end -- a trigger
     placed after real content but flagged as "padding" by that search
     would never be seen by the model's own classification pooling).
     Trigger is planted at the true end of REAL content (right before the
     first pad token), not the literal last array index.
  4. Mislabeled examples (planted for the secondary experiment) have large
     gradients and can dominate a raw dot-product ranking regardless of
     what the model actually learned about the backdoor. Backdoor AUC is
     reported three ways -- raw dot, cosine-normalized, and TRAK-corrected
     -- and separately with the mislabeled set excluded from the candidate
     pool, so a reviewer can see the ranking isn't an artifact of gradient
     magnitude.
  5. Every metric is reported as mean +/- std over --n_seeds independent
     runs (default 5), not a single-seed point estimate.

SECONDARY experiment: mislabel detection via self-influence, reported next
to loss-ranking and gradient-norm baselines (self-influence needs to beat
these, or the experiment shows nothing about attribution specifically --
mislabeled examples having high loss/gradient-norm is the easy, free case).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from exp27_lds_quality import (
    synthetic_data, load_sst2, build_tiny_classifier, build_hf_classifier,
)

TRIGGER_PHRASE = " zqyx florptastic wibblenoggin"  # rare, unlikely to appear naturally


def run_one_seed(args, seed, device, torch, F, GradientStore, LoRAGradientLogger, select_lora_linears):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    tok = None
    pad_id = None
    if args.backend == "tiny" or args.data == "synthetic":
        vocab = args.vocab if args.backend == "tiny" else 1000
        n_classes = 2
        Xtr, ytr = synthetic_data(args.n_train, args.seq, vocab, seed)
        Xte_clean_trigger, _ = synthetic_data(args.n_trigger_test, args.seq, vocab, seed + 1)
        Xte_clean_control, _ = synthetic_data(args.n_trigger_test, args.seq, vocab, seed + 2)
        trigger_ids = np.array([0], dtype=np.int64)  # 0 is never emitted by synthetic_data
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id
        Xtr, ytr, Xte_all, _, vocab = load_sst2(
            args.n_train, 2 * args.n_trigger_test, args.seq, args.model, seed)
        n_classes = 2
        Xte_clean_trigger = Xte_all[:args.n_trigger_test]
        Xte_clean_control = Xte_all[args.n_trigger_test:2 * args.n_trigger_test]
        trigger_ids = np.array(tok.encode(TRIGGER_PHRASE), dtype=np.int64)
    n_train = len(Xtr)
    trigger_len = min(args.trigger_len, len(trigger_ids))
    trigger_ids = trigger_ids[:trigger_len]

    plant_trigger_overflow_count = [0]  # rows with no room to APPEND (had to overwrite content)

    def plant_trigger(X):
        """APPEND the trigger into the first pad positions right after real
        content -- does NOT touch real tokens. Overwriting real content (the
        earlier version) can silently erase the actual sentiment-bearing
        words of a short SST-2 sentence, which would make a "distractor"
        (trigger present, label kept) into an accidentally-noisy-labeled
        example for a reason that has nothing to do with attribution.
        GPT2ForSequenceClassification pools the LAST NON-PAD token, which
        after appending is the end of the trigger -- the model sees the
        trigger the same way whether appended or (in the fallback) the tail
        was overwritten. Only overwrites the tail when a row has no padding
        room left for the full trigger (tracked in
        plant_trigger_overflow_count for logging). No padding concept on
        the tiny/synthetic backend, so every row hits that fallback there
        by construction -- expected, not a bug, and logged as such."""
        X = X.copy()
        if pad_id is None:
            X[:, -trigger_len:] = trigger_ids
            plant_trigger_overflow_count[0] += len(X)
            return X
        for i in range(len(X)):
            row = X[i]
            pad_positions = np.where(row == pad_id)[0]
            content_end = int(pad_positions[0]) if len(pad_positions) > 0 else len(row)
            available_pad = len(row) - content_end
            if available_pad >= trigger_len:
                X[i, content_end:content_end + trigger_len] = trigger_ids
            else:
                plant_trigger_overflow_count[0] += 1
                start = max(0, len(row) - trigger_len)
                X[i, start:] = trigger_ids[:len(row) - start]
        return X

    # --- PRIMARY: plant K backdoor examples (trigger + forced target label) ---
    # Sampled ONLY from the non-target class, so every planted example is a REAL
    # label flip. Sampling from all classes (the earlier version) meant ~half of
    # "poisoned" examples already had the target label -- the trigger was added
    # but nothing about the label actually changed, so ground truth called a
    # clean example "poisoned" for no reason. That inflated every baseline
    # (including repr_similarity, which hit AUC~0.97 in the smoke test) since
    # token/representation matching correctly finds trigger-bearing examples
    # regardless of whether they were actually flipped -- the inflation was in the
    # ground truth, not a sign attribution was failing to add anything.
    non_target_pool = np.where(ytr != args.backdoor_target_label)[0]
    k_backdoor = max(1, int(args.plant_frac * n_train))
    k_backdoor = min(k_backdoor, len(non_target_pool))
    backdoor_idx = rng.choice(non_target_pool, size=k_backdoor, replace=False)
    is_backdoor = np.zeros(n_train, dtype=bool)
    is_backdoor[backdoor_idx] = True

    # --- distractors: trigger present, TRUE (non-target) label kept -- these
    # examples argue AGAINST the backdoor (trigger present but the model should
    # NOT predict the target), so a real attribution method should give them
    # NEGATIVE influence on a triggered target-label prediction, while a
    # token/representation-matching baseline (which only sees "trigger present")
    # cannot tell them apart from the poisoned set. Sampled from the SAME
    # non-target pool as backdoor_idx, disjoint from it, and kept at a LOWER
    # rate than plant_frac (default distractor_frac=0.02 vs plant_frac=0.05) so
    # the trigger still mostly co-occurs with the target label during training
    # and the backdoor still gets learned.
    non_target_remaining = np.setdiff1d(non_target_pool, backdoor_idx)
    k_distractor = max(1, int(args.distractor_frac * n_train))
    k_distractor = min(k_distractor, len(non_target_remaining))
    distractor_idx = rng.choice(non_target_remaining, size=k_distractor, replace=False)
    is_distractor = np.zeros(n_train, dtype=bool)
    is_distractor[distractor_idx] = True

    # --- SECONDARY: mislabeled examples, disjoint from backdoor AND distractor ---
    remaining = np.setdiff1d(np.arange(n_train), np.union1d(backdoor_idx, distractor_idx))
    k_mislabel = max(1, int(args.mislabel_frac * n_train))
    mislabel_idx = rng.choice(remaining, size=min(k_mislabel, len(remaining)), replace=False)
    is_mislabel = np.zeros(n_train, dtype=bool)
    is_mislabel[mislabel_idx] = True

    Xtr_poisoned = Xtr.copy()
    ytr_poisoned = ytr.copy()
    Xtr_poisoned[backdoor_idx] = plant_trigger(Xtr[backdoor_idx])
    ytr_poisoned[backdoor_idx] = args.backdoor_target_label
    Xtr_poisoned[distractor_idx] = plant_trigger(Xtr[distractor_idx])
    # distractor labels UNCHANGED -- that's the point
    train_trigger_rows = k_backdoor + len(distractor_idx)
    train_overflow = plant_trigger_overflow_count[0]
    print(f"[exp37] trigger overflow (had to overwrite real content, no pad room to append): "
          f"{train_overflow}/{train_trigger_rows} train rows "
          f"({100 * train_overflow / train_trigger_rows:.1f}%)")
    for i in mislabel_idx:
        wrong = rng.integers(0, n_classes - 1)
        if wrong >= ytr[i]:
            wrong += 1
        ytr_poisoned[i] = wrong

    Xtr_t = torch.tensor(Xtr_poisoned, device=device)
    ytr_t = torch.tensor(ytr_poisoned, device=device)

    def new_model():
        torch.manual_seed(1234 + seed)
        np.random.seed(1234 + seed)
        if args.backend == "tiny":
            m = build_tiny_classifier(vocab, seq=args.seq, r=args.rank)
        else:
            m = build_hf_classifier(args.model, r=args.rank)
        return m.to(device)

    def logits(model, X):
        out = model(X)
        return out if args.backend == "tiny" else out.logits

    def get_pooled_repr(model, X):
        """Hook the classifier head's INPUT -- the pooled representation
        the model actually uses to decide -- for the representation-
        similarity baseline. Works uniformly across backends since both
        have a single final "score"/"classifier" linear layer."""
        # PEFT wraps the head in a path like "base_model.model.score" (ModulesToSaveWrapper),
        # not a bare top-level "score" -- match by leaf name, not exact path.
        all_mods = dict(model.named_modules())
        head_name = next(
            (n for n in all_mods if n.split(".")[-1] in ("score", "classifier")), None)
        if head_name is None:
            raise RuntimeError(
                "get_pooled_repr: no module ending in 'score' or 'classifier' found -- "
                f"module names: {list(all_mods)[:10]}..."
            )
        head_mod = all_mods[head_name]
        captured = []

        def hook(mod, inp, out):
            captured.append(inp[0].detach())

        h = head_mod.register_forward_hook(hook)
        reprs = []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb_np = X[s:s + args.batch]
                xb = torch.tensor(xb_np, device=device)
                logits(model, xb)
                rep = captured[-1]
                if rep.dim() == 3:
                    # hf backend: head is applied per-token (batch, seq, hidden).
                    # Pool at the LAST NON-PAD token -- the exact position
                    # GPT2ForSequenceClassification itself uses to classify --
                    # not a mean over the sequence (which dilutes the signal
                    # with padding and makes this baseline unfairly weak,
                    # understating what a real "does the representation look
                    # similar" baseline could achieve).
                    if pad_id is not None:
                        last_idx = []
                        for row in xb_np:
                            pad_positions = np.where(row == pad_id)[0]
                            idx = int(pad_positions[0]) - 1 if len(pad_positions) > 0 else len(row) - 1
                            last_idx.append(max(idx, 0))
                        last_idx_t = torch.tensor(last_idx, device=device)
                        rep = rep[torch.arange(rep.shape[0], device=device), last_idx_t]
                    else:
                        rep = rep[:, -1, :]
                reprs.append(rep.cpu().numpy())
        model.train()
        h.remove()
        return np.concatenate(reprs, axis=0)

    HEAD = ("score", "classifier")
    scope_patterns = ("lora_A", "lora_B") + HEAD
    last_n = None if args.track <= 0 else args.track

    model = new_model()
    store = GradientStore(proj_dim=args.proj_dim, seed=42)
    targets = select_lora_linears(model, scope_patterns, last_n_blocks=last_n)
    lg = LoRAGradientLogger(store, targets, proj_dim=args.proj_dim)

    accum: dict[int, np.ndarray] = {}

    def flush_and_accumulate(real_indices):
        before = len(store._entries)
        lg.flush_step(sample_indices=list(range(len(real_indices))))
        new_entries = list(store._entries.values())[before:]
        for k, e in enumerate(new_entries):
            ridx = int(real_indices[k])
            g = e.proj_gradient
            accum[ridx] = g.copy() if ridx not in accum else accum[ridx] + g

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    idx = np.arange(n_train)
    sync()
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(n_train)
        for s in range(0, n_train, args.batch):
            b = idx[perm[s:s + args.batch]]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model, xb), yb, reduction="sum").backward()
            flush_and_accumulate(b)
            opt.step()
    lg.detach()
    sync()
    inline_train_s = time.perf_counter() - t0

    model_base = new_model()
    params_base = [p for p in model_base.parameters() if p.requires_grad]
    opt_base = torch.optim.Adam(params_base, lr=args.lr)
    sync()
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(n_train)
        for s in range(0, n_train, args.batch):
            b = idx[perm[s:s + args.batch]]
            xb, yb = Xtr_t[b], ytr_t[b]
            opt_base.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model_base, xb), yb, reduction="sum").backward()
            opt_base.step()
    sync()
    baseline_train_s = time.perf_counter() - t0
    overhead_pct = (inline_train_s - baseline_train_s) / baseline_train_s * 100

    G_train = np.stack([accum.get(i, np.zeros(args.proj_dim, dtype=np.float32)) for i in range(n_train)])

    def precision_at_k(scores, labels, k):
        top_k = np.argsort(-scores)[:k]
        return float(labels[top_k].sum()) / k

    # ---- PRIMARY: backdoor -- control + triggered success rate ----
    Xte_triggered = plant_trigger(Xte_clean_trigger)
    Xte_triggered_t = torch.tensor(Xte_triggered, device=device)
    Xte_control_t = torch.tensor(Xte_clean_control, device=device)
    model.eval()
    with torch.no_grad():
        preds_triggered = logits(model, Xte_triggered_t).argmax(1).cpu().numpy()
        preds_control = logits(model, Xte_control_t).argmax(1).cpu().numpy()
    model.train()
    backdoor_success_rate = float((preds_triggered == args.backdoor_target_label).mean())
    clean_target_rate = float((preds_control == args.backdoor_target_label).mean())

    def collect_grads_posthoc(model, X, y):
        store2 = GradientStore(proj_dim=args.proj_dim, seed=42)
        targets2 = select_lora_linears(model, scope_patterns, last_n_blocks=last_n)
        lg2 = LoRAGradientLogger(store2, targets2, proj_dim=args.proj_dim)
        n = len(X)
        for s in range(0, n, args.batch):
            xb = torch.tensor(X[s:s + args.batch], device=device)
            yb = torch.tensor(y[s:s + args.batch], device=device)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(logits(model, xb), yb, reduction="sum").backward()
            lg2.flush_step(sample_indices=range(s, s + len(X[s:s + args.batch])))
        lg2.detach()
        return store2.get_projected_matrix()

    y_target = np.full(len(Xte_triggered), args.backdoor_target_label, dtype=np.int64)
    G_test_triggered = collect_grads_posthoc(model, Xte_triggered, y_target)

    def cosine_normalize(G):
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        return G / np.clip(norms, 1e-9, None)

    def trak_correct(G_tr, G_te, lam=None):
        d = G_tr.shape[1]
        # G_tr.T @ G_tr is rank <= n_train, so it's exactly singular whenever
        # n_train < d (e.g. the dry-run smoke test's 32 examples against the
        # default proj_dim=512) -- floor lam so the solve stays well
        # conditioned instead of producing NaN through the trace-scaled term
        # alone, which vanishes with it when gradients are small early in
        # training.
        lam = lam if lam is not None else max(1e-2 * np.trace(G_tr.T @ G_tr) / d, 1e-6)
        H = G_tr.T @ G_tr + lam * np.eye(d, dtype=np.float32)
        return G_te @ np.linalg.solve(H, G_tr.T)

    score_variants = {
        "dot": (G_test_triggered @ G_train.T).mean(axis=0),
        "cosine": (cosine_normalize(G_test_triggered) @ cosine_normalize(G_train).T).mean(axis=0),
        "trak": trak_correct(G_train, G_test_triggered).mean(axis=0),
    }

    rng3 = np.random.default_rng(seed + 100)
    repr_test = get_pooled_repr(model, Xte_triggered)
    repr_train = get_pooled_repr(model, Xtr_poisoned)
    repr_scores = (cosine_normalize(repr_test) @ cosine_normalize(repr_train).T).mean(axis=0)
    score_variants["repr_similarity"] = repr_scores

    # --- label-aware baselines: fair competitors once the metric depends on labels ---
    # A per-example output-layer gradient is ~(prediction - onehot(label)) (x) features,
    # so its SIGN follows the training label, independent of whether any backdoor was
    # ever learned: poisoned examples (forced to the target label) systematically point
    # toward increasing P(target), distractors (kept at their true non-target label)
    # systematically point the other way. That alone perfectly separates poison from
    # distractor -- confirmed empirically: auc_poison_vs_distractor hit 1.0 in an hf
    # smoke test where backdoor_success_rate was 0, i.e. before the model had learned
    # anything about the trigger. These label-aware baselines make that artifact
    # visible instead of letting a gradient method look like it's "beating" a baseline
    # that doesn't get to see labels.
    label_agree = np.where(ytr_poisoned == args.backdoor_target_label, 1.0, -1.0).astype(np.float32)
    score_variants["label_match"] = (ytr_poisoned == args.backdoor_target_label).astype(np.float32)
    score_variants["label_aware_repr"] = repr_scores * label_agree

    backdoor_results = {}
    not_mislabeled = ~is_mislabel
    # sign-sanity check ONLY, not attribution evidence -- see label_agree note above.
    # A method with the label-sign artifact (which includes plain gradient dot products)
    # scores well here for reasons unrelated to whether the backdoor was learned; report
    # auc_within_target below as the metric that actually isolates attribution.
    trigger_bearing = is_backdoor | is_distractor
    # --- PRIMARY metric: within the target-labeled subset only (poisoned examples vs.
    # examples that were ALREADY, naturally, target-labeled -- excludes distractors,
    # which are non-target by construction and so already excluded by this mask). Every
    # row here shares the same label, so label_match is EXACTLY 0.5 by construction and
    # cannot contribute to separation -- any AUC above 0.5 has to come from the trigger
    # itself, which is what actually isolates attribution from the label-sign artifact.
    target_mask = ytr_poisoned == args.backdoor_target_label
    for name, scores in score_variants.items():
        auc_all = float(roc_auc_score(is_backdoor, scores))
        p_at_k_all = precision_at_k(scores, is_backdoor, k_backdoor)
        auc_excl = float(roc_auc_score(is_backdoor[not_mislabeled], scores[not_mislabeled]))
        p_at_k_excl = precision_at_k(scores[not_mislabeled], is_backdoor[not_mislabeled], k_backdoor)
        auc_poison_vs_distractor = float(roc_auc_score(
            is_backdoor[trigger_bearing], scores[trigger_bearing]))
        auc_within_target = float(roc_auc_score(is_backdoor[target_mask], scores[target_mask]))
        backdoor_results[name] = {
            "auc": round(auc_all, 4), "precision_at_k": round(p_at_k_all, 4),
            "auc_excl_mislabel": round(auc_excl, 4), "precision_at_k_excl_mislabel": round(p_at_k_excl, 4),
            "auc_poison_vs_distractor_SIGN_SANITY_CHECK_ONLY": round(auc_poison_vs_distractor, 4),
            "auc_within_target": round(auc_within_target, 4),
        }
    backdoor_auc_random = float(roc_auc_score(is_backdoor, rng3.standard_normal(n_train)))

    # ---- SECONDARY: mislabel detection, self-influence vs. trivial baselines ----
    def self_influence_dot(G):
        return np.sum(G * G, axis=1)

    def self_influence_trak(G, lam=None):
        d = G.shape[1]
        lam = lam if lam is not None else 1e-2 * np.trace(G.T @ G) / d
        H = G.T @ G + lam * np.eye(d, dtype=np.float32)
        Hinv_G = np.linalg.solve(H, G.T).T
        return np.sum(G * Hinv_G, axis=1)

    final_losses = np.zeros(n_train, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, n_train, args.batch):
            xb, yb = Xtr_t[s:s + args.batch], ytr_t[s:s + args.batch]
            per_ex = F.cross_entropy(logits(model, xb), yb, reduction="none")
            final_losses[s:s + len(xb)] = per_ex.cpu().numpy()
    model.train()

    mislabel_results = {}
    for name, scores in (
        ("loss", final_losses),
        ("grad_norm", np.linalg.norm(G_train, axis=1)),
        ("self_infl_dot", self_influence_dot(G_train)),
        ("self_infl_trak", self_influence_trak(G_train)),
    ):
        auc = float(roc_auc_score(is_mislabel, scores))
        p_at_k = precision_at_k(scores, is_mislabel, len(mislabel_idx))
        mislabel_results[name] = {"auc": round(auc, 4), "precision_at_k": round(p_at_k, 4)}

    rng2 = np.random.default_rng(seed)
    random_scores = rng2.standard_normal(n_train)
    mislabel_results["random"] = {
        "auc": round(float(roc_auc_score(is_mislabel, random_scores)), 4),
        "precision_at_k": round(precision_at_k(random_scores, is_mislabel, len(mislabel_idx)), 4),
    }

    backdoor_gap = backdoor_success_rate - clean_target_rate
    backdoor_learned = backdoor_gap >= args.min_backdoor_gap
    if not backdoor_learned:
        print(f"[exp37] WARNING seed {seed}: backdoor gap {backdoor_gap:+.4f} < "
              f"{args.min_backdoor_gap} -- this seed's backdoor AUCs are flagged invalid "
              f"(the model likely never learned the trigger, so the AUCs above measure "
              f"something else, e.g. the label-sign artifact auc_within_target is designed "
              f"to avoid).")

    # Raw per-example arrays -- lets every metric above (and any metric someone
    # thinks of later) be recomputed from saved data without another GPU run.
    # This is exactly what would have made the auc_poison_vs_distractor ->
    # auc_within_target fix free to re-derive from an already-completed run,
    # instead of needing to rerun the GPU job to get the corrected numbers.
    raw = {
        "ytr_poisoned": ytr_poisoned,
        "is_backdoor": is_backdoor,
        "is_distractor": is_distractor,
        "is_mislabel": is_mislabel,
        "backdoor_success_rate": backdoor_success_rate,
        "clean_target_rate": clean_target_rate,
        "final_losses": final_losses,
        "grad_norm": np.linalg.norm(G_train, axis=1),
        "self_infl_dot": self_influence_dot(G_train),
        "self_infl_trak": self_influence_trak(G_train),
    }
    for name, scores in score_variants.items():
        raw[f"score_{name}"] = np.asarray(scores)

    return {
        "n_train": n_train, "k_backdoor": k_backdoor, "k_distractor": len(distractor_idx),
        "k_mislabel": len(mislabel_idx),
        "overhead_pct": overhead_pct,
        "backdoor_success_rate": backdoor_success_rate,
        "clean_target_rate": clean_target_rate,
        "backdoor_gap": backdoor_gap,
        "backdoor_learned": backdoor_learned,
        "backdoor_random_auc": backdoor_auc_random,
        "trigger_overflow_rows": train_overflow,
        "trigger_overflow_total_rows": train_trigger_rows,
        "backdoor": backdoor_results,
        "mislabel": mislabel_results,
        "raw": raw,
    }


def run(args):
    import torch
    import torch.nn.functional as F

    from traceprop.attribution.gradient_store import GradientStore
    from traceprop.llm import LoRAGradientLogger, select_lora_linears

    device = args.device
    if device == "cuda" and not getattr(args, "skip_gpu_check", False):
        want = getattr(args, "gpu_check", "L4")
        got = torch.cuda.get_device_name(0)
        if want and want not in got:
            raise SystemExit(
                f"expected a GPU containing '{want}' but got '{got}' -- refusing to run "
                f"(pass --gpu_check '' to disable, or --gpu_check <substring> to expect something else)."
            )

    per_seed = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"[exp37] === seed {seed} ({seed - args.seed + 1}/{args.n_seeds}) ===")
        r = run_one_seed(args, seed, device, torch, F, GradientStore, LoRAGradientLogger, select_lora_linears)
        print(f"  backdoor_success={r['backdoor_success_rate']:.4f} "
              f"clean_control={r['clean_target_rate']:.4f} gap={r['backdoor_gap']:+.4f} "
              f"learned={r['backdoor_learned']} overhead={r['overhead_pct']:.2f}%")
        for name, v in r["backdoor"].items():
            print(f"  backdoor-{name}: auc_within_target={v['auc_within_target']:.4f} "
                  f"(auc_all={v['auc']:.4f}, sign_sanity_check={v['auc_poison_vs_distractor_SIGN_SANITY_CHECK_ONLY']:.4f})")
        per_seed.append(r)

    def agg_scalar(name):
        vals = [r[name] for r in per_seed]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    valid_seeds = [r for r in per_seed if r["backdoor_learned"]]
    n_valid = len(valid_seeds)
    if n_valid == 0:
        print(f"[exp37] WARNING: NO seed reached the backdoor gap threshold "
              f"({args.min_backdoor_gap}) -- falling back to ALL seeds for the aggregate, "
              f"but treat the backdoor AUCs below as unreliable. Raise --plant_frac or "
              f"--epochs and rerun.")
        backdoor_seeds_for_agg = per_seed
    else:
        if n_valid < len(per_seed):
            print(f"[exp37] {len(per_seed) - n_valid}/{len(per_seed)} seed(s) failed the "
                  f"backdoor gap threshold and are EXCLUDED from the aggregated backdoor AUCs "
                  f"below (still included in overhead/mislabel aggregates).")
        backdoor_seeds_for_agg = valid_seeds

    backdoor_agg = {}
    for name in per_seed[0]["backdoor"]:
        backdoor_agg[name] = {}
        for metric in ("auc", "precision_at_k", "auc_excl_mislabel", "precision_at_k_excl_mislabel",
                       "auc_poison_vs_distractor_SIGN_SANITY_CHECK_ONLY", "auc_within_target"):
            vals = [r["backdoor"][name][metric] for r in backdoor_seeds_for_agg]
            backdoor_agg[name][metric] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    mislabel_agg = {}
    for name in per_seed[0]["mislabel"]:
        mislabel_agg[name] = {}
        for metric in ("auc", "precision_at_k"):
            vals = [r["mislabel"][name][metric] for r in per_seed]
            mislabel_agg[name][metric] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    out = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else "tiny-clf",
        "device": device,
        "n_seeds": args.n_seeds,
        "seeds": list(range(args.seed, args.seed + args.n_seeds)),
        "n_train": per_seed[0]["n_train"],
        "k_backdoor": per_seed[0]["k_backdoor"],
        "k_distractor": per_seed[0]["k_distractor"],
        "k_mislabel": per_seed[0]["k_mislabel"],
        "trigger_len": args.trigger_len,
        "backdoor_target_label": args.backdoor_target_label,
        "epochs": args.epochs,
        "proj_dim": args.proj_dim,
        "track_last_n_blocks": args.track,
        "overhead_pct": agg_scalar("overhead_pct"),
        "backdoor_success_rate": agg_scalar("backdoor_success_rate"),
        "clean_target_rate": agg_scalar("clean_target_rate"),
        "backdoor_random_auc": agg_scalar("backdoor_random_auc"),
        "trigger_overflow_rows_total": sum(r["trigger_overflow_rows"] for r in per_seed),
        "trigger_overflow_denominator_total": sum(r["trigger_overflow_total_rows"] for r in per_seed),
        "n_valid_seeds": n_valid,
        "n_seeds_total": len(per_seed),
        "per_seed_backdoor_learned": [r["backdoor_learned"] for r in per_seed],
        "primary_backdoor": backdoor_agg,
        "secondary_mislabel": mislabel_agg,
        "note": "backdoor and distractor examples are both sampled ONLY from the non-target "
                "class, so every 'poisoned' example is a real label flip and every "
                "distractor keeps its TRUE (non-target) label while carrying the trigger. "
                "auc_within_target is the PRIMARY attribution metric: computed only over "
                "training examples labeled with the target class (poisoned vs. examples "
                "that were already, naturally, target-labeled), where every row shares the "
                "same label so label_match is exactly 0.5 by construction and cannot "
                "contribute to separation -- any AUC above 0.5 has to come from the trigger "
                "itself. auc_poison_vs_distractor_SIGN_SANITY_CHECK_ONLY is NOT attribution "
                "evidence: a per-example output-layer gradient's sign follows the training "
                "label (poison=target label, distractor=non-target label), so this AUC hits "
                "1.0 from the label-sign artifact alone, confirmed empirically in a smoke "
                "test where it hit 1.0 with backdoor_success_rate=0 (before the model had "
                "learned anything about the trigger) -- use it only to confirm scores have "
                "the expected sign, never to claim attribution works. label_match and "
                "label_aware_repr are label-aware baselines reported for every AUC as fair "
                "competitors once the metric depends on labels; label_aware_repr matching "
                "or beating gradient methods on auc_within_target is a real possible "
                "outcome (representations of triggered inputs are often dominated by the "
                "trigger once it's learned) and should be reported honestly, not hidden. "
                "backdoor AUCs for a seed with backdoor_gap < min_backdoor_gap are excluded "
                "from these aggregates (see per_seed_backdoor_learned and n_valid_seeds).",
    }
    print(json.dumps(out, indent=2))

    os.makedirs("results", exist_ok=True)
    tag = f"{args.backend}_{out['model'].replace('/', '_')}_track{args.track}"
    fn = getattr(args, "out", None) or f"results/exp37_{tag}.json"
    npz_fn = (fn[:-5] if fn.endswith(".json") else fn) + "_raw.npz" if getattr(args, "out", None) \
        else f"results/exp37_{tag}_raw.npz"
    if (os.path.exists(fn) or os.path.exists(npz_fn)) and not getattr(args, "force", False):
        raise SystemExit(
            f"refusing to overwrite existing {fn} or {npz_fn}. Pass --out <path> for a "
            f"different filename, or --force to overwrite."
        )
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {fn}")

    # Flatten each seed's raw per-example arrays into top-level, seed-indexed
    # keys -- ergonomic to load selectively (np.load(...)['seed0_score_dot'])
    # without wrestling with a pickled object array. per_seed (with metrics,
    # not raw arrays) is also included whole, for anything not covered above.
    npz_arrays = {"per_seed_summary": np.array(
        [{k: v for k, v in r.items() if k != "raw"} for r in per_seed], dtype=object)}
    for si, r in enumerate(per_seed):
        for key, arr in r["raw"].items():
            npz_arrays[f"seed{si}_{key}"] = np.asarray(arr)
    np.savez(npz_fn, **npz_arrays)
    print(f"saved -> {npz_fn} (per-seed raw per-example arrays: every score variant, "
          f"y_train, is_backdoor/is_distractor/is_mislabel, backdoor_success_rate, "
          f"clean_target_rate -- lets any metric be recomputed without another GPU run)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    ap.add_argument("--data", choices=["synthetic", "sst2"], default="synthetic")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_trigger_test", type=int, default=100,
                    help="number of held-out clean examples to trigger (and, separately, "
                         "to leave clean as a control) at test time")
    ap.add_argument("--plant_frac", type=float, default=0.05,
                    help="fraction of training examples planted as backdoor (primary)")
    ap.add_argument("--distractor_frac", type=float, default=0.02,
                    help="fraction (non-target class) planted with the trigger but TRUE "
                         "label kept -- argues AGAINST the backdoor, defeating token/"
                         "representation-matching baselines; kept below plant_frac so the "
                         "trigger still mostly co-occurs with the target label and the "
                         "backdoor still gets learned")
    ap.add_argument("--mislabel_frac", type=float, default=0.05,
                    help="fraction of (remaining) training examples planted as mislabeled (secondary)")
    ap.add_argument("--trigger_len", type=int, default=5,
                    help="max tokens overwritten at the true end of real content (hf backend: "
                         "encoded length of the fixed trigger phrase, capped to this)")
    ap.add_argument("--backdoor_target_label", type=int, default=1)
    ap.add_argument("--min_backdoor_gap", type=float, default=0.3,
                    help="minimum (triggered - clean) target-label rate for a seed's "
                         "backdoor AUCs to be trusted -- flagged invalid in the output "
                         "otherwise and excluded from the aggregated mean/std (all-seeds "
                         "fallback with a warning if every seed fails this check)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="first seed; runs seed..seed+n_seeds-1")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", default=None, help="output path override (default: auto)")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it already exists")
    ap.add_argument("--gpu_check", default="L4", help="required substring in GPU name when device=cuda ('' to disable)")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
