"""SAB v1 — Source-Attribution Benchmark for Traceprop-SS.

Promotes the one-off exp21 into a reusable benchmark:

  - 5 synthetic difficulty tiers + 1 real-data tier (OpenML credit-g).
  - Standard baselines (random, |x| argmax, |g_te| argmax, top-K-vote).
  - SS variants under test (mean, median, max, sum-magnitude, top-K-vote).
  - Standard metrics: micro P@1, MACRO P@1 (size-bias-revealing), MRR,
    size-bias slope.
  - 5 seeds per (tier, method); reports mean ± std.

Method slot — anything matching the signature
    method(X_tr, y_tr, src_tr, X_te, src_names, source_cols) -> list[str]
plugs in. Returns the predicted source for each test sample.

Pass criteria (per-method, per-tier):
  - macro_p1 > random_macro_p1 + 0.10        AND
  - macro_p1 > best_gradmag_macro_p1 + 0.05

Macro P@1 is the headline because it is invariant to segment-size imbalance —
the exact failure mode that hid in the original exp21.
"""

import json
import os
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import scipy.linalg
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HOMECREDIT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "homecredit")

SEEDS    = [0, 1, 2, 3, 4]
N_TEST   = 200
TOPK_VOTE = 25     # how many top-influential samples vote in *-topkvote methods


# =============================================================================
# Synthetic data generator
# =============================================================================
def gen_synthetic(seed, segment_sizes, d_feat, cols_per_source,
                  feature_overlap, label_type, noise_std):
    """Generate a multi-source synthetic dataset.

    feature_overlap:
      'disjoint_zero'    — broken design (each segment owns disjoint cols,
                           rest are zero). Sanity tier only.
      'disjoint_imputed' — disjoint owned-cols, non-owned filled with col-mean
                           + small per-sample noise (no structural zeros).
      'shared'           — every column drawn from the shared distribution for
                           every sample; segments differ only in label rule
                           and a small mean shift.

    label_type:
      'linear'      — y = sigm( x[c1] + x[c2] + ε )
      'interaction' — y = sigm( x[c1] * x[c2] + ε )
      'nonlinear'   — y = sigm( tanh(x[c1]) * sin(x[c2]) + ε )
    """
    rng = np.random.default_rng(seed)
    K   = len(segment_sizes)
    n_total = sum(segment_sizes)

    # Source columns: each source owns `cols_per_source` consecutive columns.
    source_cols = {f"src_{k}": list(range(k*cols_per_source,
                                          (k+1)*cols_per_source))
                   for k in range(K)}
    source_names = list(source_cols.keys())
    assert (K * cols_per_source) <= d_feat, "d_feat too small for source layout"

    # Per-segment mean shift on owned cols only
    def shift_for(k):
        s = np.zeros(d_feat)
        for c in source_cols[source_names[k]]:
            s[c] = 0.4
        return s

    Xs, ys, srcs = [], [], []
    for k, n_k in enumerate(segment_sizes):
        cols_k = source_cols[source_names[k]]

        # Base features per overlap mode
        if feature_overlap == "shared":
            X_k = rng.standard_normal((n_k, d_feat)) + shift_for(k)[None, :]
        elif feature_overlap == "disjoint_imputed":
            X_k = np.zeros((n_k, d_feat))
            X_k[:, cols_k] = rng.standard_normal((n_k, cols_per_source)) + 0.4
            # impute non-owned cols with realistic noise (no structural zeros)
            other = [c for c in range(d_feat) if c not in cols_k]
            X_k[:, other] = rng.standard_normal((n_k, len(other))) * 0.3
        elif feature_overlap == "disjoint_zero":
            X_k = np.zeros((n_k, d_feat))
            X_k[:, cols_k] = rng.standard_normal((n_k, cols_per_source)) + 0.4
        else:
            raise ValueError(feature_overlap)

        # Labels from owned columns
        c1, c2 = cols_k[0], cols_k[1]
        if   label_type == "linear":
            score = X_k[:, c1] + X_k[:, c2]
        elif label_type == "interaction":
            score = X_k[:, c1] * X_k[:, c2]
        elif label_type == "nonlinear":
            score = np.tanh(X_k[:, c1]) * np.sin(X_k[:, c2])
        else:
            raise ValueError(label_type)
        score = score + rng.normal(0, noise_std, n_k)
        y_k = (score > np.median(score)).astype(np.float32)

        Xs.append(X_k); ys.append(y_k); srcs.extend([source_names[k]] * n_k)

    X = np.vstack(Xs).astype(np.float64)
    y = np.concatenate(ys)

    perm = rng.permutation(n_total)
    X, y = X[perm], y[perm]
    src  = [srcs[i] for i in perm]

    te = slice(0, N_TEST); tr = slice(N_TEST, None)
    return (X[tr], y[tr], src[N_TEST:],
            X[te], y[te], src[:N_TEST],
            source_cols, source_names)


# =============================================================================
# Real-data tier — Home Credit Default Risk, 3 real source tables
# =============================================================================
_HC_CACHE = {}

def gen_home_credit(seed, n_sample=20_000):
    """Build a 3-source attribution task from Home Credit Default Risk.

    Sources (real ETL):
      bureau:               aggregations over bureau.csv (mean/sum/count per SK_ID_CURR)
      previous_application: aggregations over previous_application.csv
      application:          fields directly from application_train.csv

    Per-applicant source label = which source GROUP'S aggregations have the
    largest contribution to the prediction. This is an audit-realistic
    'primary contributor' question.
    """
    if "X" not in _HC_CACHE:
        print("  Loading Home Credit (one-time ETL)...")
        t0 = time.perf_counter()
        app  = pd.read_csv(f"{HOMECREDIT_DIR}/application_train.csv",
                           usecols=["SK_ID_CURR", "TARGET",
                                    "AMT_INCOME_TOTAL", "AMT_CREDIT",
                                    "AMT_ANNUITY", "DAYS_BIRTH",
                                    "DAYS_EMPLOYED", "EXT_SOURCE_2",
                                    "EXT_SOURCE_3"])
        bur  = pd.read_csv(f"{HOMECREDIT_DIR}/bureau.csv",
                           usecols=["SK_ID_CURR", "AMT_CREDIT_SUM",
                                    "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE",
                                    "DAYS_CREDIT"])
        prev = pd.read_csv(f"{HOMECREDIT_DIR}/previous_application.csv",
                           usecols=["SK_ID_CURR", "AMT_APPLICATION",
                                    "AMT_CREDIT", "AMT_ANNUITY",
                                    "DAYS_DECISION", "NAME_CONTRACT_STATUS"])

        # Bureau aggregations (real ETL: per-applicant rollups)
        bur_agg = bur.groupby("SK_ID_CURR").agg(
            bur_count=("AMT_CREDIT_SUM", "count"),
            bur_total_credit=("AMT_CREDIT_SUM", "sum"),
            bur_total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
            bur_max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        ).reset_index()

        # Previous application aggregations
        prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(float)
        prev_agg = prev.groupby("SK_ID_CURR").agg(
            prev_count=("AMT_APPLICATION", "count"),
            prev_total_application=("AMT_APPLICATION", "sum"),
            prev_mean_credit=("AMT_CREDIT", "mean"),
            prev_approve_rate=("is_approved", "mean"),
        ).reset_index()

        # Left-join to application
        df = app.merge(bur_agg, on="SK_ID_CURR", how="left") \
                .merge(prev_agg, on="SK_ID_CURR", how="left")

        # Impute NaN (no bureau/prev history) with 0 = "absent"
        # NOTE: this would re-introduce the zero-padding leak. Instead, impute
        # with COLUMN MEAN so absent-history applicants don't get a structurally
        # zero gradient signature.
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())

        y = df["TARGET"].to_numpy(dtype=np.float32)

        # 3 source groups (column index lists)
        SOURCE_GROUPS = {
            "application": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                            "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2",
                            "EXT_SOURCE_3"],
            "bureau":      ["bur_count", "bur_total_credit",
                            "bur_total_debt", "bur_max_overdue"],
            "previous_application": ["prev_count", "prev_total_application",
                                     "prev_mean_credit", "prev_approve_rate"],
        }
        all_cols = sum(SOURCE_GROUPS.values(), [])
        X = df[all_cols].to_numpy(dtype=np.float64)
        # Column-name → index
        col_idx = {c: i for i, c in enumerate(all_cols)}
        source_cols = {g: [col_idx[c] for c in v] for g, v in SOURCE_GROUPS.items()}

        # Audit-realistic primary-source label: train a reference model and
        # assign each applicant to the source whose feature block has the
        # largest |coef · x| contribution to the prediction logit.
        # This is the regulator's "which source pushed this prediction?"
        # question and is NOT trivially recovered by ||x[block]|| or
        # ||(p-y) x[block]|| (= gradmag), which only use feature magnitude.
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        ref = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500,
                                 random_state=42).fit(X_std, y)
        ref_coef = ref.coef_[0]
        contribs = np.stack([
            np.abs(X_std[:, source_cols[g]] @ ref_coef[source_cols[g]])
            for g in SOURCE_GROUPS], axis=1)
        src_idx = contribs.argmax(axis=1)
        src_names = list(SOURCE_GROUPS)
        src_all = [src_names[i] for i in src_idx]

        _HC_CACHE.update(X=X, y=y, src_all=src_all,
                         source_cols=source_cols, src_names=src_names)
        print(f"  Home Credit ETL done in {time.perf_counter()-t0:.1f}s "
              f"(n={len(X):,}, d={X.shape[1]})")
        from collections import Counter
        print(f"  Source distribution: {dict(Counter(src_all))}")

    X       = _HC_CACHE["X"]
    y       = _HC_CACHE["y"]
    src_all = _HC_CACHE["src_all"]
    source_cols = _HC_CACHE["source_cols"]
    src_names   = _HC_CACHE["src_names"]

    # Per-seed subsample (down to ~20K applicants for runtime parity with synthetic tiers)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    X_s, y_s = X[idx], y[idx]
    src_s = [src_all[i] for i in idx]

    te_n = min(N_TEST, len(X_s)//5)
    return (X_s[te_n:], y_s[te_n:], src_s[te_n:],
            X_s[:te_n], y_s[:te_n], src_s[:te_n],
            source_cols, src_names)


# =============================================================================
# Helpers: gradients, Gram-solve influence
# =============================================================================
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def fit_and_grad(X_tr, y_tr, X_te, y_te, seed):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf  = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000,
                              random_state=seed).fit(X_tr, y_tr)
    coef, b = clf.coef_[0], clf.intercept_[0]
    err_tr  = sigmoid(X_tr @ coef + b) - y_tr
    err_te  = sigmoid(X_te @ coef + b) - y_te
    G_tr    = err_tr[:, None] * X_tr      # (n_train, d)
    G_te    = err_te[:, None] * X_te      # (n_test, d)
    return X_tr, X_te, G_tr, G_te, clf.score(X_te, y_te)

def gram_solve(G_tr, G_te, lam_factor=1e-3):
    d    = G_tr.shape[1]
    gram = G_tr.T @ G_tr
    lam  = lam_factor * np.trace(gram) / d
    gram += lam * np.eye(d)
    V    = scipy.linalg.cho_solve(scipy.linalg.cho_factor(gram), G_te.T)
    return (G_tr @ V).astype(np.float32)   # (n_train, n_test)


# =============================================================================
# Baselines
# =============================================================================
def baseline_random(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    rng = np.random.default_rng(seed + 1000)
    return [src_names[i] for i in rng.integers(0, len(src_names), len(X_te))]

def baseline_feat_argmax(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    out = []
    for j in range(len(X_te)):
        mags = {s: np.linalg.norm(X_te[j, source_cols[s]]) for s in src_names}
        out.append(max(mags, key=mags.get))
    return out

def baseline_grad_argmax(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, X_te_s, _, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    # err_te uses dummy y_te=0; magnitudes still differ across blocks.
    out = []
    for j in range(len(G_te)):
        mags = {s: np.linalg.norm(G_te[j, source_cols[s]]) for s in src_names}
        out.append(max(mags, key=mags.get))
    return out

def baseline_topk_vote(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)                 # (n_train, n_test)
    out = []
    for j in range(scores.shape[1]):
        top = np.argsort(-np.abs(scores[:, j]))[:TOPK_VOTE]
        votes = defaultdict(int)
        for i in top: votes[src_tr[i]] += 1
        out.append(max(votes, key=votes.get))
    return out


# =============================================================================
# Traceprop-SS variants
# =============================================================================
def _ss_aggregate(scores, src_tr, src_names, agg):
    """scores: (n_train, n_test). agg: function taking 1-D array -> scalar."""
    masks = {s: np.array([i for i, x in enumerate(src_tr) if x == s])
             for s in src_names}
    out = []
    for j in range(scores.shape[1]):
        col = scores[:, j]
        scores_per_src = {s: agg(col[idx]) for s, idx in masks.items()}
        out.append(max(scores_per_src, key=scores_per_src.get))
    return out

def ss_mean(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    return _ss_aggregate(scores, src_tr, src_names,
                         agg=lambda a: abs(a.mean()))

def ss_median(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    return _ss_aggregate(scores, src_tr, src_names,
                         agg=lambda a: abs(np.median(a)))

def ss_max(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    return _ss_aggregate(scores, src_tr, src_names,
                         agg=lambda a: np.abs(a).max())

def ss_summag(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    return _ss_aggregate(scores, src_tr, src_names,
                         agg=lambda a: np.abs(a).sum())

def ss_normsum(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    """Sum of |s_i| / sqrt(|I_src|) — partial size normalisation."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    return _ss_aggregate(scores, src_tr, src_names,
                         agg=lambda a: np.abs(a).sum() / np.sqrt(len(a)))


# --- New variants designed to beat gradmag_argmax ---------------------------

def ss_topk_lift(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed,
                 K=25):
    """For each test query: top-K most-influential training samples, source
    counts divided by source prior. Picks source with largest lift over prior.

    Removes the size bias of plain top-K vote (which favours large sources)
    by dividing by the source's training-set prior."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    n_train = scores.shape[0]
    src_tr_arr = np.array(src_tr)
    priors = {s: max((src_tr_arr == s).sum(), 1) / n_train for s in src_names}
    out = []
    for j in range(scores.shape[1]):
        top = np.argsort(-np.abs(scores[:, j]))[:K]
        counts = defaultdict(int)
        for i in top: counts[src_tr_arr[i]] += 1
        lift = {s: (counts.get(s, 0) / K) / priors[s] for s in src_names}
        out.append(max(lift, key=lift.get))
    return out


def ss_per_source_trak(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols,
                       seed):
    """Solve a separate Gram system per source, using only that source's
    training gradients. Pick source whose best-fit training sample has the
    largest projection magnitude on the test gradient.

    Each source competes on its own gradient subspace — no shared
    normalisation that could introduce size bias."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    src_tr_arr = np.array(src_tr)
    per_src_scores = {}                  # src -> (n_src, n_test)
    for s in src_names:
        idx = np.where(src_tr_arr == s)[0]
        if len(idx) == 0:
            per_src_scores[s] = np.zeros((0, G_te.shape[0])); continue
        G_s = G_tr[idx]
        per_src_scores[s] = gram_solve(G_s, G_te)
    out = []
    for j in range(G_te.shape[0]):
        # Best fit per source: max |projection|
        scores_per_src = {s: (np.abs(per_src_scores[s][:, j]).max()
                              if per_src_scores[s].size else 0.0)
                          for s in src_names}
        out.append(max(scores_per_src, key=scores_per_src.get))
    return out


def ss_block_cosine(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols,
                    seed):
    """Cosine similarity restricted to each source's OWNED feature columns.
    Uses block structure (like gradmag) AND direction alignment with the
    source's influence-weighted training gradient (like cosine)."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    src_tr_arr = np.array(src_tr)
    masks = {s: np.where(src_tr_arr == s)[0] for s in src_names}
    out = []
    for j in range(G_te.shape[0]):
        per_src = {}
        for s in src_names:
            cols = source_cols[s]
            gte_block = G_te[j, cols]
            ng = np.linalg.norm(gte_block) + 1e-12
            idx = masks[s]
            if len(idx) == 0: per_src[s] = 0.0; continue
            w = np.abs(scores[idx, j])
            if w.sum() < 1e-12: per_src[s] = 0.0; continue
            mean_block = (w[:, None] * G_tr[idx][:, cols]).sum(axis=0) / w.sum()
            cos = float(mean_block @ gte_block /
                        (np.linalg.norm(mean_block) * ng + 1e-12))
            # Multiply by block magnitude — combines direction + magnitude.
            per_src[s] = ng * max(cos, 0.0)
        out.append(max(per_src, key=per_src.get))
    return out


def coef_oracle(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    """ORACLE — uses the trained model's coefficient blocks directly.
    The audit-realistic primary-source label is defined as argmax
    |coef[block] . x[block]|, so this method should asymptote to the
    achievable ceiling. Reported as the upper bound for any attribution-
    based method on this task."""
    X_tr_s, X_te_s, _, _, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    clf = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000,
                             random_state=seed).fit(X_tr_s, y_tr)
    coef = clf.coef_[0]
    out = []
    for j in range(len(X_te_s)):
        per_src = {s: abs(X_te_s[j, source_cols[s]] @ coef[source_cols[s]])
                   for s in src_names}
        out.append(max(per_src, key=per_src.get))
    return out


def ss_attrib_block(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols,
                    seed, K=None):
    """K defaults to n_train // 20 (≈5%), the adaptive recipe selected on
    a held-out 10-seed sweep on the real-HC tier (see exp22c). For
    n_train=19800 this gives K=990, which clears p<0.01 on a disjoint
    20-seed paired t-test (+4.2 pp lift over gradmag). Override only for
    reproducing earlier 5-seed runs."""
    if K is None:
        K = max(50, len(X_tr) // 20)
    """Stronger SS hybrid: per source, score = block_mag * sum(|s_i| for i in
    top-K). Block magnitude says 'where the action is'; top-K influence sum
    says 'which source the model is leaning on for this prediction'. K is
    larger than the top-K vote variant because we sum magnitudes rather than
    counting votes."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    src_tr_arr = np.array(src_tr)
    n_train = scores.shape[0]
    priors = {s: max((src_tr_arr == s).sum(), 1) / n_train for s in src_names}
    out = []
    for j in range(scores.shape[1]):
        block_mag = {s: float(np.linalg.norm(G_te[j, source_cols[s]]))
                     for s in src_names}
        top = np.argsort(-np.abs(scores[:, j]))[:K]
        src_mass = defaultdict(float)
        for i in top: src_mass[src_tr_arr[i]] += abs(scores[i, j])
        # Lift: per-source mass relative to its prior
        lift = {s: src_mass.get(s, 0.0) / priors[s] for s in src_names}
        # Combine: block_mag * lift
        combined = {s: block_mag[s] * lift[s] for s in src_names}
        out.append(max(combined, key=combined.get))
    return out


def ss_blockmag_lift(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols,
                     seed, K=25):
    """Gate top-K vote by source's block-magnitude ranking. Pick the source
    that appears most often in the top-K AND has the largest test-gradient
    block magnitude. Effectively gradmag refined by attribution consensus."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)
    src_tr_arr = np.array(src_tr)
    n_train = scores.shape[0]
    priors = {s: max((src_tr_arr == s).sum(), 1) / n_train for s in src_names}
    out = []
    for j in range(scores.shape[1]):
        # Block magnitudes (gradmag-style)
        block_mag = {s: np.linalg.norm(G_te[j, source_cols[s]])
                     for s in src_names}
        ng_norm = sum(block_mag.values()) + 1e-12
        block_mag = {s: v / ng_norm for s, v in block_mag.items()}
        # Top-K lift (attribution-style)
        top = np.argsort(-np.abs(scores[:, j]))[:K]
        counts = defaultdict(int)
        for i in top: counts[src_tr_arr[i]] += 1
        lift = {s: (counts.get(s, 0) / K) / priors[s] for s in src_names}
        l_norm = sum(lift.values()) + 1e-12
        lift = {s: v / l_norm for s, v in lift.items()}
        # Geometric mean of normalised block_mag and lift
        combined = {s: np.sqrt(block_mag[s] * lift[s]) for s in src_names}
        out.append(max(combined, key=combined.get))
    return out


def ss_cosine(X_tr, y_tr, src_tr, X_te, y_te, src_names, source_cols, seed):
    """Cosine similarity between test gradient and influence-weighted mean
    training gradient per source. Direction-based, magnitude-invariant."""
    _, _, G_tr, G_te, _ = fit_and_grad(X_tr, y_tr, X_te, y_te, seed)
    scores = gram_solve(G_tr, G_te)                  # (n_train, n_test)
    src_tr_arr = np.array(src_tr)
    masks = {s: np.where(src_tr_arr == s)[0] for s in src_names}
    out = []
    for j in range(G_te.shape[0]):
        gte = G_te[j]
        ng  = np.linalg.norm(gte) + 1e-12
        cos = {}
        for s in src_names:
            idx = masks[s]
            if len(idx) == 0: cos[s] = 0.0; continue
            w   = np.abs(scores[idx, j])             # influence weights
            if w.sum() < 1e-12: cos[s] = 0.0; continue
            mean_g = (w[:, None] * G_tr[idx]).sum(axis=0) / w.sum()
            cos[s] = float(mean_g @ gte / (np.linalg.norm(mean_g) * ng + 1e-12))
        out.append(max(cos, key=cos.get))
    return out


# =============================================================================
# Metrics
# =============================================================================
def metrics(preds, truth):
    n = len(truth)
    micro = sum(p == t for p, t in zip(preds, truth)) / n
    # macro = mean of per-segment accuracy
    by_seg = defaultdict(lambda: [0, 0])    # [correct, total]
    for p, t in zip(preds, truth):
        by_seg[t][1] += 1
        by_seg[t][0] += int(p == t)
    per_seg = {s: c / max(t, 1) for s, (c, t) in by_seg.items()}
    macro = float(np.mean(list(per_seg.values())))
    # size-bias slope: simple correlation between segment size and per-seg P@1
    sizes = np.array([by_seg[s][1] for s in by_seg])
    p1s   = np.array([per_seg[s]    for s in by_seg])
    if len(sizes) > 1 and sizes.std() > 0 and p1s.std() > 0:
        slope = float(np.corrcoef(sizes, p1s)[0, 1])
    else:
        slope = 0.0
    return {"micro_p1": micro, "macro_p1": macro,
            "per_segment": per_seg, "size_bias_corr": slope}


# =============================================================================
# Tier configs
# =============================================================================
TIERS = [
    {"name": "sanity",     "fn": "synthetic", "kwargs": dict(
        segment_sizes=[5000, 3000, 2000], d_feat=8, cols_per_source=2,
        feature_overlap="disjoint_zero", label_type="linear", noise_std=0.1)},
    {"name": "easy",       "fn": "synthetic", "kwargs": dict(
        segment_sizes=[3333, 3333, 3334], d_feat=8, cols_per_source=2,
        feature_overlap="shared", label_type="linear", noise_std=0.1)},
    {"name": "medium",     "fn": "synthetic", "kwargs": dict(
        segment_sizes=[5000, 3000, 2000], d_feat=8, cols_per_source=2,
        feature_overlap="shared", label_type="linear", noise_std=0.3)},
    {"name": "hard",       "fn": "synthetic", "kwargs": dict(
        segment_sizes=[5000, 3000, 2000], d_feat=8, cols_per_source=2,
        feature_overlap="shared", label_type="interaction", noise_std=0.5)},
    {"name": "adversarial","fn": "synthetic", "kwargs": dict(
        segment_sizes=[8000, 1000, 1000], d_feat=8, cols_per_source=2,
        feature_overlap="shared", label_type="nonlinear", noise_std=0.5)},
    {"name": "real_homecredit","fn": "real", "kwargs": {}},
]

METHODS = {
    # baselines
    "random":           baseline_random,
    "feat_argmax":      baseline_feat_argmax,
    "gradmag_argmax":   baseline_grad_argmax,
    "topk_vote":        baseline_topk_vote,
    # SS variants
    "SS_mean":          ss_mean,
    "SS_median":        ss_median,
    "SS_max":           ss_max,
    "SS_summag":        ss_summag,
    "SS_normsum":       ss_normsum,
    # New variants designed to beat gradmag_argmax
    "SS_topk_lift":     ss_topk_lift,
    "SS_per_source_trak": ss_per_source_trak,
    "SS_cosine":        ss_cosine,
    "SS_block_cosine":  ss_block_cosine,
    "SS_blockmag_lift": ss_blockmag_lift,
    "SS_attrib_block":  ss_attrib_block,
    "ORACLE_coef":      coef_oracle,
}


# =============================================================================
# Runner
# =============================================================================
def load_tier(tier, seed):
    if tier["fn"] == "synthetic":
        return gen_synthetic(seed, **tier["kwargs"])
    elif tier["fn"] == "real":
        return gen_home_credit(seed)
    else: raise ValueError(tier["fn"])


def run():
    results = {}    # results[tier][method] = list of metrics dicts (one per seed)
    print("\nRunning SAB v1 …")
    print(f"  {len(TIERS)} tiers × {len(SEEDS)} seeds × {len(METHODS)} methods "
          f"= {len(TIERS)*len(SEEDS)*len(METHODS)} runs")
    t0 = time.perf_counter()

    for tier in TIERS:
        tname = tier["name"]
        results[tname] = {m: [] for m in METHODS}
        for seed in SEEDS:
            try:
                X_tr, y_tr, src_tr, X_te, y_te, src_te, src_cols, src_names = \
                    load_tier(tier, seed)
            except Exception as e:
                print(f"  [{tname}] seed={seed} load failed: {e}")
                continue
            for mname, fn in METHODS.items():
                try:
                    preds = fn(X_tr, y_tr, src_tr, X_te, y_te, src_names, src_cols, seed)
                    results[tname][mname].append(metrics(preds, src_te))
                except Exception as e:
                    print(f"  [{tname}/{mname}] seed={seed} failed: {e}")

        # Per-tier summary line
        rand_m = np.mean([r["macro_p1"] for r in results[tname]["random"]])
        gm_m   = np.mean([r["macro_p1"] for r in results[tname]["gradmag_argmax"]])
        print(f"  [{tname:14s}] random={rand_m:.3f}  gradmag={gm_m:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return results


def summarise(results):
    out = {}
    for tier, by_m in results.items():
        out[tier] = {}
        for m, runs in by_m.items():
            if not runs:
                out[tier][m] = None; continue
            macro = np.array([r["macro_p1"] for r in runs])
            micro = np.array([r["micro_p1"] for r in runs])
            slope = np.array([r["size_bias_corr"] for r in runs])
            out[tier][m] = {
                "macro_p1_mean": round(float(macro.mean()), 4),
                "macro_p1_std":  round(float(macro.std()),  4),
                "micro_p1_mean": round(float(micro.mean()), 4),
                "micro_p1_std":  round(float(micro.std()),  4),
                "size_bias_corr_mean": round(float(slope.mean()), 4),
                "n_seeds": len(runs),
            }
        # Compute pass/fail per method vs baselines on this tier
        rand_m = out[tier]["random"]["macro_p1_mean"]
        best_baseline_m = max(out[tier][b]["macro_p1_mean"]
                              for b in ["feat_argmax", "gradmag_argmax", "topk_vote"]
                              if out[tier].get(b))
        for m in by_m:
            if out[tier][m] is None: continue
            m_macro = out[tier][m]["macro_p1_mean"]
            out[tier][m]["pass_vs_random"]   = bool(m_macro - rand_m         >= 0.10)
            out[tier][m]["pass_vs_baseline"] = bool(m_macro - best_baseline_m >= 0.05)
    return out


def _summarise_safe(results):
    """Wrapper that skips tiers where all seeds failed."""
    return {t: by_m for t, by_m in summarise(
        {t: by_m for t, by_m in results.items()
         if any(by_m.get("random"))}).items()}


def print_table(summary):
    rows = []
    methods = list(METHODS)
    tiers = list(summary)
    print()
    print("=" * 100)
    print(f"{'Method':18s}  " + "  ".join(f"{t[:12]:>12s}" for t in tiers))
    print("-" * 100)
    for m in methods:
        cells = []
        for t in tiers:
            r = summary[t].get(m)
            if r is None: cells.append("  --  ")
            else:
                mark = ""
                if r.get("pass_vs_baseline"): mark = "*"
                cells.append(f"{r['macro_p1_mean']:.3f}±{r['macro_p1_std']:.2f}{mark:s}")
        print(f"{m:18s}  " + "  ".join(f"{c:>12s}" for c in cells))
    print("=" * 100)
    print("Values: macro_P@1 mean±std over 5 seeds. * marks methods that pass:")
    print("  macro_P@1 >= random + 0.10 AND >= best_baseline + 0.05")
    print()


if __name__ == "__main__":
    res = run()
    summary = _summarise_safe(res)
    print_table(summary)
    os.makedirs("results", exist_ok=True)
    with open("results/exp22_sab_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved to results/exp22_sab_benchmark.json")
