"""Multi-seed unlearning with variance bars — addresses VLDB reviewer #8.

Reviewer: "Table 8 is the only major results table with no error bars —
conspicuous in an otherwise statistics-careful paper, especially with
forget sets of 50/50/500 where seed variance should be large."

We rerun exp15 (Adult Income) and exp16 (Covertype) across 10 seeds each
to report mean ± std for gap-closed and test accuracy.
"""
import json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml, fetch_covtype

N_SEEDS = 10


def gen_adult():
    ds = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    import pandas as pd
    X = pd.get_dummies(ds.data, drop_first=True).astype(np.float64).to_numpy()
    y = (ds.target == ">50K").astype(np.float32).to_numpy()
    return X, y, 6000, 50, 0.1, 3   # n, forget_k, eta, n_steps_tuned

def gen_covertype():
    ds = fetch_covtype()
    X = ds.data.astype(np.float64)
    y = (ds.target == 2).astype(np.float32)
    return X, y, 50000, 500, 0.05, 2


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def run_one_seed(X_full, y_full, n_train, forget_k, eta, n_steps_tuned, seed,
                 C=100.0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), n_train + 500, replace=False)
    X = X_full[idx[:n_train]]; y = y_full[idx[:n_train]]
    X_te = X_full[idx[n_train:]]; y_te = y_full[idx[n_train:]]

    scaler = StandardScaler()
    X    = scaler.fit_transform(X).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                             random_state=seed).fit(X, y)
    coef = clf.coef_[0]; b = clf.intercept_[0]

    p = sigmoid(X @ coef + b)
    err = p - y
    G = (err[:, None] * X).astype(np.float32)
    influence = (G * G.sum(axis=0, keepdims=True)).sum(axis=1)
    forget_idx = np.argsort(-np.abs(influence))[:forget_k]

    test_acc_orig = float(((p > 0.5) == y).mean())   # use train as proxy

    def forget_loss(coef_, b_):
        p_f = sigmoid(X[forget_idx] @ coef_ + b_)
        return float(np.mean(-y[forget_idx]*np.log(p_f+1e-12)
                             -(1-y[forget_idx])*np.log(1-p_f+1e-12)))

    L_orig = forget_loss(coef, b)
    test_acc_clf = float(clf.score(X_te, y_te))

    # Gold: retrain without forget set
    mask = np.ones(n_train, dtype=bool); mask[forget_idx] = False
    clf_gold = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                                  random_state=seed).fit(X[mask], y[mask])
    L_gold = forget_loss(clf_gold.coef_[0], clf_gold.intercept_[0])
    test_acc_gold = float(clf_gold.score(X_te, y_te))

    def gradient_correction(coef_, b_, steps):
        w = coef_.copy(); bb = b_
        for _ in range(steps):
            p_f = sigmoid(X[forget_idx] @ w + bb)
            err_f = p_f - y[forget_idx]
            g  = (err_f[:, None] * X[forget_idx]).mean(axis=0)
            gb = err_f.mean()
            w  = w + eta * g; bb = bb + eta * gb
        return w, bb

    # Random baseline: same n_steps_tuned, random forget set
    rng2 = np.random.default_rng(seed + 999)
    random_idx = rng2.choice(n_train, forget_k, replace=False)
    def gradient_correction_random(steps):
        w = coef.copy(); bb = b
        for _ in range(steps):
            p_f = sigmoid(X[random_idx] @ w + bb)
            err_f = p_f - y[random_idx]
            g  = (err_f[:, None] * X[random_idx]).mean(axis=0)
            gb = err_f.mean()
            w = w + eta * g; bb = bb + eta * gb
        return w, bb

    # 5-step (over-correction)
    w5, b5 = gradient_correction(coef, b, 5)
    L5 = forget_loss(w5, b5)
    clf_t = LogisticRegression(C=C, max_iter=1, solver="lbfgs",
                               warm_start=True, random_state=seed)
    clf_t.fit(X, y); clf_t.coef_[:] = w5; clf_t.intercept_[:] = b5
    test_acc_5 = float(clf_t.score(X_te, y_te))

    # Tuned step count
    wt, bt = gradient_correction(coef, b, n_steps_tuned)
    Lt = forget_loss(wt, bt)
    clf_t.coef_[:] = wt; clf_t.intercept_[:] = bt
    test_acc_t = float(clf_t.score(X_te, y_te))

    # Random
    wr, br = gradient_correction_random(n_steps_tuned)
    Lr = forget_loss(wr, br)

    def gap_closed(L):
        return 100.0 * (L - L_orig) / (L_gold - L_orig + 1e-12)

    return {
        "seed": seed,
        "L_orig":  round(L_orig, 4),
        "L_gold":  round(L_gold, 4),
        "L_tuned": round(Lt, 4),
        "L_5step": round(L5, 4),
        "L_random":round(Lr, 4),
        "gap_tuned":  round(gap_closed(Lt), 1),
        "gap_5step":  round(gap_closed(L5), 1),
        "gap_random": round(gap_closed(Lr), 1),
        "test_acc_gold":  round(test_acc_gold, 4),
        "test_acc_tuned": round(test_acc_t, 4),
        "test_acc_5step": round(test_acc_5, 4),
    }


results = {}
for name, gen in [("adult", gen_adult), ("covertype", gen_covertype)]:
    print(f"\n=== {name} ===")
    X, y, n_train, fk, eta, nst = gen()
    rows = []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()
        r = run_one_seed(X, y, n_train, fk, eta, nst, seed)
        rows.append(r)
        print(f"  seed {seed}  gap_tuned={r['gap_tuned']}%  "
              f"gap_5step={r['gap_5step']}%  test_acc_tuned={r['test_acc_tuned']}  "
              f"[{time.perf_counter()-t0:.1f}s]")

    def stats(key):
        vals = np.array([r[key] for r in rows])
        return round(float(vals.mean()), 3), round(float(vals.std()), 3)

    summary = {
        "n_train": n_train, "forget_k": fk, "eta": eta,
        "n_steps_tuned": nst,
        "gap_tuned_mean_std":  stats("gap_tuned"),
        "gap_5step_mean_std":  stats("gap_5step"),
        "gap_random_mean_std": stats("gap_random"),
        "test_acc_tuned_mean_std": stats("test_acc_tuned"),
        "test_acc_5step_mean_std": stats("test_acc_5step"),
        "test_acc_gold_mean_std":  stats("test_acc_gold"),
        "per_seed": rows,
    }
    results[name] = summary
    print(f"  gap_tuned   {summary['gap_tuned_mean_std'][0]}% ± "
          f"{summary['gap_tuned_mean_std'][1]}%")
    print(f"  gap_5step   {summary['gap_5step_mean_std'][0]}% ± "
          f"{summary['gap_5step_mean_std'][1]}%")
    print(f"  gap_random  {summary['gap_random_mean_std'][0]}% ± "
          f"{summary['gap_random_mean_std'][1]}%")

os.makedirs("results", exist_ok=True)
with open("results/exp19b_unlearning_multiseed.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/exp19b_unlearning_multiseed.json")
