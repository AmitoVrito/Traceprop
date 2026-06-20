"""Q1 cost-model fit and predict — addresses reviewer W1.

Fits the cost-model constants for plans P_0 (scan-driven, no indexes)
and P_2 (covering-index-driven, both indexes) by measuring Q1 latency
at a grid of (|G|, |L|, |S|) configurations, then uses the fitted
model to predict latency at unseen larger scales and compares.

Model:
  cost(P_0) = c_seq * |G|  + c_pk * |G|  + c_hash * |G|
            = (c_seq + c_pk + c_hash) * |G|
            (linear in |G|, independent of |L| at constant |G|)

  cost(P_2) = c_idx_l * |L| * sigma                       (drive from L)
            + c_idx_g * (|G|/|L|) * |L| * sigma           (probe G)
            + c_tree * |S| * log(|S|)                     (group-by)

With |L| > 0, fixed sigma=1 (NOT NULL filter selectivity ~ 1.0 in our
setup), and |S| small, the dominant terms are the linear ones in |L|
and |G|. We therefore fit
  P_0:  t = A_0 + B_0 * |G|                  (assumes |L| <= |G|)
  P_2:  t = A_2 + B_2 * |L| + C_2 * |G|
via least squares, report the fits, and use them to predict latency at
|G| = 10^6 (one decade beyond the training grid). Test the prediction
on a held-out (10^6, 10^5, 20) configuration.
"""
import json, os, sqlite3, time
import numpy as np

N_TRIALS_PER_CFG = 10
TRAIN_GRID = [
    # (n_grad, n_nodes, n_sources)
    (10_000,   10_000,   5),
    (10_000,   10_000,  20),
    (10_000,  100_000,  20),
    (50_000,   50_000,  20),
    (50_000,  100_000,  20),
    (100_000,  10_000,  20),
    (100_000, 100_000,   5),
    (100_000, 100_000,  20),
    (100_000, 100_000, 100),
    (500_000, 100_000,  20),
]
HELDOUT_GRID = [
    (1_000_000, 100_000, 20),    # 10x prediction
    (1_000_000, 500_000, 20),    # different |L|
    (250_000,  250_000,  50),    # in-grid sanity
]

Q1 = """
SELECT ln.source_file, SUM(gs.influence_score) AS agg_inf
FROM   gradient_store gs
JOIN   lineage_nodes  ln ON gs.source_node_id = ln.node_id
WHERE  gs.influence_score IS NOT NULL
  AND  ln.source_file IS NOT NULL
GROUP BY ln.source_file
ORDER BY agg_inf DESC
LIMIT 10;
"""

def setup_db(db_path, n_grad, n_nodes, n_sources, with_indexes):
    if os.path.exists(db_path): os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("CREATE TABLE lineage_nodes (node_id INTEGER PRIMARY KEY, "
                "source_file TEXT, source_rows TEXT)")
    cur.execute("CREATE TABLE gradient_store (sample_idx INTEGER PRIMARY KEY, "
                "influence_score REAL, source_node_id INTEGER)")
    rng = np.random.default_rng(42)
    sources = [f"src_{i}.csv" for i in range(n_sources)]
    cur.executemany("INSERT INTO lineage_nodes VALUES (?,?,?)",
                    [(i, sources[i % n_sources], str(i)) for i in range(n_nodes)])
    gs = rng.integers(0, n_nodes, n_grad)
    inf = rng.standard_normal(n_grad).astype(float)
    cur.executemany("INSERT INTO gradient_store VALUES (?,?,?)",
                    [(i, float(inf[i]), int(gs[i])) for i in range(n_grad)])
    if with_indexes:
        cur.execute("CREATE INDEX idx_gs_src_node ON gradient_store(source_node_id)")
        cur.execute("CREATE INDEX idx_ln_src_file ON lineage_nodes(source_file) "
                    "WHERE source_file IS NOT NULL")
    cur.execute("ANALYZE")
    con.commit()
    return con

def time_q1(con, n_trials):
    cur = con.cursor()
    for _ in range(3): cur.execute(Q1).fetchall()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter(); cur.execute(Q1).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

def measure_grid(grid, label):
    print(f"\n=== {label} ===")
    out = []
    for ng, nn, ns in grid:
        # Skip rare cases where |L| is unrealistically large
        if nn > 10 * ng: continue
        for with_idx in (False, True):
            db = f"/tmp/exp24b_{label}_{ng}_{nn}_{ns}_{int(with_idx)}.db"
            con = setup_db(db, ng, nn, ns, with_idx)
            ms = time_q1(con, N_TRIALS_PER_CFG)
            con.close(); os.remove(db)
            plan = "P_2" if with_idx else "P_0"
            out.append({"plan": plan, "n_grad": ng, "n_nodes": nn,
                        "n_sources": ns, "latency_ms": round(ms, 3)})
            print(f"  {plan} |G|={ng:>7,d} |L|={nn:>7,d} |S|={ns:>3d}  "
                  f"{ms:>8.2f} ms")
    return out

train_rows = measure_grid(TRAIN_GRID, "train")

# Fit cost models via least squares
def fit_p0(rows):
    r = [x for x in rows if x["plan"] == "P_0"]
    G = np.array([x["n_grad"]   for x in r], dtype=float)
    y = np.array([x["latency_ms"] for x in r], dtype=float)
    # t = A + B * |G|
    X = np.column_stack([np.ones_like(G), G])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {"A": float(beta[0]), "B_per_G": float(beta[1]),
            "rmse_ms": round(rmse, 3), "n_points": len(r)}

def fit_p2(rows):
    r = [x for x in rows if x["plan"] == "P_2"]
    G = np.array([x["n_grad"]    for x in r], dtype=float)
    L = np.array([x["n_nodes"]   for x in r], dtype=float)
    y = np.array([x["latency_ms"] for x in r], dtype=float)
    # t = A + B * |L| + C * |G|
    X = np.column_stack([np.ones_like(G), L, G])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {"A": float(beta[0]), "B_per_L": float(beta[1]),
            "C_per_G": float(beta[2]),
            "rmse_ms": round(rmse, 3), "n_points": len(r)}

p0_fit = fit_p0(train_rows)
p2_fit = fit_p2(train_rows)
print()
print("=" * 70)
print("Fitted cost models")
print(f"  P_0:  t (ms) = {p0_fit['A']:+.3f} + "
      f"{p0_fit['B_per_G']*1e6:+.3f} * |G| / 10^6")
print(f"         training-RMSE = {p0_fit['rmse_ms']:.2f} ms over "
      f"{p0_fit['n_points']} configs")
print(f"  P_2:  t (ms) = {p2_fit['A']:+.3f} + "
      f"{p2_fit['B_per_L']*1e6:+.3f} * |L|/10^6 + "
      f"{p2_fit['C_per_G']*1e6:+.3f} * |G|/10^6")
print(f"         training-RMSE = {p2_fit['rmse_ms']:.2f} ms over "
      f"{p2_fit['n_points']} configs")

def predict_p0(ng):    return p0_fit["A"] + p0_fit["B_per_G"] * ng
def predict_p2(ng, nn): return (p2_fit["A"] + p2_fit["B_per_L"] * nn
                                + p2_fit["C_per_G"] * ng)

print("\n=== Held-out predictions ===")
holdout_rows = measure_grid(HELDOUT_GRID, "holdout")
holdout_eval = []
for ng, nn, ns in HELDOUT_GRID:
    measured_p0 = next((x["latency_ms"] for x in holdout_rows
                       if x["plan"] == "P_0" and x["n_grad"] == ng
                       and x["n_nodes"] == nn and x["n_sources"] == ns), None)
    measured_p2 = next((x["latency_ms"] for x in holdout_rows
                       if x["plan"] == "P_2" and x["n_grad"] == ng
                       and x["n_nodes"] == nn and x["n_sources"] == ns), None)
    pred_p0 = predict_p0(ng)
    pred_p2 = predict_p2(ng, nn)
    holdout_eval.append({
        "n_grad": ng, "n_nodes": nn, "n_sources": ns,
        "P_0_measured_ms":  measured_p0,
        "P_0_predicted_ms": round(pred_p0, 2),
        "P_0_abs_error_pct": (round(abs(pred_p0 - measured_p0) / measured_p0 * 100, 2)
                              if measured_p0 else None),
        "P_2_measured_ms":  measured_p2,
        "P_2_predicted_ms": round(pred_p2, 2),
        "P_2_abs_error_pct": (round(abs(pred_p2 - measured_p2) / measured_p2 * 100, 2)
                              if measured_p2 else None),
        "measured_speedup": (round(measured_p0 / measured_p2, 2)
                             if measured_p0 and measured_p2 else None),
        "predicted_speedup": round(pred_p0 / pred_p2, 2),
    })
    print(f"  |G|={ng:>7,d} |L|={nn:>7,d} |S|={ns:>3d}  "
          f"P_0 pred={pred_p0:>7.2f} measured={measured_p0 or 0:>7.2f} "
          f"P_2 pred={pred_p2:>7.2f} measured={measured_p2 or 0:>7.2f} "
          f"speedup_pred={pred_p0/pred_p2:.2f}x")

out = {
    "experiment": "exp24b_q1_cost_model",
    "train_grid": TRAIN_GRID,
    "heldout_grid": HELDOUT_GRID,
    "n_trials_per_cfg": N_TRIALS_PER_CFG,
    "train_measurements": train_rows,
    "p0_fit": p0_fit,
    "p2_fit": p2_fit,
    "heldout_measurements": holdout_rows,
    "heldout_prediction_eval": holdout_eval,
}
os.makedirs("results", exist_ok=True)
with open("results/exp24b_q1_cost_model.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp24b_q1_cost_model.json")
