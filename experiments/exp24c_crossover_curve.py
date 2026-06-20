"""Q1 cross-over curve — fills in intermediate |G| between the sweet
spot (10^5) and the cross-over point predicted at 10^6.

Sweeps |G| in {10^5, 2e5, 3e5, 5e5, 7e5, 10^6} at fixed |L|=10^5,
|S|=20 and measures P_0 and P_2 latency at each point. The curves
should cross between 10^5 and 10^6.
"""
import json, os, sqlite3, time
import numpy as np

SWEEP = [100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000]
N_LINEAGE = 100_000
N_SOURCES = 20
N_TRIALS  = 10

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

def setup(db, ng, nl, ns, with_idx):
    if os.path.exists(db): os.remove(db)
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("CREATE TABLE lineage_nodes (node_id INTEGER PRIMARY KEY, source_file TEXT, source_rows TEXT)")
    cur.execute("CREATE TABLE gradient_store (sample_idx INTEGER PRIMARY KEY, influence_score REAL, source_node_id INTEGER)")
    rng = np.random.default_rng(42)
    srcs = [f"src_{i}.csv" for i in range(ns)]
    cur.executemany("INSERT INTO lineage_nodes VALUES (?,?,?)",
                    [(i, srcs[i % ns], str(i)) for i in range(nl)])
    gs = rng.integers(0, nl, ng); inf = rng.standard_normal(ng).astype(float)
    cur.executemany("INSERT INTO gradient_store VALUES (?,?,?)",
                    [(i, float(inf[i]), int(gs[i])) for i in range(ng)])
    if with_idx:
        cur.execute("CREATE INDEX i_gs ON gradient_store(source_node_id)")
        cur.execute("CREATE INDEX i_ln ON lineage_nodes(source_file) WHERE source_file IS NOT NULL")
    cur.execute("ANALYZE"); con.commit()
    return con

def time_q1(con, n):
    c = con.cursor()
    for _ in range(3): c.execute(Q1).fetchall()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); c.execute(Q1).fetchall()
        t.append((time.perf_counter()-t0)*1000)
    return float(np.median(t))

print(f"=== Cross-over curve (|L|={N_LINEAGE}, |S|={N_SOURCES}) ===")
print(f"{'|G|':>10}  {'P_0 ms':>9}  {'P_2 ms':>9}  {'P_2/P_0':>8}")
rows = []
for ng in SWEEP:
    p0 = time_q1(setup(f"/tmp/cv_p0_{ng}.db", ng, N_LINEAGE, N_SOURCES, False), N_TRIALS)
    p2 = time_q1(setup(f"/tmp/cv_p2_{ng}.db", ng, N_LINEAGE, N_SOURCES, True),  N_TRIALS)
    ratio = p2 / p0
    rows.append({"n_grad": ng, "p0_ms": round(p0, 2),
                 "p2_ms": round(p2, 2), "p2_over_p0": round(ratio, 3)})
    print(f"  {ng:>10,d}  {p0:>9.2f}  {p2:>9.2f}  {ratio:>8.3f}")

# Where does the crossover happen?
cross = None
for i in range(len(rows) - 1):
    if rows[i]["p2_over_p0"] < 1 and rows[i+1]["p2_over_p0"] > 1:
        # Linear interp in log|G|
        x0, x1 = np.log10(rows[i]["n_grad"]), np.log10(rows[i+1]["n_grad"])
        y0, y1 = rows[i]["p2_over_p0"], rows[i+1]["p2_over_p0"]
        cross = 10 ** (x0 + (1 - y0) * (x1 - x0) / (y1 - y0))
        break
if cross is not None:
    print(f"\nP_0 = P_2 cross-over at |G| ≈ {cross:,.0f}")
else:
    print("\nNo cross-over observed in the swept range")

os.makedirs("results", exist_ok=True)
out = {"sweep": rows,
       "n_lineage": N_LINEAGE, "n_sources": N_SOURCES,
       "crossover_n_grad": cross}
with open("results/exp24c_crossover_curve.json","w") as f:
    json.dump(out, f, indent=2)
print("Saved.")
