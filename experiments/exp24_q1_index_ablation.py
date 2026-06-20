"""Q1 query-processing ablation — index selection for the
trace-to-source query (Sec 4 of the SIGMOD draft).

Q1 in SQL:
    SELECT ln.source_file, ln.source_rows,
           SUM(gs.influence_score) AS agg_inf
    FROM   gradient_store gs
    JOIN   lineage_nodes  ln ON gs.source_node_id = ln.node_id
    WHERE  gs.influence_score IS NOT NULL
      AND  ln.source_file IS NOT NULL
    GROUP BY ln.source_file, ln.source_rows
    ORDER BY agg_inf DESC
    LIMIT 10;

We measure end-to-end Q1 latency under four index configurations on a
SQLite LineageStore loaded with N gradient entries joining to a
LineageGraph of M nodes spanning S source files. Configurations:
  cfg0: no indexes (baseline)
  cfg1: idx on gs.source_node_id (the join key)
  cfg2: idx on ln.source_file (the predicate column)
  cfg3: cfg1 + cfg2 (both indexes)

Reports mean ± std over 30 trials per configuration.
"""
import json, os, sqlite3, time
import numpy as np

N_GRAD     = 100_000     # gradient store rows
N_NODES    = 100_000     # lineage nodes (one per training sample + source rollups)
N_SOURCES  = 20          # number of source_file values
N_TRIALS   = 30

def setup_db(db_path, with_idx_join, with_idx_source):
    if os.path.exists(db_path): os.remove(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""CREATE TABLE lineage_nodes (
        node_id INTEGER PRIMARY KEY,
        source_file TEXT,
        source_rows TEXT
    )""")
    cur.execute("""CREATE TABLE gradient_store (
        sample_idx INTEGER PRIMARY KEY,
        influence_score REAL,
        source_node_id INTEGER
    )""")
    rng = np.random.default_rng(42)
    sources = [f"source_{i}.csv" for i in range(N_SOURCES)]
    # populate lineage_nodes: each has a source_file
    rows = [(i, sources[i % N_SOURCES], str(i)) for i in range(N_NODES)]
    cur.executemany("INSERT INTO lineage_nodes VALUES (?,?,?)", rows)
    # gradient_store: random source_node_id pointing into lineage_nodes
    gs = rng.integers(0, N_NODES, N_GRAD)
    inf = rng.standard_normal(N_GRAD).astype(float)
    rows = [(i, float(inf[i]), int(gs[i])) for i in range(N_GRAD)]
    cur.executemany("INSERT INTO gradient_store VALUES (?,?,?)", rows)
    if with_idx_join:
        cur.execute("CREATE INDEX idx_gs_source_node_id ON gradient_store(source_node_id)")
    if with_idx_source:
        cur.execute("CREATE INDEX idx_ln_source_file ON lineage_nodes(source_file) "
                    "WHERE source_file IS NOT NULL")
    cur.execute("ANALYZE")
    con.commit()
    return con

Q1 = """
SELECT ln.source_file,
       SUM(gs.influence_score) AS agg_inf
FROM   gradient_store gs
JOIN   lineage_nodes  ln ON gs.source_node_id = ln.node_id
WHERE  gs.influence_score IS NOT NULL
  AND  ln.source_file IS NOT NULL
GROUP BY ln.source_file
ORDER BY agg_inf DESC
LIMIT 10;
"""

def time_query(con, n_trials):
    times = []
    cur = con.cursor()
    # warm up cache
    for _ in range(3):
        cur.execute(Q1).fetchall()
    for _ in range(n_trials):
        t0 = time.perf_counter()
        rows = cur.execute(Q1).fetchall()
        times.append(time.perf_counter() - t0)
    return times, rows

def explain_query(con):
    cur = con.cursor()
    return cur.execute("EXPLAIN QUERY PLAN " + Q1).fetchall()

configs = [
    ("cfg0_no_indexes",   False, False),
    ("cfg1_join_idx",     True,  False),
    ("cfg2_source_idx",   False, True),
    ("cfg3_both_idx",     True,  True),
]

results = {}
print("=" * 70)
print(f"Q1 latency at N_grad={N_GRAD:,} N_nodes={N_NODES:,} S_files={N_SOURCES}")
print("=" * 70)
for name, idx_join, idx_src in configs:
    db = f"/tmp/exp24_{name}.db"
    con = setup_db(db, idx_join, idx_src)
    times, sample_rows = time_query(con, N_TRIALS)
    plan = explain_query(con)
    con.close(); os.remove(db)
    times = np.array(times) * 1000  # ms
    results[name] = {
        "indexes": {"gs.source_node_id": idx_join, "ln.source_file": idx_src},
        "latency_ms_mean": round(float(times.mean()), 3),
        "latency_ms_std":  round(float(times.std()),  3),
        "latency_ms_min":  round(float(times.min()),  3),
        "n_trials": N_TRIALS,
        "query_plan": [list(r) for r in plan],
        "result_top1": sample_rows[0] if sample_rows else None,
    }
    print(f"  {name:20s}  {times.mean():>7.2f} ± {times.std():>5.2f} ms")
    for step in plan:
        print(f"    plan: {step}")

best  = min(results.values(), key=lambda v: v["latency_ms_mean"])
worst = max(results.values(), key=lambda v: v["latency_ms_mean"])
speedup = worst["latency_ms_mean"] / best["latency_ms_mean"]
print(f"\n  best-vs-worst speedup: {speedup:.1f}x")
print("=" * 70)

out = {
    "experiment": "exp24_q1_index_ablation",
    "n_grad": N_GRAD, "n_nodes": N_NODES, "n_sources": N_SOURCES,
    "n_trials": N_TRIALS,
    "configs": results,
    "best_config":  min(results, key=lambda k: results[k]["latency_ms_mean"]),
    "worst_config": max(results, key=lambda k: results[k]["latency_ms_mean"]),
    "best_over_worst_speedup": round(float(speedup), 2),
}
os.makedirs("results", exist_ok=True)
with open("results/exp24_q1_index_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to results/exp24_q1_index_ablation.json")
