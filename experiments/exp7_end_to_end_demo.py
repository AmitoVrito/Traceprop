"""Experiment 7: End-to-End trace_to_file() demonstration."""

import numpy as np
import tempfile
import os
import traceprop as tp
from traceprop.graph import reset_graph

print("=== Traceprop End-to-End Demo ===\n")

# 1. Create realistic CSVs
reset_graph()
np.random.seed(42)

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("age,income,credit_score\n")
    for i in range(200):
        f.write(f"{np.random.randint(25,65)},{np.random.randint(30000,120000)},{np.random.randint(550,850)}\n")
    csv_path_A = f.name

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("age,income,credit_score\n")
    for i in range(100):
        f.write(f"{np.random.randint(25,65)},{np.random.randint(30000,120000)},{np.random.randint(550,850)}\n")
    csv_path_B = f.name

# 2. Load with source tagging
source_A = tp.from_csv(csv_path_A, source_id="bank_A_customers", delimiter=",", skip_header=True)
source_B = tp.from_csv(csv_path_B, source_id="bank_B_customers", delimiter=",", skip_header=True)

print(f"Loaded source_A: {source_A.shape}")
print(f"Loaded source_B: {source_B.shape}\n")

# 3. Preprocessing (all tracked)
combined = tp.from_numpy(
    np.vstack([np.asarray(source_A), np.asarray(source_B)]),
    source_id="combined_training",
)

mean = combined.mean(axis=0)
std = combined.std(axis=0)
normalized = (combined - mean) / (std + 1e-8)

view = tp.provenance(normalized)
print(f"Preprocessing tracked: {len(view.ops())} ops recorded")
print(f"Source IDs in path: {view.source_ids_in_path()}\n")

# 4. Simulate training + gradient logging
np.random.seed(42)
ctx = tp.training_context(source_id="combined_training", proj_dim=64)

for i in range(len(np.asarray(combined))):
    sample_grad = np.asarray(normalized)[i] * np.array([1.0, -0.5, 0.8])
    source_id = "bank_A_customers" if i < 200 else "bank_B_customers"
    ctx.log_gradient(
        gradient=sample_grad,
        sample_index=i,
        source_node_id=view.node_id,
    )

print(f"Logged {len(ctx.gradient_store)} training gradients\n")

# 5. Attribution on a test prediction
test_grad = np.array([0.3, -0.1, 0.8])
engine = tp.attribution_engine(ctx.gradient_store)
result = engine.attribute(test_grad, top_k=5)

print("Top 5 most influential training samples:")
for entry in result.top(5):
    print(f"  Rank {entry['rank']}: sample_index={entry['sample_index']}, "
          f"score={entry['influence_score']:.4f}, "
          f"source={entry.get('source_id', 'unknown')}")

# 6. trace_to_file
print("\nEnd-to-end trace for rank-0 influential sample:")
trace = result.trace_to_file(rank=0)
print(f"  Influence score: {trace.get('influence_score', 'N/A')}")
print(f"  Sources: {trace.get('sources', {})}")
print(f"  Ops count: {len(trace.get('ops', []))}\n")

# 7. Compliance report
print("Generating EU AI Act compliance report...")
report = tp.compliance_report(
    normalized,
    system_name="CreditScoringModel_v3",
    system_version="3.0.0",
    deployer_name="Acme Bank GmbH",
    high_risk_category="Annex III, A.5 — Employment and credit scoring",
)
if report:
    print(f"  Report keys: {list(report.keys())}")
    dg = report.get("data_governance", {})
    print(f"  Sources documented: {dg.get('total_input_sources', 'N/A')}")
    print(f"  Transformation steps: {dg.get('data_transformation_steps', 'N/A')}")

# Cleanup
os.unlink(csv_path_A)
os.unlink(csv_path_B)

print("\n=== Demo Complete ===")
