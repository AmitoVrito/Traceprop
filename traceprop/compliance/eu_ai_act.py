"""EU AI Act compliance report generation."""

from __future__ import annotations

import time
from typing import Any

from traceprop.graph import LineageGraph

EU_AI_ACT_LOG_VERSION = "1.0"
EU_AI_ACT_RETENTION_DAYS = 3650  # 10 years per Article 12(1)


def generate_compliance_report(
    node_id: int,
    graph: LineageGraph,
    system_name: str,
    system_version: str,
    deployer_name: str,
    high_risk_category: str,
    output_path: str | None = None,
) -> dict:
    """Generate an EU AI Act Article 12 compliance report.

    Returns a dict containing lineage data, system metadata, and
    compliance information for a given tensor node.
    """
    # Gather lineage
    ancestor_ids = graph.ancestors(node_id)
    node = graph.nodes.get(node_id)

    # Collect edges in the ancestor subgraph
    relevant_edges = []
    all_node_ids = ancestor_ids | {node_id}
    for eid, edge in graph.edges.items():
        if edge.output_id in all_node_ids or any(i in all_node_ids for i in edge.input_ids):
            relevant_edges.append({
                "id": edge.id,
                "op_name": edge.op_name,
                "input_ids": list(edge.input_ids),
                "output_id": edge.output_id,
                "timestamp": edge.timestamp,
                "source_ids": list(edge.source_ids),
            })

    # Collect ancestor nodes
    ancestor_nodes = []
    for nid in ancestor_ids:
        n = graph.nodes.get(nid)
        if n is not None:
            ancestor_nodes.append({
                "id": n.id,
                "shape": list(n.shape),
                "dtype": n.dtype,
                "timestamp": n.timestamp,
                "meta": n.meta,
            })

    report = {
        "log_version": EU_AI_ACT_LOG_VERSION,
        "system_name": system_name,
        "system_version": system_version,
        "deployer_name": deployer_name,
        "high_risk_category": high_risk_category,
        "target_node": {
            "id": node.id if node else node_id,
            "shape": list(node.shape) if node else None,
            "dtype": node.dtype if node else None,
            "timestamp": node.timestamp if node else None,
        },
        "ancestor_count": len(ancestor_ids),
        "ancestors": ancestor_nodes,
        "edges": relevant_edges,
        "generated_at_ns": time.monotonic_ns(),
        "retention_days": EU_AI_ACT_RETENTION_DAYS,
    }

    if output_path is not None:
        import json
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report


def check_retention_compliance(created_at_timestamp: float) -> dict:
    """Check if a record meets EU AI Act retention requirements.

    Args:
        created_at_timestamp: Unix timestamp (seconds) of record creation.

    Returns:
        Dict with compliance status and details.
    """
    now = time.time()
    age_days = (now - created_at_timestamp) / 86400
    retention_met = age_days <= EU_AI_ACT_RETENTION_DAYS

    return {
        "compliant": retention_met,
        "age_days": round(age_days, 2),
        "retention_required_days": EU_AI_ACT_RETENTION_DAYS,
        "expires_in_days": round(EU_AI_ACT_RETENTION_DAYS - age_days, 2) if retention_met else 0,
    }
