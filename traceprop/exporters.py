"""Export lineage data to OpenTelemetry spans and Parquet files."""

from __future__ import annotations

import json

from traceprop.exceptions import ExportError, safe_provenance
from traceprop.graph import get_graph


@safe_provenance
def to_otel_spans(service_name: str = "traceprop") -> list:
    """Export lineage graph as OpenTelemetry spans.

    Requires: opentelemetry-api, opentelemetry-sdk
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter
    except ImportError:
        raise ExportError("Install opentelemetry-sdk: pip install traceprop[otel]")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = InMemorySpanExporter()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("traceprop")

    graph = get_graph()
    for edge in graph.edges.values():
        with tracer.start_as_current_span(edge.op_name) as span:
            span.set_attribute("traceprop.edge_id", edge.id)
            span.set_attribute("traceprop.input_ids", list(edge.input_ids))
            span.set_attribute("traceprop.output_id", edge.output_id)

    spans = exporter.get_finished_spans()
    provider.shutdown()
    return list(spans)


@safe_provenance
def to_parquet(path: str) -> None:
    """Export lineage graph nodes and edges to Parquet files.

    Creates {path}_nodes.parquet and {path}_edges.parquet.
    Requires: pyarrow
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ExportError("Install pyarrow: pip install traceprop[parquet]")

    graph = get_graph()

    # Nodes
    node_data = {
        "id": [],
        "shape": [],
        "dtype": [],
        "timestamp": [],
    }
    for n in graph.nodes.values():
        node_data["id"].append(n.id)
        node_data["shape"].append(str(n.shape))
        node_data["dtype"].append(n.dtype)
        node_data["timestamp"].append(n.timestamp)

    nodes_table = pa.table(node_data)
    pq.write_table(nodes_table, f"{path}_nodes.parquet")

    # Edges
    edge_data = {
        "id": [],
        "op_name": [],
        "input_ids": [],
        "output_id": [],
        "timestamp": [],
    }
    for e in graph.edges.values():
        edge_data["id"].append(e.id)
        edge_data["op_name"].append(e.op_name)
        edge_data["input_ids"].append(str(e.input_ids))
        edge_data["output_id"].append(e.output_id)
        edge_data["timestamp"].append(e.timestamp)

    edges_table = pa.table(edge_data)
    pq.write_table(edges_table, f"{path}_edges.parquet")


def export_to_parquet(tensor, path: str) -> None:
    """Export tensor data to Parquet with lineage embedded in file metadata.

    Requires: pyarrow
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ExportError("Install pyarrow: pip install traceprop[parquet]")

    import numpy as np

    from traceprop.query import ProvenanceView

    # Get the raw numpy data
    arr = np.asarray(tensor)
    if arr.ndim == 1:
        table = pa.table({"data": arr})
    elif arr.ndim == 2:
        columns = {f"col_{i}": arr[:, i] for i in range(arr.shape[1])}
        table = pa.table(columns)
    else:
        table = pa.table({"data": arr.ravel()})

    # Build lineage metadata
    view = ProvenanceView(tensor)
    graph = get_graph()
    lineage = {
        "node_id": view.node_id,
        "ancestors": sorted(view.ancestors()),
        "source_ids": sorted(view.source_ids_in_path()),
        "ops": [],
    }
    for op in view.ops():
        lineage["ops"].append({
            "id": op.id,
            "op_name": op.op_name,
            "input_ids": list(op.input_ids),
            "output_id": op.output_id,
        })

    # Embed lineage as Parquet file metadata
    existing_meta = table.schema.metadata or {}
    existing_meta[b"traceprop_lineage"] = json.dumps(lineage).encode()
    table = table.replace_schema_metadata(existing_meta)

    pq.write_table(table, path)


def read_lineage_from_parquet(path: str) -> dict | None:
    """Read embedded lineage metadata from a Parquet file.

    Returns the lineage dict or None if no lineage is embedded.
    Requires: pyarrow
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ExportError("Install pyarrow: pip install traceprop[parquet]")

    meta = pq.read_metadata(path)
    schema_meta = meta.schema.metadata or {}
    raw = schema_meta.get(b"traceprop_lineage")
    if raw is None:
        return None
    return json.loads(raw.decode())
