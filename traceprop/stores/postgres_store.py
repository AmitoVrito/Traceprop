"""PostgreSQL store for persisting lineage data."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

from traceprop.exceptions import StoreUnavailableWarning

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


class PostgresStore:
    """Persist lineage nodes and edges to PostgreSQL."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn = None
        if psycopg2 is None:
            warnings.warn(
                "psycopg2 not installed; PostgresStore is unavailable. "
                "Install with: pip install traceprop[postgres]",
                StoreUnavailableWarning,
                stacklevel=2,
            )
            return
        try:
            self._conn = psycopg2.connect(dsn)
            self._conn.autocommit = True
            self._create_schema()
        except Exception as e:
            warnings.warn(
                f"Failed to connect to PostgreSQL: {e}",
                StoreUnavailableWarning,
                stacklevel=2,
            )
            self._conn = None

    def _create_schema(self) -> None:
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tp_nodes (
                    id INTEGER PRIMARY KEY,
                    shape TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    timestamp BIGINT NOT NULL,
                    meta TEXT,
                    source_ids TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tp_edges (
                    id INTEGER PRIMARY KEY,
                    op_name TEXT NOT NULL,
                    input_ids TEXT NOT NULL,
                    output_id INTEGER NOT NULL,
                    timestamp BIGINT NOT NULL,
                    meta TEXT,
                    source_ids TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tp_edges_output ON tp_edges(output_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tp_nodes_id ON tp_nodes(id)")

    def is_available(self) -> bool:
        return self._conn is not None

    def persist_node(self, node) -> None:
        """Persist a TensorNode. Fails silently on error."""
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                meta_json = json.dumps(node.meta) if node.meta else None
                cur.execute(
                    "INSERT INTO tp_nodes (id, shape, dtype, timestamp, meta) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (node.id, json.dumps(node.shape), node.dtype, node.timestamp, meta_json),
                )
        except Exception:
            logger.debug("Failed to persist node %s", node.id, exc_info=True)

    def persist_edge(self, edge) -> None:
        """Persist an OpEdge. Fails silently on error."""
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                meta_json = json.dumps(edge.meta) if edge.meta else None
                source_ids_json = json.dumps(edge.source_ids) if edge.source_ids else None
                cur.execute(
                    "INSERT INTO tp_edges (id, op_name, input_ids, output_id, timestamp, meta, source_ids) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (edge.id, edge.op_name, json.dumps(edge.input_ids), edge.output_id, edge.timestamp, meta_json, source_ids_json),
                )
        except Exception:
            logger.debug("Failed to persist edge %s", edge.id, exc_info=True)

    def query_by_source(self, source_id: str) -> list[dict]:
        """Query edges that reference a source_id."""
        if self._conn is None:
            return []
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT id, op_name, input_ids, output_id, timestamp, meta, source_ids FROM tp_edges WHERE source_ids LIKE %s",
                    (f"%{source_id}%",),
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "op_name": r[1],
                        "input_ids": json.loads(r[2]),
                        "output_id": r[3],
                        "timestamp": r[4],
                        "meta": json.loads(r[5]) if r[5] else None,
                        "source_ids": json.loads(r[6]) if r[6] else None,
                    }
                    for r in rows
                ]
        except Exception:
            logger.debug("Failed to query by source %s", source_id, exc_info=True)
            return []
