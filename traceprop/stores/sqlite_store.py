"""SQLite store backend for persistent lineage data."""

from __future__ import annotations

import json
import sqlite3

from traceprop.graph import OpEdge, TensorNode


class SQLiteStore:
    """Persists lineage data to a SQLite database."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                meta TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY,
                op_name TEXT NOT NULL,
                input_ids TEXT NOT NULL,
                output_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edges_output ON edges(output_id);
        """)
        self._conn.commit()

    def save_node(self, node: TensorNode) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes (id, shape, dtype, timestamp, meta) VALUES (?, ?, ?, ?, ?)",
            (node.id, json.dumps(node.shape), node.dtype, node.timestamp,
             json.dumps(node.meta) if node.meta else None),
        )
        self._conn.commit()

    def save_nodes_batch(self, nodes: list[TensorNode]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO nodes (id, shape, dtype, timestamp, meta) VALUES (?, ?, ?, ?, ?)",
            [(n.id, json.dumps(n.shape), n.dtype, n.timestamp,
              json.dumps(n.meta) if n.meta else None) for n in nodes],
        )
        self._conn.commit()

    def save_edge(self, edge: OpEdge) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (id, op_name, input_ids, output_id, timestamp, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (edge.id, edge.op_name, json.dumps(edge.input_ids), edge.output_id,
             edge.timestamp, json.dumps(edge.meta) if edge.meta else None),
        )
        self._conn.commit()

    def save_edges_batch(self, edges: list[OpEdge]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO edges (id, op_name, input_ids, output_id, timestamp, meta) VALUES (?, ?, ?, ?, ?, ?)",
            [(e.id, e.op_name, json.dumps(e.input_ids), e.output_id,
              e.timestamp, json.dumps(e.meta) if e.meta else None) for e in edges],
        )
        self._conn.commit()

    def get_node(self, node_id: int) -> TensorNode | None:
        row = self._conn.execute("SELECT id, shape, dtype, timestamp, meta FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_edge(self, edge_id: int) -> OpEdge | None:
        row = self._conn.execute("SELECT id, op_name, input_ids, output_id, timestamp, meta FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def all_nodes(self) -> list[TensorNode]:
        rows = self._conn.execute("SELECT id, shape, dtype, timestamp, meta FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    def all_edges(self) -> list[OpEdge]:
        rows = self._conn.execute("SELECT id, op_name, input_ids, output_id, timestamp, meta FROM edges").fetchall()
        return [self._row_to_edge(r) for r in rows]

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM nodes; DELETE FROM edges;")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_node(row) -> TensorNode:
        node = TensorNode.__new__(TensorNode)
        node.id = row[0]
        node.shape = tuple(json.loads(row[1]))
        node.dtype = row[2]
        node.timestamp = row[3]
        node.meta = json.loads(row[4]) if row[4] else None
        return node

    @staticmethod
    def _row_to_edge(row) -> OpEdge:
        edge = OpEdge.__new__(OpEdge)
        edge.id = row[0]
        edge.op_name = row[1]
        edge.input_ids = tuple(json.loads(row[2]))
        edge.output_id = row[3]
        edge.timestamp = row[4]
        edge.meta = json.loads(row[5]) if row[5] else None
        return edge
