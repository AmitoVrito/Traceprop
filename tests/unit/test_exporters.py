"""Tests for exporters (OTel and Parquet are optional deps, so we test conditionally)."""

import numpy as np
import pytest

import traceprop as tp
from traceprop.graph import reset_graph


@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


def _make_lineage():
    a = tp.array([1.0, 2.0])
    b = tp.array([3.0, 4.0])
    return a + b


class TestOtelExporter:
    def test_export(self):
        pytest.importorskip("opentelemetry")
        from traceprop.exporters import to_otel_spans

        _make_lineage()
        spans = to_otel_spans()
        assert spans is not None
        assert len(spans) >= 1


class TestParquetExporter:
    def test_export(self, tmp_path):
        pytest.importorskip("pyarrow")
        from traceprop.exporters import to_parquet

        _make_lineage()
        prefix = str(tmp_path / "lineage")
        to_parquet(prefix)
        assert (tmp_path / "lineage_nodes.parquet").exists()
        assert (tmp_path / "lineage_edges.parquet").exists()


class TestEnhancedParquetExport:
    def test_export_to_parquet_with_lineage(self, tmp_path):
        pytest.importorskip("pyarrow")
        from traceprop.exporters import export_to_parquet, read_lineage_from_parquet

        c = _make_lineage()
        path = str(tmp_path / "output.parquet")
        export_to_parquet(c, path)
        assert (tmp_path / "output.parquet").exists()

        lineage = read_lineage_from_parquet(path)
        assert lineage is not None
        assert lineage["node_id"] == c._provenance_node_id
        assert isinstance(lineage["ancestors"], list)
        assert isinstance(lineage["ops"], list)

    def test_export_with_source_ids(self, tmp_path):
        pytest.importorskip("pyarrow")
        from traceprop.exporters import export_to_parquet, read_lineage_from_parquet

        a = tp.array([1.0, 2.0], source_id="src_A")
        b = tp.array([3.0, 4.0], source_id="src_B")
        c = a + b

        path = str(tmp_path / "sourced.parquet")
        export_to_parquet(c, path)

        lineage = read_lineage_from_parquet(path)
        assert "src_A" in lineage["source_ids"]
        assert "src_B" in lineage["source_ids"]

    def test_read_lineage_no_metadata(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq
        from traceprop.exporters import read_lineage_from_parquet

        table = pa.table({"x": [1, 2, 3]})
        path = str(tmp_path / "plain.parquet")
        pq.write_table(table, path)

        lineage = read_lineage_from_parquet(path)
        assert lineage is None

    def test_export_2d_array(self, tmp_path):
        pytest.importorskip("pyarrow")
        from traceprop.exporters import export_to_parquet, read_lineage_from_parquet

        a = tp.array([[1.0, 2.0], [3.0, 4.0]])
        path = str(tmp_path / "matrix.parquet")
        export_to_parquet(a, path)

        lineage = read_lineage_from_parquet(path)
        assert lineage is not None
