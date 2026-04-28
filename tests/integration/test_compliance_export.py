"""Tests for EU AI Act compliance export."""

from __future__ import annotations

import time

import numpy as np
import pytest

import traceprop as tp
from traceprop.compliance.eu_ai_act import (
    EU_AI_ACT_LOG_VERSION,
    EU_AI_ACT_RETENTION_DAYS,
    check_retention_compliance,
    generate_compliance_report,
)
from traceprop.graph import get_graph, reset_graph


@pytest.fixture(autouse=True)
def _clean_graph():
    reset_graph()
    yield
    reset_graph()


class TestGenerateComplianceReport:
    def test_basic_report(self):
        x = tp.array([1.0, 2.0, 3.0], source_id="input-1")
        w = tp.array([0.5, 0.5, 0.5], source_id="weights")
        y = x * w
        report = generate_compliance_report(
            node_id=y._provenance_node_id,
            graph=get_graph(),
            system_name="TestSystem",
            system_version="1.0",
            deployer_name="TestCorp",
            high_risk_category="medical",
        )
        assert report["log_version"] == EU_AI_ACT_LOG_VERSION
        assert report["system_name"] == "TestSystem"
        assert report["deployer_name"] == "TestCorp"
        assert report["high_risk_category"] == "medical"
        assert report["ancestor_count"] == 2
        assert report["retention_days"] == EU_AI_ACT_RETENTION_DAYS

    def test_report_contains_edges(self):
        x = tp.array([1.0, 2.0])
        y = x + 1
        report = generate_compliance_report(
            node_id=y._provenance_node_id,
            graph=get_graph(),
            system_name="S",
            system_version="1",
            deployer_name="D",
            high_risk_category="C",
        )
        assert len(report["edges"]) >= 1

    def test_report_to_file(self, tmp_path):
        x = tp.array([1.0])
        y = x * 2
        out = tmp_path / "report.json"
        report = generate_compliance_report(
            node_id=y._provenance_node_id,
            graph=get_graph(),
            system_name="S",
            system_version="1",
            deployer_name="D",
            high_risk_category="C",
            output_path=str(out),
        )
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["system_name"] == "S"

    def test_convenience_function(self):
        x = tp.array([1.0, 2.0])
        y = x + 1
        report = tp.compliance_report(
            y,
            system_name="S",
            system_version="1",
            deployer_name="D",
            high_risk_category="C",
        )
        assert report is not None
        assert report["ancestor_count"] >= 1

    def test_convenience_no_provenance(self):
        result = tp.compliance_report(
            42,  # not a tensor
            system_name="S",
            system_version="1",
            deployer_name="D",
            high_risk_category="C",
        )
        assert result is None


class TestRetentionCompliance:
    def test_recent_record_compliant(self):
        result = check_retention_compliance(time.time() - 86400)  # 1 day ago
        assert result["compliant"] is True
        assert result["age_days"] < 2

    def test_old_record_non_compliant(self):
        old_ts = time.time() - (EU_AI_ACT_RETENTION_DAYS + 1) * 86400
        result = check_retention_compliance(old_ts)
        assert result["compliant"] is False
        assert result["expires_in_days"] == 0

    def test_retention_days_constant(self):
        assert EU_AI_ACT_RETENTION_DAYS == 3650
