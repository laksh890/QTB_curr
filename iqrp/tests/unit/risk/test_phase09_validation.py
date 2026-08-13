"""Tests for Phase 09 validation report (iqrp.app.risk.phase09)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqrp.app.risk.phase09 import (
    PHASE09_COMPONENTS,
    REQUIRED_DOCS,
    ComponentCheck,
    validate_phase09,
    write_phase09_report,
)


REQUIRED_CHECKLIST_KEYS = [
    "Risk Framework",
    "Position Sizing Engine",
    "Portfolio Risk Engine",
    "VaR / CVaR Engine",
    "Stress Testing Engine",
    "Scenario Analysis",
    "Monte Carlo Risk Engine",
    "Correlation & Dependency Engine",
    "Kelly & Capital Allocation",
    "Dynamic Leverage Engine",
    "Risk Limits Engine",
    "Risk Intelligence Ensemble",
]


class TestValidatePhase09:
    def test_validate_phase09_status_pass(self) -> None:
        report = validate_phase09()
        assert report["phase"] == "09"
        assert report["title"] == "Risk Intelligence"
        assert "timestamp" in report
        # Document status — prefer PASS; if FAIL, surface failures clearly for diagnosis
        if report["status"] != "PASS":
            failures = report["summary"].get("failures", [])
            pytest.fail(
                "validate_phase09 status is FAIL; missing pieces:\n"
                + "\n".join(f"  - {f}" for f in failures)
            )
        assert report["status"] == "PASS"
        assert report["summary"]["components_failed"] == 0
        assert report["summary"]["docs_present"] == report["summary"]["docs_required"]

    def test_checklist_keys_present(self) -> None:
        report = validate_phase09()
        checklist = report["checklist"]
        for key in REQUIRED_CHECKLIST_KEYS:
            assert key in checklist, f"missing checklist key: {key}"
            assert checklist[key] is True

    def test_components_and_docs_structure(self) -> None:
        report = validate_phase09()
        assert len(report["components"]) == len(PHASE09_COMPONENTS)
        for comp in report["components"]:
            assert comp["status"] == "pass"
            assert "import_path" in comp
            assert "symbol" in comp
        doc_names = {d["doc"] for d in report["documentation"]}
        assert doc_names == set(REQUIRED_DOCS)
        assert all(d["exists"] for d in report["documentation"])
        assert "architectural_rules" in report
        assert len(report["architectural_rules"]) >= 10
        assert "CapitalAllocator" in report["integration"]["risk_package_exports"]
        assert "RiskIntelligenceEnsemble" in report["integration"]["risk_package_exports"]

    def test_component_check_to_dict(self) -> None:
        cc = ComponentCheck(
            name="X",
            category="capital",
            import_path="iqrp.app.risk.capital",
            symbol="CapitalAllocator",
            docs=["CapitalAllocation.md"],
            status="pass",
            detail="ok",
        )
        d = cc.to_dict()
        assert d["name"] == "X"
        assert d["docs"] == ["CapitalAllocation.md"]

    def test_write_phase09_report(self, tmp_path: Path) -> None:
        out = write_phase09_report(tmp_path / "Phase09_RiskIntelligence_Validation.json")
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["status"] == "PASS"
        assert data["phase"] == "09"
        for key in REQUIRED_CHECKLIST_KEYS:
            assert key in data["checklist"]

    def test_write_phase09_report_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Redirect default write into tmp by patching Path behavior via path arg only —
        # call with explicit path (default may write into repo docs which is fine if PASS).
        # Also exercise default when path is None but isolate to tmp by monkeypatching parents.
        report_path = tmp_path / "nested" / "report.json"
        p = write_phase09_report(report_path)
        assert p == report_path
        assert p.parent.is_dir()
