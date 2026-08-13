"""Phase 12 validation: PASS status, checklist keys, write report."""

from __future__ import annotations

import json
from pathlib import Path

from iqrp.app.execution.phase12 import (
    PHASE12_COMPONENTS,
    REQUIRED_DOCS,
    ComponentCheck,
    validate_phase12,
    write_phase12_report,
)


def test_validate_phase12_pass() -> None:
    report = validate_phase12(write_stubs=True)
    assert report["phase"] == "12"
    assert report["title"] == "Institutional Execution Platform"
    assert report["status"] == "PASS", report["summary"]
    assert report["summary"]["components_failed"] == 0
    assert report["summary"]["components_passed"] == report["summary"]["components_total"]
    assert len(report["architectural_rules"]) >= 5


def test_phase12_checklist_keys() -> None:
    report = validate_phase12(write_stubs=True)
    checklist = report["checklist"]
    expected = {c.name for c in PHASE12_COMPONENTS}
    # checklist includes component names; engine may appear once
    for name in expected:
        assert name in checklist, f"missing checklist key: {name}"
        assert checklist[name] is True


def test_phase12_components_and_docs() -> None:
    report = validate_phase12(write_stubs=True)
    assert len(report["components"]) == len(PHASE12_COMPONENTS)
    assert all(c["status"] == "pass" for c in report["components"])
    assert len(report["documentation"]) == len(REQUIRED_DOCS)
    assert all(d["exists"] for d in report["documentation"])
    exports = report["integration"]["execution_package_exports"]
    for req in ("ExecutionEngine", "ExecutionSettings", "OrderManager", "KillSwitch"):
        assert req in exports


def test_write_phase12_report(tmp_path: Path) -> None:
    out = write_phase12_report(tmp_path / "Phase12_ExecutionPlatform_Validation.json")
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["phase"] == "12"
    md = tmp_path / "Phase12_ExecutionPlatform.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "PASS" in text


def test_component_check_to_dict() -> None:
    c = ComponentCheck(
        name="X",
        category="test",
        import_path="iqrp.app.execution",
        symbol="ExecutionEngine",
        docs=["ExecutionPlatform.md"],
    )
    d = c.to_dict()
    assert d["name"] == "X"
    assert d["status"] == "pending"
