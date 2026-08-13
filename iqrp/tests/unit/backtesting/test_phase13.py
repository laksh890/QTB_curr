"""Phase 13 validation: PASS status, checklist keys, write report."""

from __future__ import annotations

import json
from pathlib import Path

from iqrp.app.backtesting.phase13 import (
    PHASE13_COMPONENTS,
    REQUIRED_DOCS,
    ComponentCheck,
    validate_phase13,
    write_phase13_report,
)


def test_validate_phase13_pass() -> None:
    report = validate_phase13(write_stubs=True)
    assert report["phase"] == "13"
    assert report["title"] == "Institutional Backtesting Platform"
    assert report["status"] == "PASS", report["summary"]
    assert report["summary"]["components_failed"] == 0
    assert report["summary"]["components_passed"] == report["summary"]["components_total"]
    assert len(report["architectural_rules"]) >= 5


def test_phase13_checklist_keys() -> None:
    report = validate_phase13(write_stubs=True)
    checklist = report["checklist"]
    expected = {c.name for c in PHASE13_COMPONENTS}
    for name in expected:
        assert name in checklist, f"missing checklist key: {name}"
        assert checklist[name] is True


def test_phase13_components_and_docs() -> None:
    report = validate_phase13(write_stubs=True)
    assert len(report["components"]) == len(PHASE13_COMPONENTS)
    assert all(c["status"] == "pass" for c in report["components"])
    assert len(report["documentation"]) == len(REQUIRED_DOCS)
    assert all(d["exists"] for d in report["documentation"])
    exports = report["integration"]["backtesting_package_exports"]
    for req in ("BacktestEngine", "BacktestSettings", "BacktestResult", "ExperimentRegistry"):
        assert req in exports


def test_write_phase13_report(tmp_path: Path) -> None:
    out = write_phase13_report(tmp_path / "Phase13_BacktestingPlatform_Validation.json")
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["phase"] == "13"
    md = tmp_path / "Phase13_BacktestingPlatform.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "PASS" in text
    assert "Phase 13" in text


def test_component_check_to_dict() -> None:
    c = ComponentCheck(
        name="X",
        category="test",
        import_path="iqrp.app.backtesting",
        symbol="BacktestEngine",
        docs=["BacktestingPlatform.md"],
    )
    d = c.to_dict()
    assert d["name"] == "X"
    assert d["status"] == "pending"
    assert d["symbol"] == "BacktestEngine"


def test_validate_without_stubs_still_passes() -> None:
    report = validate_phase13(write_stubs=False)
    assert report["status"] == "PASS"
