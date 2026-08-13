"""Phase 11 validation: PASS status, checklist keys, write report."""

from __future__ import annotations

import json
from pathlib import Path

from iqrp.app.alpha.phase11 import (
    PHASE11_COMPONENTS,
    REQUIRED_DOCS,
    ComponentCheck,
    validate_phase11,
    write_phase11_report,
)
from iqrp.app.alpha.processes import available_scenarios, simulate_alpha_scenario


def test_validate_phase11_pass() -> None:
    report = validate_phase11(write_stubs=True)
    assert report["phase"] == "11"
    assert report["status"] == "PASS"
    assert report["summary"]["components_failed"] == 0
    assert report["summary"]["components_passed"] == report["summary"]["components_total"]


def test_checklist_keys() -> None:
    report = validate_phase11(write_stubs=True)
    checklist = report["checklist"]
    expected = {c.name for c in PHASE11_COMPONENTS}
    assert set(checklist.keys()) == expected
    assert all(checklist.values())
    for rule in report["architectural_rules"]:
        assert isinstance(rule, str) and len(rule) > 5


def test_write_phase11_report(tmp_path: Path) -> None:
    out = write_phase11_report(tmp_path / "Phase11_AlphaResearch_Validation.json")
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    md = tmp_path / "Phase11_AlphaResearch.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "PASS" in text
    assert "Statistical significance" in text or "Architectural" in text


def test_component_check_to_dict() -> None:
    c = ComponentCheck(
        name="X",
        category="test",
        import_path="iqrp.app.alpha",
        symbol="AlphaResearchEngine",
        docs=["AlphaResearch.md"],
    )
    d = c.to_dict()
    assert d["name"] == "X"
    assert d["status"] == "pending"


def test_required_docs_covered() -> None:
    report = validate_phase11(write_stubs=True)
    present = {d["doc"] for d in report["documentation"] if d["exists"]}
    assert set(REQUIRED_DOCS).issubset(present)


def test_scenarios_for_phase11() -> None:
    names = available_scenarios()
    assert "genuine_momentum" in names
    assert "random_noise" in names
    for name in names:
        sc = simulate_alpha_scenario(name, n=100, seed=0)
        assert "truth" in sc
        assert "signal" in sc and "returns" in sc
