"""Phase 10 validation: PASS status, checklist keys, write report."""

from __future__ import annotations

import json
from pathlib import Path

from iqrp.app.portfolio.phase10 import (
    PHASE10_COMPONENTS,
    REQUIRED_DOCS,
    validate_phase10,
    write_phase10_report,
)


EXPECTED_CHECKLIST_KEYS = [
    "Portfolio Construction Framework",
    "Expected Return Engine",
    "Covariance Engine",
    "Mean-Variance Optimization",
    "Minimum Variance",
    "Maximum Sharpe",
    "Risk Parity",
    "Equal Risk Contribution",
    "Hierarchical Risk Parity",
    "Maximum Diversification",
    "CVaR Optimization",
    "Drawdown-aware Optimization",
    "Black-Litterman",
    "Robust Optimization",
    "Transaction Cost Modeling",
    "Turnover Control",
    "Liquidity-aware Optimization",
    "Factor Constraints",
    "Currency Constraints",
    "Multi-Strategy Allocation",
    "Multi-Period Optimization",
    "Dynamic Rebalancing",
    "Portfolio Validation",
    "Risk Intelligence Pre-Trade Validation",
]


def test_validate_phase10_pass():
    report = validate_phase10()
    assert report["phase"] == "10"
    assert report["title"] == "Portfolio Construction"
    assert report["status"] == "PASS", report["summary"]
    assert report["summary"]["components_failed"] == 0
    assert report["summary"]["components_passed"] == report["summary"]["components_total"]
    assert len(report["architectural_rules"]) >= 5


def test_phase10_checklist_keys():
    report = validate_phase10()
    checklist = report["checklist"]
    for key in EXPECTED_CHECKLIST_KEYS:
        assert key in checklist, f"missing checklist key: {key}"
        assert checklist[key] is True


def test_phase10_components_and_docs():
    report = validate_phase10()
    assert len(report["components"]) == len(PHASE10_COMPONENTS)
    assert all(c["status"] == "pass" for c in report["components"])
    assert len(report["documentation"]) == len(REQUIRED_DOCS)
    assert all(d["exists"] for d in report["documentation"])
    exports = report["integration"]["portfolio_package_exports"]
    for req in (
        "PortfolioConstructionEngine",
        "PortfolioSettings",
        "Portfolio",
        "OptimizationResult",
    ):
        assert req in exports


def test_write_phase10_report(tmp_path: Path):
    out = write_phase10_report(tmp_path / "Phase10_PortfolioConstruction_Validation.json")
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["phase"] == "10"
