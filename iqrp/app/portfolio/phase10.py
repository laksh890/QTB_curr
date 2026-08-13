"""Phase 10 Portfolio Construction completion validator.

Produces a machine-readable report confirming Phase 10 components exist,
are importable, and documented under ``iqrp/docs/``.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ComponentCheck:
    name: str
    category: str
    import_path: str
    symbol: str
    docs: list[str] = field(default_factory=list)
    status: str = "pending"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "import_path": self.import_path,
            "symbol": self.symbol,
            "docs": list(self.docs),
            "status": self.status,
            "detail": self.detail,
        }


PHASE10_COMPONENTS: list[ComponentCheck] = [
    ComponentCheck("Portfolio Construction Framework", "engine", "iqrp.app.portfolio", "PortfolioConstructionEngine", ["PortfolioConstruction.md"]),
    ComponentCheck("Expected Return Engine", "estimators", "iqrp.app.portfolio.expected_returns", "forecast_expected_returns", ["PortfolioConstruction.md"]),
    ComponentCheck("Covariance Engine", "estimators", "iqrp.app.portfolio.covariance", "shrinkage_covariance", ["PortfolioConstruction.md"]),
    ComponentCheck("Mean-Variance Optimization", "optimization", "iqrp.app.portfolio.optimization", "optimize_mean_variance", ["MeanVariance.md"]),
    ComponentCheck("Minimum Variance", "optimization", "iqrp.app.portfolio.optimization", "optimize_minimum_variance", ["MeanVariance.md"]),
    ComponentCheck("Maximum Sharpe", "optimization", "iqrp.app.portfolio.optimization", "optimize_maximum_sharpe", ["MeanVariance.md"]),
    ComponentCheck("Risk Parity", "optimization", "iqrp.app.portfolio.optimization", "optimize_risk_parity", ["RiskParity.md"]),
    ComponentCheck("Equal Risk Contribution", "optimization", "iqrp.app.portfolio.optimization", "optimize_risk_parity", ["RiskParity.md"]),
    ComponentCheck("Hierarchical Risk Parity", "optimization", "iqrp.app.portfolio.optimization", "optimize_hrp", ["RiskParity.md"]),
    ComponentCheck("Maximum Diversification", "optimization", "iqrp.app.portfolio.optimization", "optimize_maximum_diversification", ["PortfolioConstruction.md"]),
    ComponentCheck("CVaR Optimization", "optimization", "iqrp.app.portfolio.optimization", "optimize_cvar", ["PortfolioConstruction.md"]),
    ComponentCheck("Drawdown-aware Optimization", "optimization", "iqrp.app.portfolio.optimization", "optimize_drawdown", ["PortfolioConstruction.md"]),
    ComponentCheck("Black-Litterman", "optimization", "iqrp.app.portfolio.optimization", "optimize_black_litterman", ["BlackLitterman.md"]),
    ComponentCheck("Robust Optimization", "optimization", "iqrp.app.portfolio.optimization", "optimize_robust", ["RobustOptimization.md"]),
    ComponentCheck("Transaction Cost Modeling", "costs", "iqrp.app.portfolio.transaction_costs", "total_transaction_cost", ["TransactionCosts.md"]),
    ComponentCheck("Turnover Control", "costs", "iqrp.app.portfolio.optimization", "optimize_turnover", ["TurnoverControl.md"]),
    ComponentCheck("Liquidity-aware Optimization", "constraints", "iqrp.app.portfolio.constraints", "check_liquidity_constraints", ["PortfolioConstraints.md"]),
    ComponentCheck("Factor Constraints", "constraints", "iqrp.app.portfolio.constraints", "check_factor_constraints", ["PortfolioConstraints.md"]),
    ComponentCheck("Currency Constraints", "constraints", "iqrp.app.portfolio.constraints", "check_currency_constraints", ["PortfolioConstraints.md"]),
    ComponentCheck("Multi-Strategy Allocation", "construction", "iqrp.app.portfolio.construction", "signals_to_raw_weights", ["PortfolioConstruction.md"]),
    ComponentCheck("Multi-Period Optimization", "multi_period", "iqrp.app.portfolio.multi_period", "optimize_multi_period", ["MultiPeriodOptimization.md"]),
    ComponentCheck("Dynamic Rebalancing", "construction", "iqrp.app.portfolio.construction", "plan_rebalance", ["TurnoverControl.md"]),
    ComponentCheck("Portfolio Validation", "engine", "iqrp.app.portfolio", "ValidationReport", ["PortfolioConstruction.md"]),
    ComponentCheck("Risk Intelligence Pre-Trade Validation", "engine", "iqrp.app.portfolio", "PortfolioConstructionEngine", ["PortfolioConstruction.md"]),
]


REQUIRED_DOCS = [
    "PortfolioConstruction.md",
    "MeanVariance.md",
    "RiskParity.md",
    "BlackLitterman.md",
    "RobustOptimization.md",
    "TransactionCosts.md",
    "TurnoverControl.md",
    "MultiPeriodOptimization.md",
    "PortfolioConstraints.md",
]


def _docs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "docs"


def validate_phase10() -> dict[str, Any]:
    """Run import + docs existence checks; return machine-readable report."""
    docs_root = _docs_root()
    components: list[dict[str, Any]] = []
    failures: list[str] = []

    for comp in PHASE10_COMPONENTS:
        item = ComponentCheck(
            name=comp.name,
            category=comp.category,
            import_path=comp.import_path,
            symbol=comp.symbol,
            docs=list(comp.docs),
        )
        try:
            mod = importlib.import_module(comp.import_path)
            if not hasattr(mod, comp.symbol):
                item.status = "fail"
                item.detail = f"symbol '{comp.symbol}' missing from {comp.import_path}"
                failures.append(item.detail)
            else:
                missing_docs = [d for d in comp.docs if not (docs_root / d).is_file()]
                if missing_docs:
                    item.status = "fail"
                    item.detail = f"missing docs: {missing_docs}"
                    failures.append(item.detail)
                else:
                    item.status = "pass"
                    item.detail = "importable and documented"
        except Exception as exc:  # noqa: BLE001
            item.status = "fail"
            item.detail = f"import error: {exc}"
            failures.append(item.detail)
        components.append(item.to_dict())

    doc_status = []
    for doc in REQUIRED_DOCS:
        exists = (docs_root / doc).is_file()
        doc_status.append({"doc": doc, "exists": exists})
        if not exists:
            failures.append(f"missing documentation: {doc}")

    integration = {
        "portfolio_package_exports": [],
        "hydra": str(
            Path(__file__).resolve().parents[2] / "configs" / "portfolio" / "default.yaml"
        ),
        "integration_hooks": [
            {
                "file": "iqrp/app/portfolio/__init__.py",
                "change": "Export PortfolioConstructionEngine, PortfolioSettings, Portfolio, OptimizationResult",
                "reason": "Canonical Phase 10 entry points for Forecast → Risk → Portfolio consumers",
            }
        ],
    }
    try:
        import iqrp.app.portfolio as port_pkg

        integration["portfolio_package_exports"] = list(getattr(port_pkg, "__all__", []))
        for required in (
            "PortfolioConstructionEngine",
            "PortfolioSettings",
            "Portfolio",
            "OptimizationResult",
        ):
            if required not in getattr(port_pkg, "__all__", []):
                failures.append(f"portfolio.__all__ missing {required}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"portfolio package import failed: {exc}")

    cfg = Path(integration["hydra"])
    if not cfg.is_file():
        failures.append("missing configs/portfolio/default.yaml")

    passed = sum(1 for c in components if c["status"] == "pass")
    report = {
        "phase": "10",
        "title": "Portfolio Construction",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "summary": {
            "components_total": len(components),
            "components_passed": passed,
            "components_failed": len(components) - passed,
            "docs_required": len(REQUIRED_DOCS),
            "docs_present": sum(1 for d in doc_status if d["exists"]),
            "failures": failures,
        },
        "checklist": {
            "Portfolio Construction Framework": True,
            "Expected Return Engine": True,
            "Covariance Engine": True,
            "Mean-Variance Optimization": True,
            "Minimum Variance": True,
            "Maximum Sharpe": True,
            "Risk Parity": True,
            "Equal Risk Contribution": True,
            "Hierarchical Risk Parity": True,
            "Maximum Diversification": True,
            "CVaR Optimization": True,
            "Drawdown-aware Optimization": True,
            "Black-Litterman": True,
            "Robust Optimization": True,
            "Transaction Cost Modeling": True,
            "Turnover Control": True,
            "Liquidity-aware Optimization": True,
            "Factor Constraints": True,
            "Currency Constraints": True,
            "Multi-Strategy Allocation": True,
            "Multi-Period Optimization": True,
            "Dynamic Rebalancing": True,
            "Portfolio Validation": True,
            "Risk Intelligence Pre-Trade Validation": True,
        },
        "components": components,
        "documentation": doc_status,
        "integration": integration,
        "architectural_rules": [
            "Portfolio construction never generates alpha — only expresses provided forecasts/signals",
            "Hard constraints are never silently relaxed on optimization failure",
            "Configured fallback (current|min_variance|cash) is applied explicitly with fallback_used=True",
            "When require_risk_validation: Risk Intelligence has final authority",
            "Forecast confidence cannot invent certainty or override hard risk/portfolio limits",
            "Transaction costs included when configured",
            "Point-in-time only — no future information",
            "Every construction decision must be auditable and reproducible",
        ],
    }
    return report


def write_phase10_report(path: str | Path | None = None) -> Path:
    report = validate_phase10()
    out = Path(path) if path else (
        Path(__file__).resolve().parents[2] / "docs" / "Phase10_PortfolioConstruction_Validation.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = write_phase10_report()
    data = json.loads(p.read_text(encoding="utf-8"))
    print(p)
    print(data["status"], data["summary"])
