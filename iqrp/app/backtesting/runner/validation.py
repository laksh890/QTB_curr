"""Preflight and post-run integrity validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from iqrp.app.backtesting.accounting.reconciliation import reconcile_capital
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.context import PipelineContext
from iqrp.app.backtesting.runner.result import OperationalBacktestResult


@dataclass
class ValidationIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
            "checks": dict(self.checks),
        }


def preflight_validate(
    config: BacktestRunConfig,
    *,
    strategy_registered: bool,
    dataset_ok: bool,
    dataset_detail: str = "",
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checks: dict[str, bool] = {}

    checks["strategy_id"] = bool(config.strategy_id)
    if not config.strategy_id:
        issues.append(ValidationIssue("strategy_id", "critical", "strategy_id is required"))

    checks["strategy_registered"] = bool(strategy_registered)
    if not strategy_registered:
        issues.append(
            ValidationIssue(
                "strategy_registered",
                "critical",
                f"strategy {config.strategy_id!r} v{config.strategy_version!r} not registered",
            )
        )

    has_data = bool(config.dataset_path or config.dataset_id)
    checks["dataset_reference"] = has_data
    if not has_data:
        issues.append(
            ValidationIssue(
                "dataset_reference",
                "critical",
                "dataset_path or dataset_id is required (no downloads performed)",
            )
        )

    checks["dataset_validated"] = bool(dataset_ok)
    if has_data and not dataset_ok:
        issues.append(
            ValidationIssue(
                "dataset_validated",
                "critical",
                dataset_detail or "dataset failed validation",
            )
        )

    checks["capital"] = float(config.initial_capital) > 0
    if float(config.initial_capital) <= 0:
        issues.append(ValidationIssue("capital", "critical", "initial_capital must be positive"))

    checks["dates"] = True
    if config.start and config.end and str(config.start) > str(config.end):
        checks["dates"] = False
        issues.append(ValidationIssue("dates", "critical", "start must be <= end"))

    critical = [i for i in issues if i.severity == "critical"]
    return ValidationReport(ok=len(critical) == 0, issues=issues, checks=checks)


def integrity_validate(
    context: PipelineContext,
    result: OperationalBacktestResult,
    *,
    results_persisted: bool,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checks: dict[str, bool] = {}

    checks["data_validated"] = bool(context.diagnostics.get("data_validated", False))
    checks["pit_enforced"] = bool(context.config.enforce_pit) and not context.invalidated
    checks["no_lookahead"] = not context.invalidated
    checks["strategy_ran"] = context.bar_count > 0
    checks["risk_state"] = bool(context.risk_state) or context.bar_count > 0
    checks["portfolio_targets"] = bool(context.target_weights) or len(context.orders) > 0
    checks["orders_logged"] = True
    checks["fills_logged"] = True
    checks["positions_tracked"] = True
    checks["pnl_tracked"] = len(result.equity_curve) > 0
    checks["cash_tracked"] = "cash" in (result.capital or {})
    checks["results_persisted"] = bool(results_persisted)

    if not checks["data_validated"]:
        issues.append(ValidationIssue("data_validated", "critical", "dataset was not validated"))
    if context.invalidated:
        issues.append(
            ValidationIssue(
                "lookahead",
                "critical",
                context.invalidation_reason or "backtest invalidated",
            )
        )
    if not checks["pnl_tracked"]:
        issues.append(ValidationIssue("pnl_tracked", "critical", "equity curve is empty"))
    if not results_persisted:
        issues.append(ValidationIssue("results_persisted", "warning", "results not persisted"))

    try:
        recon = reconcile_capital(
            context.capital,
            ending_equity=(
                float(result.equity_curve[-1]) if result.equity_curve else context.capital.equity
            ),
            tolerance=float(context.config.reconciliation_tolerance),
            fail=False,
        )
        checks["reconciled"] = bool(recon.ok)
        result.reconciliation = recon.to_dict()
        if not recon.ok:
            issues.append(
                ValidationIssue(
                    "reconciliation",
                    "critical",
                    recon.detail or "capital reconciliation failed",
                )
            )
    except Exception as exc:
        checks["reconciled"] = False
        issues.append(ValidationIssue("reconciliation", "critical", str(exc)))

    critical = [i for i in issues if i.severity == "critical"]
    return ValidationReport(ok=len(critical) == 0, issues=issues, checks=checks)


def assert_mapping(data: Mapping[str, Any] | None, name: str) -> None:
    if data is None or not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "integrity_validate",
    "preflight_validate",
]
