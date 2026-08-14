"""Risk budget construction across scopes and risk types."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.capital.types import RiskBudget, _utc_now

SCOPES = ("portfolio", "strategy", "asset", "sector", "factor", "market", "account")
RISK_TYPES = (
    "volatility",
    "var",
    "cvar",
    "liquidity",
    "concentration",
    "drawdown",
    "factor",
    "tail",
)


def build_risk_budgets(
    names: list[str],
    *,
    risk_budgets: dict[str, float] | None = None,
    scopes: dict[str, Any] | None = None,
    risk_types: dict[str, Any] | None = None,
    total_risk_budget: float = 1.0,
    data_version: str = "1.0.0",
    model_version: str = "1.0.0",
    confidence: float = 1.0,
) -> list[RiskBudget]:
    """Build hierarchical RiskBudget objects for scopes and risk types.

    ``scopes`` maps scope → budget (float) or {name: budget}.
    ``risk_types`` maps risk_type → budget (float) or {name: budget}.
    Per-strategy risk_budgets default to equal share of total_risk_budget.
    """
    n = len(names)
    ts = _utc_now()
    conf = float(np.clip(confidence, 0.0, 1.0))
    out: list[RiskBudget] = []

    # Strategy-level volatility budgets (primary capital risk budgets)
    if risk_budgets:
        rb = {str(k): float(v) for k, v in risk_budgets.items()}
        total = float(sum(max(v, 0.0) for v in rb.values())) or float(total_risk_budget)
    else:
        share = float(total_risk_budget) / n if n else 0.0
        rb = dict.fromkeys(names, share)
        total = float(total_risk_budget)

    for nm in names:
        b = float(rb.get(nm, 0.0))
        out.append(
            RiskBudget(
                name=nm,
                scope="strategy",
                risk_type="volatility",
                budget=b,
                used=0.0,
                timestamp=ts,
                data_version=data_version,
                model_version=model_version,
                inputs={"total_risk_budget": total},
                params={},
                output={},
                confidence=conf,
                reasons=["strategy volatility risk budget"],
            )
        )

    # Portfolio aggregate
    out.append(
        RiskBudget(
            name="portfolio",
            scope="portfolio",
            risk_type="volatility",
            budget=float(total),
            used=0.0,
            timestamp=ts,
            data_version=data_version,
            model_version=model_version,
            inputs={"names": list(names)},
            params={},
            output={},
            confidence=conf,
            reasons=["portfolio total volatility budget"],
        )
    )

    # Additional hierarchical scopes
    for scope, payload in (scopes or {}).items():
        scope_key = str(scope).lower()
        if scope_key not in SCOPES:
            scope_key = "portfolio"
        if isinstance(payload, dict):
            for key, val in payload.items():
                out.append(
                    RiskBudget(
                        name=str(key),
                        scope=scope_key,
                        risk_type="volatility",
                        budget=float(val),
                        timestamp=ts,
                        data_version=data_version,
                        model_version=model_version,
                        confidence=conf,
                        reasons=[f"scope budget for {scope_key}"],
                    )
                )
        else:
            out.append(
                RiskBudget(
                    name=scope_key,
                    scope=scope_key,
                    risk_type="volatility",
                    budget=float(payload),
                    timestamp=ts,
                    data_version=data_version,
                    model_version=model_version,
                    confidence=conf,
                    reasons=[f"scope budget for {scope_key}"],
                )
            )

    # Risk-type budgets
    for rtype, payload in (risk_types or {}).items():
        rt = str(rtype).lower()
        if rt not in RISK_TYPES:
            rt = "volatility"
        if isinstance(payload, dict):
            for key, val in payload.items():
                out.append(
                    RiskBudget(
                        name=str(key),
                        scope="portfolio",
                        risk_type=rt,
                        budget=float(val),
                        timestamp=ts,
                        data_version=data_version,
                        model_version=model_version,
                        confidence=conf,
                        reasons=[f"{rt} risk-type budget"],
                    )
                )
        else:
            out.append(
                RiskBudget(
                    name=rt,
                    scope="portfolio",
                    risk_type=rt,
                    budget=float(payload),
                    timestamp=ts,
                    data_version=data_version,
                    model_version=model_version,
                    confidence=conf,
                    reasons=[f"{rt} risk-type budget"],
                )
            )

    return out


def mark_budgets_used(
    budgets: list[RiskBudget],
    used_by_name: dict[str, float],
) -> list[RiskBudget]:
    """Update strategy-scope used amounts from allocation risk usage."""
    for b in budgets:
        if b.scope == "strategy" and b.name in used_by_name:
            b.used = float(used_by_name[b.name])
        elif b.scope == "portfolio" and b.risk_type == "volatility" and b.name == "portfolio":
            b.used = float(sum(used_by_name.values()))
    return budgets


def strategy_budget_vector(
    names: list[str],
    budgets: list[RiskBudget],
) -> dict[str, float]:
    """Extract strategy-level volatility budgets as a name→budget map."""
    out = dict.fromkeys(names, 0.0)
    for b in budgets:
        if b.scope == "strategy" and b.risk_type == "volatility" and b.name in out:
            out[b.name] = float(b.budget)
    s = float(sum(out.values()))
    if s <= 0 and names:
        share = 1.0 / len(names)
        return dict.fromkeys(names, share)
    return out
