"""Institutional Capital Allocation Engine — CapitalAllocator."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.capital.capital_budget import allocate_capital_budgets
from iqrp.app.risk.capital.capacity import estimate_capacity
from iqrp.app.risk.capital.config import CapitalSettings
from iqrp.app.risk.capital.constraints import (
    apply_participation_constraint,
    apply_turnover_constraint,
    project_weights,
)
from iqrp.app.risk.capital.correlation import (
    correlation_crowding_scales,
    effective_risk_budgets,
    strategy_correlation,
)
from iqrp.app.risk.capital.diagnostics import diagnose_allocation
from iqrp.app.risk.capital.drawdown import drawdown_scales
from iqrp.app.risk.capital.dynamic import dynamic_risk_scales
from iqrp.app.risk.capital.equal_risk import equal_risk_weights
from iqrp.app.risk.capital.evaluator import evaluate_allocation
from iqrp.app.risk.capital.hierarchical import herc_weights, hrp_weights
from iqrp.app.risk.capital.optimizer import optimize_risk_budgets
from iqrp.app.risk.capital.processes import simulate_capital_scenario
from iqrp.app.risk.capital.risk_budget import (
    build_risk_budgets,
    mark_budgets_used,
    strategy_budget_vector,
)
from iqrp.app.risk.capital.risk_parity import capital_risk_parity
from iqrp.app.risk.capital.strategy_allocation import (
    allocate_strategy as _allocate_strategy,
)
from iqrp.app.risk.capital.strategy_allocation import build_strategy_allocations
from iqrp.app.risk.capital.types import CapitalAllocation, RiskBudget, _utc_now
from iqrp.app.risk.capital.volatility import volatility_budgets
from iqrp.app.risk.market.correlation import covariance_matrix


class CapitalAllocator:
    """Allocate institutional capital across strategies under hard risk limits.

    Historical performance alone never sets weights. Use ``expected_opportunity`` /
    score inputs when provided; otherwise risk / uncertainty / liquidity / capacity /
    correlation drive the allocation. Confidence cannot expand beyond a 1.0 scale.
    """

    def __init__(self, settings: CapitalSettings | None = None) -> None:
        self.settings = settings or CapitalSettings.default()
        self.last_allocation: CapitalAllocation | None = None

    # ------------------------------------------------------------------ API
    def allocate(
        self,
        names: list[str],
        *,
        method: str = "risk_parity",
        cov: np.ndarray | None = None,
        returns: np.ndarray | None = None,
        risk_budgets: dict[str, float] | None = None,
        capital: float = 1.0,
        vols: np.ndarray | list[float] | None = None,
        adv: np.ndarray | list[float] | None = None,
        spreads: np.ndarray | list[float] | None = None,
        drawdowns: np.ndarray | list[float] | None = None,
        expected_opportunity: np.ndarray | list[float] | None = None,
        forecast_confidence: np.ndarray | list[float] | None = None,
        model_agreement: np.ndarray | list[float] | None = None,
        regime: str = "normal",
        risk_state: str = "NORMAL",
        scopes: dict | None = None,
        risk_types: dict | None = None,
    ) -> CapitalAllocation:
        cfg = self.settings
        names = [str(n) for n in names]
        n = len(names)
        method_key = str(method).lower().strip()
        reasons: list[str] = [f"method={method_key}"]
        constraints_applied: list[str] = []

        if n == 0:
            empty = CapitalAllocation(
                method=method_key,
                names=[],
                data_version=cfg.data_version,
                model_version=cfg.model_version,
                reasons=["empty_names"],
                confidence=0.0,
            )
            self.last_allocation = empty
            return empty

        cov_m = self._resolve_cov(n, cov=cov, returns=returns, vols=vols)
        budgets = build_risk_budgets(
            names,
            risk_budgets=risk_budgets,
            scopes=scopes,
            risk_types=risk_types,
            data_version=cfg.data_version,
            model_version=cfg.model_version,
            confidence=_mean_unit(forecast_confidence, default=1.0),
        )
        rb_vec = strategy_budget_vector(names, budgets)

        # Correlation-aware effective risk budgets (shared budget for crowded names)
        corr_m = self._resolve_corr(n, cov=cov_m, returns=returns)
        eff = effective_risk_budgets(
            rb_vec,
            corr_m,
            names=names,
            threshold=cfg.correlation_crowding_threshold,
            floor=cfg.correlation_scale_floor,
        )
        rb_eff = eff["effective"]
        corr_adj = {k: float(v) for k, v in eff["scales"].items()}
        reasons.append("correlation_aware_risk_budgets")

        # Raw method weights (never from historical mean returns)
        w_raw, method_meta = self._method_weights(
            method_key,
            names=names,
            cov=cov_m,
            returns=returns,
            vols=vols,
            risk_budgets=rb_eff,
            expected_opportunity=expected_opportunity,
            forecast_confidence=forecast_confidence,
            model_agreement=model_agreement,
            regime=regime,
            risk_state=risk_state,
            capital=capital,
            adv=adv,
            spreads=spreads,
            drawdowns=drawdowns,
            corr=corr_m,
        )
        reasons.extend(method_meta.get("reasons", []))

        # Opportunity tilt only if explicitly provided
        if expected_opportunity is not None and method_key != "dynamic":
            opp = np.asarray(expected_opportunity, dtype=np.float64).ravel()
            if opp.size == n:
                opp = np.maximum(opp, 0.0)
                if float(np.sum(opp)) > 0:
                    opp = opp / float(np.sum(opp))
                    w_raw = w_raw * opp
                    s = float(np.sum(w_raw))
                    if s > 0:
                        w_raw = w_raw / s
                    reasons.append("expected_opportunity_tilt")

        # Capacity adjustment (missing → conservative)
        cap_info = estimate_capacity(
            names,
            capital=capital,
            weights=w_raw,
            adv=adv,
            spreads=spreads,
            vols=vols if vols is not None else np.sqrt(np.maximum(np.diag(cov_m), 0.0)),
            max_participation=cfg.max_participation,
            impact_coeff=cfg.impact_coeff,
            ttl_days=cfg.capacity_ttl_days,
            missing_capacity_scale=cfg.missing_capacity_scale,
            missing_liquidity_scale=cfg.missing_liquidity_scale,
            default_adv=cfg.default_adv,
            default_spread=cfg.default_spread,
        )
        cap_adj = {k: float(v) for k, v in cap_info["scales"].items()}
        w_cap = w_raw * np.asarray([cap_adj[nm] for nm in names], dtype=np.float64)
        if float(np.sum(w_cap)) > 0:
            w_cap = w_cap / float(np.sum(w_cap))
        if cap_info.get("missing_capacity") or cap_info.get("missing_liquidity"):
            reasons.append("missing_capacity_or_liquidity_conservative_downscale")
        else:
            reasons.append("capacity_adjustment_applied")

        # Drawdown adjustment
        dd_info = drawdown_scales(
            names,
            returns=returns,
            drawdowns=drawdowns,
            caution=cfg.drawdown.caution,
            reduced_risk=cfg.drawdown.reduced_risk,
            capital_preservation=cfg.drawdown.capital_preservation,
            trading_halt=cfg.drawdown.trading_halt,
            state_scales=cfg.risk_state_scales,
        )
        dd_adj = {k: float(v) for k, v in dd_info["scales"].items()}
        w_dd = w_cap * np.asarray([dd_adj[nm] for nm in names], dtype=np.float64)
        # Portfolio risk_state ceiling
        rs = str(risk_state).upper()
        rs_scale = float(np.clip(cfg.risk_state_scales.get(rs, 1.0), 0.0, 1.0))
        w_dd = w_dd * rs_scale
        if float(np.sum(w_dd)) > 0:
            w_dd = w_dd / float(np.sum(w_dd))
        else:
            w_dd = np.zeros(n, dtype=np.float64)
            reasons.append("risk_state_or_drawdown_zeroed_weights")
        reasons.append(f"drawdown_and_risk_state_scale={rs_scale:.4f}")

        # Correlation crowding on weights (share effective risk)
        w_corr = w_dd * np.asarray([corr_adj[nm] for nm in names], dtype=np.float64)
        if float(np.sum(w_corr)) > 0:
            w_corr = w_corr / float(np.sum(w_corr))

        # Hard constraint projection — NEVER override limits; preserve halt zeros
        proj = project_weights(w_corr, settings=cfg)
        w_final = np.asarray(proj["weights"], dtype=np.float64)
        constraints_applied.extend(proj["constraints_applied"])

        if float(np.sum(w_final)) > 1e-12:
            # Participation hard clip
            part = apply_participation_constraint(
                w_final,
                capital=capital,
                adv=adv,
                max_participation=cfg.max_participation,
                ttl_days=cfg.capacity_ttl_days,
            )
            w_final = np.asarray(part["weights"], dtype=np.float64)
            if part.get("scaled"):
                constraints_applied.append("participation_cap")
            if float(np.sum(w_final)) > 1e-12:
                w_final = w_final / float(np.sum(w_final))
            else:
                w_final = np.zeros(n, dtype=np.float64)
        else:
            w_final = np.zeros(n, dtype=np.float64)
            reasons.append("zero_capital_after_risk_controls")

        # Capital amounts
        cap_budgets = allocate_capital_budgets(names, w_final, capital=capital)
        amounts = cap_budgets["amounts"]

        # Risk budget usage proportional to allocated weight of total budget mass
        total_rb = float(sum(rb_eff.values())) or 1.0
        used = {names[i]: float(w_final[i] * total_rb) for i in range(n)}
        mark_budgets_used(budgets, used)

        strategies = build_strategy_allocations(
            names,
            w_final,
            capital=capital,
            risk_budgets=rb_eff,
            settings=cfg,
            capacity_scales=cap_adj,
            correlation_scales=corr_adj,
            drawdown_scales=dd_adj,
        )

        conf = _mean_unit(forecast_confidence, default=1.0)
        agree = _mean_unit(model_agreement, default=1.0)
        confidence = float(np.clip(min(conf, agree), 0.0, 1.0))

        diag = diagnose_allocation(cov_m, w_final, names=names)
        eval_out = evaluate_allocation(
            w_final,
            names=names,
            cov=cov_m,
            risk_budgets=rb_eff,
            capacity_scales=cap_adj,
            capital=capital,
            max_notional=cap_info.get("max_notional"),
        )

        allocation = CapitalAllocation(
            timestamp=_utc_now(),
            data_version=cfg.data_version,
            model_version=cfg.model_version,
            method=method_key,
            names=names,
            weights={names[i]: float(w_final[i]) for i in range(n)},
            capital_amounts=amounts,
            risk_budgets_used=used,
            strategies=strategies,
            risk_budgets=budgets,
            inputs={
                "capital": float(capital),
                "has_cov": cov is not None,
                "has_returns": returns is not None,
                "has_vols": vols is not None,
                "has_adv": adv is not None,
                "has_spreads": spreads is not None,
                "has_drawdowns": drawdowns is not None,
                "has_expected_opportunity": expected_opportunity is not None,
                "regime": str(regime),
                "risk_state": rs,
                "scopes": scopes or {},
                "risk_types": risk_types or {},
            },
            params={
                "max_weight": cfg.max_weight,
                "max_gross_exposure": cfg.max_gross_exposure,
                "max_participation": cfg.max_participation,
                "correlation_crowding_threshold": cfg.correlation_crowding_threshold,
            },
            output={
                "method_meta": method_meta,
                "diagnostics": diag,
                "evaluation": eval_out,
                "capacity": {
                    "missing_capacity": cap_info.get("missing_capacity"),
                    "missing_liquidity": cap_info.get("missing_liquidity"),
                    "scores": cap_info.get("scores"),
                },
            },
            constraints_applied=constraints_applied,
            correlation_adjustment=corr_adj,
            capacity_adjustment=cap_adj,
            drawdown_adjustment=dd_adj,
            confidence=confidence,
            reasons=reasons,
        )
        self.last_allocation = allocation
        return allocation

    def allocate_strategy(
        self,
        name: str,
        *,
        weight: float,
        capital: float = 1.0,
        risk_budget: float = 0.0,
    ) -> Any:
        return _allocate_strategy(
            name,
            weight=weight,
            capital=capital,
            risk_budget=risk_budget,
            settings=self.settings,
        )

    def allocate_risk_budget(
        self,
        names: list[str],
        *,
        risk_budgets: dict[str, float] | None = None,
        scopes: dict | None = None,
        risk_types: dict | None = None,
        **kwargs: Any,
    ) -> CapitalAllocation:
        return self.allocate(
            names,
            method="risk_budget",
            risk_budgets=risk_budgets,
            scopes=scopes,
            risk_types=risk_types,
            **kwargs,
        )

    def allocate_capital(
        self,
        names: list[str],
        *,
        capital: float = 1.0,
        method: str = "equal_capital",
        **kwargs: Any,
    ) -> CapitalAllocation:
        return self.allocate(names, method=method, capital=capital, **kwargs)

    def risk_budget(
        self,
        names: list[str],
        *,
        risk_budgets: dict[str, float] | None = None,
        scopes: dict | None = None,
        risk_types: dict | None = None,
        **kwargs: Any,
    ) -> list[RiskBudget]:
        conf = _mean_unit(kwargs.get("forecast_confidence"), default=1.0)
        return build_risk_budgets(
            names,
            risk_budgets=risk_budgets,
            scopes=scopes,
            risk_types=risk_types,
            data_version=self.settings.data_version,
            model_version=self.settings.model_version,
            confidence=conf,
        )

    def capital_budget(
        self,
        names: list[str],
        weights: np.ndarray | list[float] | dict[str, float],
        *,
        capital: float = 1.0,
    ) -> dict[str, Any]:
        return allocate_capital_budgets(names, weights, capital=capital)

    def capacity(
        self,
        names: list[str],
        *,
        capital: float = 1.0,
        weights: np.ndarray | list[float] | None = None,
        adv: np.ndarray | list[float] | None = None,
        spreads: np.ndarray | list[float] | None = None,
        vols: np.ndarray | list[float] | None = None,
    ) -> dict[str, Any]:
        cfg = self.settings
        return estimate_capacity(
            names,
            capital=capital,
            weights=weights,
            adv=adv,
            spreads=spreads,
            vols=vols,
            max_participation=cfg.max_participation,
            impact_coeff=cfg.impact_coeff,
            ttl_days=cfg.capacity_ttl_days,
            missing_capacity_scale=cfg.missing_capacity_scale,
            missing_liquidity_scale=cfg.missing_liquidity_scale,
            default_adv=cfg.default_adv,
            default_spread=cfg.default_spread,
        )

    def optimize(
        self,
        names: list[str],
        *,
        objective: str = "risk_budget_match",
        cov: np.ndarray | None = None,
        returns: np.ndarray | None = None,
        risk_budgets: dict[str, float] | None = None,
        expected_opportunity: np.ndarray | list[float] | None = None,
        capital: float = 1.0,
        **kwargs: Any,
    ) -> CapitalAllocation:
        """Optimize risk budgets under hard constraints, then run full allocate pipeline."""
        n = len(names)
        cov_m = self._resolve_cov(n, cov=cov, returns=returns, vols=kwargs.get("vols"))
        rb_vec = None
        if risk_budgets is not None:
            rb_vec = np.asarray([float(risk_budgets.get(nm, 1.0 / max(n, 1))) for nm in names])
        opt = optimize_risk_budgets(
            cov_m,
            objective=objective,  # type: ignore[arg-type]
            risk_budgets=rb_vec,
            target_vol=float(kwargs.get("target_vol", self.settings.target_volatility)),
            target_cvar=kwargs.get("target_cvar"),
            target_drawdown=kwargs.get("target_drawdown"),
            expected_opportunity=expected_opportunity,
            max_weight=float(self.settings.max_weight),
            max_leverage=float(self.settings.max_leverage),
            names=names,
            returns=returns,
        )
        # Feed optimized budgets into risk-budget allocation (never skip hard pipeline)
        return self.allocate(
            names,
            method="risk_budget",
            cov=cov_m,
            returns=returns,
            risk_budgets=opt["weights"],
            capital=capital,
            expected_opportunity=expected_opportunity,
            **{k: v for k, v in kwargs.items() if k not in ("target_vol", "target_cvar", "target_drawdown", "vols")},
        )

    def allocate_scenarios(
        self,
        *,
        method: str = "risk_parity",
        scenarios: list[str] | None = None,
        capital: float = 1.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> dict[str, CapitalAllocation]:
        """Scenario-aware allocation across synthetic market regimes."""
        kinds = scenarios or [
            "independent",
            "correlated",
            "low_liquidity",
            "high_volatility",
            "regime",
            "drawdown",
            "tail",
        ]
        out: dict[str, CapitalAllocation] = {}
        for kind in kinds:
            try:
                scen = simulate_capital_scenario(kind, seed=seed)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                scen = simulate_capital_scenario("independent", seed=seed)
            names = list(scen.get("names") or [f"s{i}" for i in range(4)])
            rets = scen.get("returns")
            cov = None
            if rets is not None:
                arr = np.asarray(rets, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    cov = np.cov(arr.T)
            out[str(kind)] = self.allocate(
                names,
                method=method,
                cov=cov,
                returns=rets,
                capital=capital,
                vols=scen.get("vols"),
                adv=scen.get("adv"),
                spreads=scen.get("spreads"),
                drawdowns=scen.get("drawdowns"),
                risk_state=str(scen.get("risk_state", "NORMAL")),
                **kwargs,
            )
        return out

    def rebalance(
        self,
        current_weights: np.ndarray | list[float] | dict[str, float],
        target: CapitalAllocation | np.ndarray | list[float] | dict[str, float],
        *,
        names: list[str] | None = None,
        capital: float = 1.0,
        adv: np.ndarray | list[float] | None = None,
        max_turnover: float | None = None,
        max_participation: float | None = None,
    ) -> CapitalAllocation:
        """Move toward target weights subject to turnover / participation caps."""
        cfg = self.settings
        if isinstance(target, CapitalAllocation):
            keys = list(target.names)
            t_w = np.asarray([float(target.weights.get(k, 0.0)) for k in keys], dtype=np.float64)
            base = target
        else:
            keys = names or (
                list(current_weights.keys())
                if isinstance(current_weights, dict)
                else [f"s{i}" for i in range(len(np.asarray(current_weights).ravel()))]
            )
            if isinstance(target, dict):
                t_w = np.asarray([float(target.get(k, 0.0)) for k in keys], dtype=np.float64)
            else:
                t_w = np.asarray(target, dtype=np.float64).ravel()
            base = None

        if isinstance(current_weights, dict):
            c_w = np.asarray([float(current_weights.get(k, 0.0)) for k in keys], dtype=np.float64)
        else:
            c_w = np.asarray(current_weights, dtype=np.float64).ravel()
            if c_w.size != len(keys):
                c_w = np.zeros(len(keys), dtype=np.float64)

        turn_cap = float(
            cfg.rebalance_turnover_cap if max_turnover is None else max_turnover
        )
        part_cap = float(
            cfg.rebalance_participation_cap if max_participation is None else max_participation
        )

        turned = apply_turnover_constraint(c_w, t_w, max_turnover=turn_cap)
        w = np.asarray(turned["weights"], dtype=np.float64)
        part = apply_participation_constraint(
            w,
            capital=capital,
            adv=adv,
            max_participation=part_cap,
            ttl_days=cfg.capacity_ttl_days,
        )
        w = np.asarray(part["weights"], dtype=np.float64)
        proj = project_weights(w, settings=cfg)
        w = np.asarray(proj["weights"], dtype=np.float64)
        if float(np.sum(w)) > 0:
            w = w / float(np.sum(w))

        caps = allocate_capital_budgets(keys, w, capital=capital)
        strategies = build_strategy_allocations(keys, w, capital=capital, settings=cfg)

        constraints = list(proj["constraints_applied"])
        if turned.get("scaled"):
            constraints.append("turnover_cap")
        if part.get("scaled"):
            constraints.append("participation_cap")

        allocation = CapitalAllocation(
            timestamp=_utc_now(),
            data_version=cfg.data_version,
            model_version=cfg.model_version,
            method="rebalance",
            names=keys,
            weights={keys[i]: float(w[i]) for i in range(len(keys))},
            capital_amounts=caps["amounts"],
            risk_budgets_used=(base.risk_budgets_used if base else {}),
            strategies=strategies,
            risk_budgets=list(base.risk_budgets) if base else [],
            inputs={
                "current_weights": (
                    current_weights
                    if isinstance(current_weights, dict)
                    else c_w.tolist()
                ),
                "turnover": turned.get("turnover"),
            },
            params={"max_turnover": turn_cap, "max_participation": part_cap},
            output={"turnover_scaled": turned.get("scaled"), "participation_scaled": part.get("scaled")},
            constraints_applied=constraints,
            correlation_adjustment=dict(base.correlation_adjustment) if base else {},
            capacity_adjustment=dict(base.capacity_adjustment) if base else {},
            drawdown_adjustment=dict(base.drawdown_adjustment) if base else {},
            confidence=float(base.confidence) if base else 1.0,
            reasons=["rebalance_toward_target"],
        )
        self.last_allocation = allocation
        return allocation

    def export_state(self) -> dict[str, Any]:
        return {
            "settings": self.settings.model_dump(),
            "last_allocation": (
                self.last_allocation.to_dict() if self.last_allocation is not None else None
            ),
        }

    # --------------------------------------------------------------- helpers
    def _resolve_cov(
        self,
        n: int,
        *,
        cov: np.ndarray | None,
        returns: np.ndarray | None,
        vols: np.ndarray | list[float] | None,
    ) -> np.ndarray:
        if cov is not None:
            c = np.asarray(cov, dtype=np.float64)
            if c.shape == (n, n):
                return 0.5 * (c + c.T)
        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            if r.ndim == 2 and r.shape[1] >= n:
                cm = covariance_matrix(r[:, :n])
                c = np.asarray(cm["matrix"], dtype=np.float64)
                if c.shape == (n, n):
                    return 0.5 * (c + c.T)
        if vols is not None:
            v = np.asarray(vols, dtype=np.float64).ravel()
            if v.size == n:
                return np.diag(np.maximum(v, self.settings.vol_floor) ** 2)
        return np.eye(n, dtype=np.float64) * (0.01**2)

    def _resolve_corr(
        self,
        n: int,
        *,
        cov: np.ndarray,
        returns: np.ndarray | None,
    ) -> np.ndarray:
        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            if r.ndim == 2 and r.shape[1] >= n:
                out = strategy_correlation(r[:, :n])
                m = np.asarray(out["matrix"], dtype=np.float64)
                if m.shape == (n, n):
                    return m
        vol = np.sqrt(np.maximum(np.diag(cov), 1e-18))
        denom = np.outer(vol, vol)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def _method_weights(
        self,
        method: str,
        *,
        names: list[str],
        cov: np.ndarray,
        returns: np.ndarray | None,
        vols: np.ndarray | list[float] | None,
        risk_budgets: dict[str, float],
        expected_opportunity: np.ndarray | list[float] | None,
        forecast_confidence: np.ndarray | list[float] | None,
        model_agreement: np.ndarray | list[float] | None,
        regime: str,
        risk_state: str,
        capital: float,
        adv: np.ndarray | list[float] | None,
        spreads: np.ndarray | list[float] | None,
        drawdowns: np.ndarray | list[float] | None,
        corr: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.settings
        n = len(names)
        meta: dict[str, Any] = {"method": method, "reasons": []}

        if method == "equal_capital":
            w = np.full(n, 1.0 / n)
            meta["reasons"].append("equal_capital")
            return w, meta

        if method in ("equal_risk",):
            out = equal_risk_weights(
                cov,
                names=names,
                max_iter=cfg.risk_parity_max_iter,
                tol=cfg.risk_parity_tol,
            )
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["base"] = {k: out[k] for k in ("converged", "iterations") if k in out}
            meta["reasons"].append("equal_risk_contribution")
            return w, meta

        if method in ("risk_parity",):
            out = capital_risk_parity(
                cov,
                names=names,
                risk_budgets=None,
                max_iter=cfg.risk_parity_max_iter,
                tol=cfg.risk_parity_tol,
            )
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["base"] = {k: out[k] for k in ("converged", "iterations") if k in out}
            meta["reasons"].append("risk_parity")
            return w, meta

        if method in ("risk_budget",):
            out = capital_risk_parity(
                cov,
                names=names,
                risk_budgets=risk_budgets,
                max_iter=cfg.risk_parity_max_iter,
                tol=cfg.risk_parity_tol,
            )
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["reasons"].append("risk_budget_parity")
            return w, meta

        if method in ("volatility",):
            out = volatility_budgets(
                names,
                vols=vols,
                returns=returns,
                cov=cov,
                target_volatility=cfg.target_volatility,
                vol_floor=cfg.vol_floor,
                risk_budgets=risk_budgets,
            )
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["reasons"].append("volatility_budgeting")
            return w, meta

        if method == "hrp":
            out = hrp_weights(cov, names=names, corr=corr, linkage=cfg.hrp_linkage)
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["order"] = out.get("order")
            meta["reasons"].append("hierarchical_risk_parity")
            return w, meta

        if method == "herc":
            out = herc_weights(cov, names=names, corr=corr, linkage=cfg.hrp_linkage)
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["order"] = out.get("order")
            meta["reasons"].append("hierarchical_equal_risk_contribution")
            return w, meta

        if method == "correlation":
            scales = correlation_crowding_scales(
                corr,
                threshold=cfg.correlation_crowding_threshold,
                floor=cfg.correlation_scale_floor,
                names=names,
            )
            # Start from risk parity then apply crowding scales as weights
            base = capital_risk_parity(cov, names=names)
            w = np.asarray(base["weight_vector"], dtype=np.float64)
            w = w * np.asarray([scales[nm] for nm in names], dtype=np.float64)
            if float(np.sum(w)) > 0:
                w = w / float(np.sum(w))
            meta["reasons"].append("correlation_crowding_allocation")
            return w, meta

        if method == "drawdown":
            dd = drawdown_scales(
                names,
                returns=returns,
                drawdowns=drawdowns,
                caution=cfg.drawdown.caution,
                reduced_risk=cfg.drawdown.reduced_risk,
                capital_preservation=cfg.drawdown.capital_preservation,
                trading_halt=cfg.drawdown.trading_halt,
                state_scales=cfg.risk_state_scales,
            )
            base = capital_risk_parity(cov, names=names)
            w = np.asarray(base["weight_vector"], dtype=np.float64)
            w = w * np.asarray([dd["scales"][nm] for nm in names], dtype=np.float64)
            if float(np.sum(w)) > 0:
                w = w / float(np.sum(w))
            else:
                w = np.zeros(n)
            meta["reasons"].append("drawdown_aware_allocation")
            return w, meta

        if method == "capacity":
            base = np.full(n, 1.0 / n)
            cap = estimate_capacity(
                names,
                capital=capital,
                weights=base,
                adv=adv,
                spreads=spreads,
                vols=vols,
                max_participation=cfg.max_participation,
                impact_coeff=cfg.impact_coeff,
                ttl_days=cfg.capacity_ttl_days,
                missing_capacity_scale=cfg.missing_capacity_scale,
                missing_liquidity_scale=cfg.missing_liquidity_scale,
                default_adv=cfg.default_adv,
                default_spread=cfg.default_spread,
            )
            w = base * np.asarray([cap["scales"][nm] for nm in names], dtype=np.float64)
            if float(np.sum(w)) > 0:
                w = w / float(np.sum(w))
            meta["reasons"].append("capacity_driven_allocation")
            return w, meta

        if method == "dynamic":
            out = dynamic_risk_scales(
                names,
                settings=cfg,
                cov=cov,
                returns=returns,
                vols=vols,
                adv=adv,
                spreads=spreads,
                drawdowns=drawdowns,
                forecast_confidence=forecast_confidence,
                model_agreement=model_agreement,
                expected_opportunity=expected_opportunity,
                regime=regime,
                risk_state=risk_state,
                capital=capital,
            )
            w = np.asarray(out["weight_vector"], dtype=np.float64)
            meta["dynamic"] = {
                "portfolio_scale": out.get("portfolio_scale"),
                "regime_scale": out.get("regime_scale"),
                "opportunity_applied": out.get("opportunity_applied"),
            }
            meta["reasons"].append("dynamic_risk_budgeting")
            return w, meta

        # Fallback: risk parity
        out = capital_risk_parity(cov, names=names)
        w = np.asarray(out["weight_vector"], dtype=np.float64)
        meta["reasons"].append(f"unknown_method_{method}_fallback_risk_parity")
        return w, meta


def _mean_unit(values: Any, *, default: float) -> float:
    if values is None:
        return float(default)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return float(default)
    return float(np.clip(np.nanmean(arr), 0.0, 1.0))
