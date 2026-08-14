"""Institutional Risk Intelligence Engine — sole gate between alpha and execution.

Architectural rules enforced here:
1. Risk never generates alpha.
2. Hard limits cannot be overridden by forecast confidence.
3. Every rejection has an explicit reason and audit trail.
4. Point-in-time correctness — no future information.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.risk.base import (
    LimitBreach,
    LimitSeverity,
    RiskDecision,
    RiskReport,
    RiskState,
    as_returns,
    as_weights,
    build_report,
)
from iqrp.app.risk.config import RiskSettings
from iqrp.app.risk.leverage.dynamic_leverage import recommended_leverage
from iqrp.app.risk.leverage.leverage_limits import clip_leverage
from iqrp.app.risk.limits import build_default_limits, check_all_limits
from iqrp.app.risk.market.liquidity import liquidity_risk
from iqrp.app.risk.market.volatility import realized_volatility
from iqrp.app.risk.model_risk.forecast_uncertainty import forecast_uncertainty
from iqrp.app.risk.model_risk.model_disagreement import model_disagreement
from iqrp.app.risk.model_risk.model_drift import model_drift
from iqrp.app.risk.monitoring.alerts import build_alerts
from iqrp.app.risk.monitoring.breaches import summarize_breaches
from iqrp.app.risk.monitoring.dashboards import dashboard_payload
from iqrp.app.risk.monitoring.risk_monitor import RiskMonitor
from iqrp.app.risk.portfolio.concentration import concentration_risk
from iqrp.app.risk.portfolio.exposure import exposure_summary
from iqrp.app.risk.portfolio.portfolio_risk import (
    component_risk_contribution,
    marginal_risk_contribution,
    portfolio_risk,
)
from iqrp.app.risk.serializer import RiskSerializer
from iqrp.app.risk.sizing.drawdown_adjusted import drawdown_adjusted_size
from iqrp.app.risk.sizing.fractional_kelly import fractional_kelly
from iqrp.app.risk.sizing.kelly import kelly_fraction
from iqrp.app.risk.sizing.volatility_target import (
    confidence_adjusted_size,
    fixed_fractional_size,
    regime_adjusted_size,
    volatility_target_size,
)
from iqrp.app.risk.stress.historical import historical_stress
from iqrp.app.risk.stress.hypothetical import hypothetical_stress
from iqrp.app.risk.stress.reverse_stress import reverse_stress
from iqrp.app.risk.tail.cvar import historical_cvar, monte_carlo_cvar, parametric_cvar
from iqrp.app.risk.tail.drawdown import drawdown_state
from iqrp.app.risk.tail.expected_shortfall import expected_shortfall
from iqrp.app.risk.tail.var import (
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)


class RiskIntelligenceEngine:
    """Central risk gate for pre-trade validation and portfolio risk analytics."""

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings.default()
        self._monitor = RiskMonitor(
            caution=self.settings.drawdown.caution,
            reduced_risk=self.settings.drawdown.reduced_risk,
            capital_preservation=self.settings.drawdown.capital_preservation,
            trading_halt=self.settings.drawdown.trading_halt,
            var_confidence=self.settings.var.confidence,
        )
        self._serializer = RiskSerializer()
        self._audit_log: list[dict[str, Any]] = []
        self._last_report: RiskReport | None = None

    def calculate_risk(
        self,
        returns: np.ndarray | list[float],
        *,
        weights: np.ndarray | list[float] | None = None,
    ) -> RiskReport:
        r = as_returns(returns)
        vol = realized_volatility(r)
        dd = self.drawdown(r)
        state = RiskState(dd["risk_state"])
        tail = {
            "var": self.var(r).to_dict(),
            "cvar": self.cvar(r).to_dict(),
            "expected_shortfall": self.expected_shortfall(r).to_dict(),
        }
        w = as_weights(weights) if weights is not None else np.asarray([1.0])
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > 1:
            cov = np.cov(arr.T)
            w = as_weights(w, n=cov.shape[0])
            port = portfolio_risk(w, cov)
        else:
            port = {"portfolio_volatility": vol.to_dict(), "weights": w.tolist()}
        conc = concentration_risk(w)
        exp = exposure_summary(w)
        limits = {
            L.name: L.to_dict()
            for L in build_default_limits(
                max_position=self.settings.limits.max_position,
                max_gross_exposure=self.settings.limits.max_gross_exposure,
                max_net_exposure=self.settings.limits.max_net_exposure,
                max_concentration=self.settings.limits.max_concentration,
                max_daily_loss=self.settings.limits.max_daily_loss,
                max_drawdown=self.settings.drawdown.trading_halt,
                max_participation=self.settings.limits.max_participation,
                min_adv_coverage=self.settings.limits.min_adv_coverage,
            )
        }
        breaches = check_all_limits(
            weights=w,
            current_drawdown=float(dd["current_drawdown"]),
            max_position=self.settings.limits.max_position,
            max_gross_exposure=self.settings.limits.max_gross_exposure,
            max_net_exposure=self.settings.limits.max_net_exposure,
            max_concentration=self.settings.limits.max_concentration,
            max_daily_loss=self.settings.limits.max_daily_loss,
            max_drawdown=self.settings.drawdown.trading_halt,
        )
        report = build_report(
            portfolio_risk=port if isinstance(port, dict) else {"value": port},
            position_risk={"weights": w.tolist()},
            tail_risk=tail,
            liquidity_risk={
                "note": "call liquidity_risk() with ADV/spread for position-level metrics"
            },
            concentration=conc if isinstance(conc, dict) else {},
            factor_exposure={},
            drawdown=dd,
            stress={},
            limits=limits,
            breaches=[b.to_dict() for b in breaches],
            risk_state=state,
            timestamp=datetime.now(UTC).isoformat(),
            data_version=self.settings.data_version,
            model_version=self.settings.model_version,
            metadata={
                "exposure": exp if isinstance(exp, dict) else {},
                "volatility": vol.to_dict(),
            },
        )
        self._last_report = report
        if r.size:
            self._monitor.update(float(r[-1]), measures={"var": tail["var"]}, breaches=breaches)
        return report

    def portfolio_risk(self, weights: Any, cov: Any) -> dict[str, Any]:
        return portfolio_risk(weights, cov)

    def var(
        self,
        returns: Any,
        *,
        method: str | None = None,
        confidence: float | None = None,
        horizon: int | None = None,
    ) -> Any:
        m = method or self.settings.var.method
        conf = confidence if confidence is not None else self.settings.var.confidence
        h = horizon if horizon is not None else self.settings.var.horizon
        if m == "parametric":
            return parametric_var(returns, confidence=conf, horizon=h)
        if m == "monte_carlo":
            return monte_carlo_var(
                returns,
                confidence=conf,
                horizon=h,
                n_simulations=self.settings.var.n_simulations,
                seed=self.settings.seed,
            )
        if m == "fhs":
            return filtered_historical_var(returns, confidence=conf, horizon=h)
        return historical_var(returns, confidence=conf, horizon=h)

    def cvar(
        self,
        returns: Any,
        *,
        method: str | None = None,
        confidence: float | None = None,
    ) -> Any:
        m = method or self.settings.es.method
        conf = confidence if confidence is not None else self.settings.es.confidence
        if m == "parametric":
            return parametric_cvar(returns, confidence=conf)
        if m == "monte_carlo":
            return monte_carlo_cvar(
                returns,
                confidence=conf,
                n_simulations=self.settings.monte_carlo.n_simulations,
                seed=self.settings.seed,
            )
        return historical_cvar(returns, confidence=conf)

    def expected_shortfall(self, returns: Any, *, confidence: float | None = None) -> Any:
        conf = confidence if confidence is not None else self.settings.es.confidence
        return expected_shortfall(
            returns,
            confidence=conf,
            method=self.settings.es.method,
            n_simulations=self.settings.monte_carlo.n_simulations,
            seed=self.settings.seed,
        )

    def stress_test(
        self,
        weights: Any,
        returns: Any | None = None,
        *,
        event_indices: np.ndarray | list[int] | None = None,
        shocks: Any | None = None,
        cov: Any | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if returns is not None and event_indices is not None:
            out["historical"] = historical_stress(
                returns, event_window=np.asarray(event_indices), weights=weights
            )
        if shocks is not None and cov is not None:
            out["hypothetical"] = hypothetical_stress(weights, cov, shocks)
        elif shocks is not None:
            w = as_weights(weights)
            shock_vec = np.asarray(
                list(shocks.values()) if isinstance(shocks, dict) else shocks, dtype=np.float64
            )
            if shock_vec.size == 1:
                shock_vec = np.full(w.size, float(shock_vec[0]))
            out["hypothetical"] = hypothetical_stress(weights, np.eye(w.size), shock_vec)
        return out

    def reverse_stress(
        self,
        weights: Any,
        *,
        loss_limit: float | None = None,
        direction: Any | None = None,
    ) -> dict[str, Any]:
        w = as_weights(weights)
        u = (
            np.asarray(direction, dtype=np.float64).reshape(-1)
            if direction is not None
            else np.ones(w.size)
        )
        return reverse_stress(
            w,
            u,
            loss_limit=(
                loss_limit if loss_limit is not None else self.settings.limits.max_daily_loss
            ),
        )

    def position_size(
        self,
        *,
        realized_vol: float,
        edge: float = 0.0,
        win_prob: float | None = None,
        current_drawdown: float = 0.0,
        confidence: float = 1.0,
        regime: str = "normal",
        equity: float = 1.0,
        method: str | None = None,
    ) -> dict[str, Any]:
        m = method or self.settings.sizing.method
        cfg = self.settings.sizing
        if m == "kelly":
            size = kelly_fraction(edge=edge, win_prob=win_prob, max_kelly=cfg.max_kelly)
        elif m == "fractional_kelly":
            size = fractional_kelly(
                edge=edge,
                win_prob=win_prob,
                fraction=cfg.kelly_fraction,
                max_kelly=cfg.max_kelly,
            )
        elif m == "fixed_fractional":
            size = fixed_fractional_size(equity=equity, risk_fraction=cfg.risk_per_trade)
        elif m == "drawdown_adjusted":
            base = volatility_target_size(
                realized_vol=realized_vol,
                target_vol=cfg.target_volatility,
                max_leverage=cfg.max_leverage,
            ).value
            size = drawdown_adjusted_size(
                base_size=base,
                current_drawdown=current_drawdown,
                max_drawdown_limit=self.settings.drawdown.trading_halt,
            )
        else:
            size = volatility_target_size(
                realized_vol=realized_vol,
                target_vol=cfg.target_volatility,
                max_leverage=cfg.max_leverage,
            )
        adj = confidence_adjusted_size(base_size=size.value, confidence=confidence)
        regime_adj = regime_adjusted_size(base_size=adj.value, regime=regime)
        final = float(np.clip(regime_adj.value, 0.0, cfg.max_leverage))
        return {
            "method": m,
            "size": final,
            "raw": size.to_dict(),
            "confidence_adjusted": adj.to_dict(),
            "regime_adjusted": regime_adj.to_dict(),
            "note": "Hard max_leverage / max_kelly caps always apply; confidence cannot authorize unlimited size.",
        }

    def risk_contribution(self, weights: Any, cov: Any) -> dict[str, Any]:
        return {
            "marginal": marginal_risk_contribution(weights, cov),
            "component": component_risk_contribution(weights, cov),
        }

    def exposure(self, weights: Any) -> dict[str, Any]:
        return exposure_summary(weights)

    def liquidity_risk(
        self,
        *,
        position_size: float | None = None,
        notional: float | None = None,
        adv: float,
        spread: float,
        price: float = 1.0,
        volatility: float = 0.0,
        max_participation: float | None = None,
        impact_coeff: float = 0.1,
    ) -> dict[str, Any]:
        size = float(
            position_size
            if position_size is not None
            else (notional if notional is not None else 0.0)
        )
        return liquidity_risk(
            position_size=size,
            adv=adv,
            spread=spread,
            price=price,
            volatility=volatility,
            max_participation=(
                max_participation
                if max_participation is not None
                else self.settings.limits.max_participation
            ),
            impact_coeff=impact_coeff,
        )

    def drawdown(self, returns: Any) -> dict[str, Any]:
        cfg = self.settings.drawdown
        return drawdown_state(
            returns,
            caution=cfg.caution,
            reduced_risk=cfg.reduced_risk,
            capital_preservation=cfg.capital_preservation,
            trading_halt=cfg.trading_halt,
        )

    def risk_state(self, returns: Any) -> RiskState:
        return RiskState(self.drawdown(returns)["risk_state"])

    def check_limits(
        self,
        *,
        weights: Any | None = None,
        daily_loss: float = 0.0,
        current_drawdown: float = 0.0,
        participation: float | None = None,
        adv_coverage: float | None = None,
    ) -> list[LimitBreach]:
        return check_all_limits(
            weights=weights,
            daily_loss=daily_loss,
            current_drawdown=current_drawdown,
            participation=participation,
            adv_coverage=adv_coverage,
            max_position=self.settings.limits.max_position,
            max_gross_exposure=self.settings.limits.max_gross_exposure,
            max_net_exposure=self.settings.limits.max_net_exposure,
            max_concentration=self.settings.limits.max_concentration,
            max_daily_loss=self.settings.limits.max_daily_loss,
            max_drawdown=self.settings.drawdown.trading_halt,
            max_participation=self.settings.limits.max_participation,
            min_adv_coverage=self.settings.limits.min_adv_coverage,
        )

    def validate_position(
        self,
        *,
        proposed_weight: float,
        weights: Any,
        returns: Any,
        realized_vol: float | None = None,
        participation: float | None = None,
        adv_coverage: float | None = None,
        forecast_confidence: float = 0.0,
        asset_index: int = 0,
    ) -> RiskDecision:
        """Mandatory pre-trade gate. Forecast confidence cannot override hard limits."""
        w = as_weights(weights).copy()
        if 0 <= asset_index < w.size:
            w[asset_index] = float(proposed_weight)
        else:
            w = np.append(w, float(proposed_weight))

        r = as_returns(returns)
        dd = self.drawdown(r)
        state = RiskState(dd["risk_state"])
        vol = float(realized_vol) if realized_vol is not None else realized_volatility(r).value

        breaches = self.check_limits(
            weights=w,
            current_drawdown=float(dd["current_drawdown"]),
            participation=participation,
            adv_coverage=adv_coverage,
        )
        lev = float(np.sum(np.abs(w)))
        if lev > self.settings.limits.max_leverage + 1e-12:
            breaches.append(
                LimitBreach(
                    limit_name="max_leverage",
                    severity=LimitSeverity.HARD,
                    observed=lev,
                    threshold=self.settings.limits.max_leverage,
                    reason=f"leverage {lev:.4f} exceeds hard max {self.settings.limits.max_leverage}",
                    scope="portfolio",
                )
            )

        hard = [b for b in breaches if b.severity == LimitSeverity.HARD]
        if hard:
            reason = (
                "REJECTED: hard limit breach(es)"
                + (
                    f"; forecast confidence={forecast_confidence:.2f} cannot override hard risk limits"
                    if forecast_confidence > 0
                    else ""
                )
                + ". "
                + "; ".join(b.reason for b in hard)
            )
            approved = False
        elif state == RiskState.TRADING_HALT:
            reason = "REJECTED: risk state TRADING_HALT — capital preservation halt active"
            approved = False
        elif breaches:
            reason = "APPROVED_WITH_WARNINGS: " + "; ".join(b.reason for b in breaches)
            approved = True
        else:
            reason = "APPROVED: all risk checks passed"
            approved = True

        sizing = self.position_size(
            realized_vol=vol,
            current_drawdown=float(dd["current_drawdown"]),
            confidence=forecast_confidence,
        )
        lev_rec = self.recommended_leverage(
            realized_vol=vol,
            current_drawdown=float(dd["current_drawdown"]),
            confidence=forecast_confidence,
        )
        decision = RiskDecision(
            approved=approved,
            reason=reason,
            risk_state=state,
            breaches=breaches,
            recommended_size=float(sizing["size"]),
            recommended_leverage=float(lev_rec.value),
            audit={
                "timestamp": datetime.now(UTC).isoformat(),
                "proposed_weight": float(proposed_weight),
                "asset_index": int(asset_index),
                "forecast_confidence": float(forecast_confidence),
                "data_version": self.settings.data_version,
                "model_version": self.settings.model_version,
                "parameters": {
                    "limits": self.settings.limits.model_dump(),
                    "drawdown": self.settings.drawdown.model_dump(),
                    "leverage": self.settings.leverage.model_dump(),
                },
                "drawdown": dd,
                "limits_checked": True,
            },
        )
        self._audit_log.append(decision.to_dict())
        return decision

    def recommended_leverage(
        self,
        *,
        realized_vol: float,
        forecast_vol: float | None = None,
        current_drawdown: float = 0.0,
        confidence: float = 1.0,
        liquidity_score: float = 1.0,
        regime: str = "normal",
    ) -> Any:
        cfg = self.settings.leverage
        vol = max(
            float(realized_vol), float(forecast_vol if forecast_vol is not None else realized_vol)
        )
        rec = recommended_leverage(
            realized_vol=vol,
            target_vol=self.settings.sizing.target_volatility,
            current_drawdown=current_drawdown,
            max_drawdown=self.settings.drawdown.trading_halt,
            confidence=confidence,
            liquidity_score=liquidity_score,
            regime=regime,
            base_leverage=cfg.base_leverage,
            max_leverage=cfg.max_leverage,
            min_leverage=cfg.min_leverage,
            confidence_cap=cfg.confidence_cap,
        )
        return clip_leverage(
            rec.value, min_leverage=cfg.min_leverage, max_leverage=cfg.max_leverage
        )

    def model_risk_assessment(
        self,
        forecasts: np.ndarray | dict[str, np.ndarray] | None = None,
        *,
        realizations: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if forecasts is not None:
            if isinstance(forecasts, dict):
                stack = np.stack(
                    [np.asarray(v, dtype=np.float64).reshape(-1) for v in forecasts.values()],
                    axis=0,
                )
            else:
                stack = np.asarray(forecasts, dtype=np.float64)
            out["disagreement"] = model_disagreement(stack).to_dict()
            if realizations is not None:
                # use mean forecast across models
                mean_fc = stack.mean(axis=0) if stack.ndim == 2 else stack
                out["uncertainty"] = forecast_uncertainty(mean_fc, realizations).to_dict()
        if residuals is not None:
            out["drift"] = model_drift(residuals).to_dict()
        return out

    def monitor_snapshot(self) -> dict[str, Any]:
        snap = self._monitor.snapshot()
        state = self._last_report.risk_state if self._last_report else RiskState.NORMAL
        breach_dicts = self._last_report.breaches if self._last_report else []
        typed: list[LimitBreach] = []
        for b in breach_dicts:
            if isinstance(b, LimitBreach):
                typed.append(b)
            elif isinstance(b, dict) and "limit_name" in b:
                sev = b.get("severity", "WARNING")
                typed.append(
                    LimitBreach(
                        limit_name=str(b["limit_name"]),
                        severity=sev if isinstance(sev, LimitSeverity) else LimitSeverity(str(sev)),
                        observed=float(b.get("observed", 0.0)),
                        threshold=float(b.get("threshold", 0.0)),
                        reason=str(b.get("reason", "")),
                        scope=str(b.get("scope", "portfolio")),
                        metadata=dict(b.get("metadata") or {}),
                    )
                )
        alerts = build_alerts(breaches=typed, risk_state=state)
        return {
            "snapshot": snap if isinstance(snap, dict) else snap,
            "alerts": alerts,
            "breaches": summarize_breaches(breach_dicts),
            "dashboard": dashboard_payload(
                risk_state=state,
                portfolio_risk=self._last_report.portfolio_risk if self._last_report else None,
                tail_risk=self._last_report.tail_risk if self._last_report else None,
                drawdown=self._last_report.drawdown if self._last_report else None,
                breaches=breach_dicts,
            ),
        }

    def save(self, path: str | Path) -> Path:
        return self._serializer.save(self, path)

    @classmethod
    def load(cls, path: str | Path, settings: RiskSettings | None = None) -> RiskIntelligenceEngine:
        ser = RiskSerializer()
        payload = ser.load(path)
        eng = cls(settings=settings or RiskSettings.default())
        eng.import_state(payload)
        return eng

    def export_state(self) -> dict[str, Any]:
        return {
            "settings": self.settings.model_dump(),
            "audit_log": list(self._audit_log[-100:]),
            "last_report": None if self._last_report is None else self._last_report.to_dict(),
            "data_version": self.settings.data_version,
            "model_version": self.settings.model_version,
        }

    def import_state(self, payload: dict[str, Any]) -> RiskIntelligenceEngine:
        if "settings" in payload:
            try:
                self.settings = RiskSettings.from_mapping(payload["settings"])
            except Exception:
                pass
        self._audit_log = list(payload.get("audit_log") or [])
        return self
