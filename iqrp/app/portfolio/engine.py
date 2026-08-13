"""PortfolioConstructionEngine — institutional portfolio construction facade."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from iqrp.app.portfolio.base.optimizer import OptimizationResult
from iqrp.app.portfolio.base.portfolio import Portfolio
from iqrp.app.portfolio.base.position import Position
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.construction.constructor import PortfolioConstructor, PortfolioResult
from iqrp.app.portfolio.construction.rebalance import RebalanceBands, RebalancePlan, plan_rebalance
from iqrp.app.portfolio.construction.signal_to_weight import signals_to_raw_weights
from iqrp.app.portfolio.construction.target_positions import TargetPositions, weights_to_positions
from iqrp.app.portfolio.construction.target_weights import TargetWeights, build_target_weights
from iqrp.app.portfolio.constraints import check_all_constraints
from iqrp.app.portfolio.constraints.turnover import turnover as one_way_turnover
from iqrp.app.portfolio.covariance import (
    ewma_covariance,
    factor_covariance,
    ledoit_wolf_covariance,
    robust_covariance,
    sample_covariance,
    shrinkage_covariance,
)
from iqrp.app.portfolio.diagnostics import portfolio_diagnostics
from iqrp.app.portfolio.expected_returns import (
    black_litterman_posterior,
    forecast_expected_returns,
    historical_expected_returns,
    shrinkage_expected_returns,
)
from iqrp.app.portfolio.optimization import (
    optimize_black_litterman,
    optimize_cvar,
    optimize_drawdown,
    optimize_entropy,
    optimize_herc,
    optimize_hrp,
    optimize_maximum_diversification,
    optimize_maximum_sharpe,
    optimize_mean_variance,
    optimize_minimum_variance,
    optimize_risk_parity,
    optimize_robust,
    optimize_turnover,
)
from iqrp.app.portfolio.portfolio_risk import risk_contribution as pr_risk_contribution
from iqrp.app.portfolio.serializer import PortfolioSerializer
from iqrp.app.portfolio.transaction_costs import total_transaction_cost

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ValidationReport:
    """Constraint + optional Risk Intelligence pre-trade validation."""

    valid: bool
    violations: list[Any] = field(default_factory=list)
    hard_violations: list[Any] = field(default_factory=list)
    soft_violations: list[Any] = field(default_factory=list)
    risk_decision: dict[str, Any] | None = None
    risk_breaches: list[Any] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _v(x: Any) -> Any:
            return x.to_dict() if hasattr(x, "to_dict") else x

        return {
            "valid": bool(self.valid),
            "violations": [_v(v) for v in self.violations],
            "hard_violations": [_v(v) for v in self.hard_violations],
            "soft_violations": [_v(v) for v in self.soft_violations],
            "risk_decision": self.risk_decision,
            "risk_breaches": [_v(b) for b in self.risk_breaches],
            "messages": list(self.messages),
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
        }

    def __str__(self) -> str:  # pragma: no cover - smoke print
        return (
            f"ValidationReport(valid={self.valid}, "
            f"hard={len(self.hard_violations)}, soft={len(self.soft_violations)}, "
            f"risk={None if self.risk_decision is None else self.risk_decision.get('action', self.risk_decision.get('status'))})"
        )


_OPTIMIZER_MAP: dict[str, Callable[..., dict[str, Any]]] = {
    "min_variance": optimize_minimum_variance,
    "minimum_variance": optimize_minimum_variance,
    "mean_variance": optimize_mean_variance,
    "max_sharpe": optimize_maximum_sharpe,
    "maximum_sharpe": optimize_maximum_sharpe,
    "max_diversification": optimize_maximum_diversification,
    "maximum_diversification": optimize_maximum_diversification,
    "risk_parity": optimize_risk_parity,
    "erc": optimize_risk_parity,
    "hrp": optimize_hrp,
    "herc": optimize_herc,
    "min_cvar": optimize_cvar,
    "cvar": optimize_cvar,
    "drawdown": optimize_drawdown,
    "turnover_aware": optimize_turnover,
    "turnover": optimize_turnover,
    "robust": optimize_robust,
    "black_litterman": optimize_black_litterman,
    "entropy": optimize_entropy,
}


def _extract_weights(raw: Any, n: int | None = None) -> list[float]:
    if raw is None:
        return [0.0] * (n or 0)
    if isinstance(raw, dict):
        if n is not None and n > 0:
            # preserve insertion order of dict values if length matches
            vals = list(raw.values())
            if len(vals) == n:
                return [float(v) for v in vals]
            keys = sorted(raw.keys())
            return [float(raw[k]) for k in keys]
        return [float(v) for v in raw.values()]
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr.tolist()]


def dict_to_optimization_result(
    raw: dict[str, Any],
    *,
    names: Sequence[str] | None = None,
    method: str = "",
    data_version: str = "1.0.0",
    model_version: str = "1.0.0",
    seed: int | None = None,
    mu: Any | None = None,
    cov: Any | None = None,
    fallback_used: bool = False,
    fallback_kind: str | None = None,
) -> OptimizationResult:
    """Convert optimizer dict output into OptimizationResult."""
    w_raw = raw.get("weights")
    w = _extract_weights(w_raw)
    name_list: list[str]
    if isinstance(w_raw, dict):
        name_list = list(w_raw.keys())
    elif names is not None:
        name_list = list(names)
    else:
        name_list = [f"a{i}" for i in range(len(w))]
    if len(name_list) != len(w):
        name_list = [f"a{i}" for i in range(len(w))]

    success = bool(raw.get("success", False))
    status = str(raw.get("status", "ok" if success else "failed"))
    reason = raw.get("failure_reason")

    expected_return = None
    expected_variance = None
    w_arr = np.asarray(w, dtype=np.float64)
    try:
        if mu is not None and w_arr.size:
            m = np.asarray(mu, dtype=np.float64).reshape(-1)
            if m.size == w_arr.size:
                expected_return = float(w_arr @ m)
        if cov is not None and w_arr.size:
            c = np.asarray(cov, dtype=np.float64)
            if c.ndim == 2 and c.shape[0] == w_arr.size:
                expected_variance = float(w_arr @ c @ w_arr)
    except Exception:  # noqa: BLE001
        pass

    if expected_return is None and raw.get("expected_return") is not None:
        expected_return = float(raw["expected_return"])
    if expected_variance is None and raw.get("expected_variance") is not None:
        expected_variance = float(raw["expected_variance"])
    if expected_variance is None and isinstance(raw.get("diagnostics"), dict):
        diag = raw["diagnostics"]
        if "variance" in diag:
            expected_variance = float(diag["variance"])
        elif "portfolio_variance" in diag:
            expected_variance = float(diag["portfolio_variance"])

    return OptimizationResult(
        success=success,
        weights=w,
        names=name_list,
        status=status if not fallback_used else ("fallback" if success else status),
        failure_reason=None if reason is None else str(reason),
        conflicting_constraints=list(raw.get("conflicting_constraints") or []),
        diagnostics=dict(raw.get("diagnostics") or {}),
        fallback_used=bool(fallback_used),
        fallback_kind=fallback_kind,
        objective_value=(
            float(raw["objective_value"]) if raw.get("objective_value") is not None else None
        ),
        expected_return=expected_return,
        expected_variance=expected_variance,
        method=str(raw.get("name") or raw.get("method") or method),
        data_version=data_version,
        model_version=model_version,
        seed=seed,
        params={"optimizer_method": raw.get("method")},
        audit={"raw_status": status},
    )


class PortfolioConstructionEngine:
    """Facade over estimators, optimizers, constraints, costs, and risk gates.

    Architectural rules:
    - Does **not** generate alpha — only expresses provided forecasts/signals.
    - On optimization failure: never silently relax hard constraints; apply
      configured fallback (current | min_variance | cash) with ``fallback_used``.
    - When ``require_risk_validation``: Risk Intelligence has final authority.
    - Point-in-time only — no future data.
    """

    def __init__(
        self,
        settings: PortfolioSettings | None = None,
        risk_engine: Any | None = None,
        risk_ensemble: Any | None = None,
    ) -> None:
        self.settings = settings or PortfolioSettings.default()
        self.risk_engine = risk_engine
        self.risk_ensemble = risk_ensemble
        self._constructor = PortfolioConstructor(self.settings)
        self._serializer = PortfolioSerializer()
        self._risk_init_attempted = False
        self._risk_skip_reason: str | None = None

        if self.settings.require_risk_validation and self.risk_engine is None and self.risk_ensemble is None:
            self._ensure_risk_engine()

    def _ensure_risk_engine(self) -> None:
        if self._risk_init_attempted:
            return
        self._risk_init_attempted = True
        if self.risk_engine is not None or self.risk_ensemble is not None:
            return
        try:
            from iqrp.app.risk import RiskIntelligenceEngine

            self.risk_engine = RiskIntelligenceEngine()
            logger.info("PortfolioConstructionEngine: constructed RiskIntelligenceEngine for validation")
        except Exception as exc:  # noqa: BLE001
            self._risk_skip_reason = f"RiskIntelligenceEngine unavailable: {exc}"
            logger.warning("require_risk_validation set but risk engine missing: %s", self._risk_skip_reason)

    # ------------------------------------------------------------------ optimize
    def optimize(
        self,
        *,
        mu: Any | None = None,
        cov: Any | None = None,
        returns: Any | None = None,
        method: str | None = None,
        current_weights: Any | None = None,
        names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Dispatch to ``optimization.*`` and return :class:`OptimizationResult`."""
        settings = self.settings
        method_key = str(method or settings.method).strip().lower()
        # aliases
        if method_key in ("risk_budget", "equal_risk"):
            method_key = "risk_parity"

        if cov is None and returns is not None:
            cov_out = self.covariance(returns=returns)
            cov = np.asarray(cov_out["matrix"], dtype=np.float64)
        if mu is None and returns is not None and method_key in (
            "mean_variance",
            "max_sharpe",
            "maximum_sharpe",
            "turnover_aware",
            "turnover",
            "black_litterman",
            "entropy",
        ):
            mu_out = self.expected_returns(returns=returns)
            mu = np.asarray(mu_out.get("mu") or mu_out.get("vector"), dtype=np.float64)

        fn = _OPTIMIZER_MAP.get(method_key)
        if fn is None:
            return OptimizationResult.failure(
                reason=f"Unknown optimization method '{method_key}'",
                names=names,
                method=method_key,
                data_version=settings.data_version,
                model_version=settings.model_version,
            )

        opt_kwargs: dict[str, Any] = {
            "long_only": bool(kwargs.get("long_only", settings.long_only)),
            "max_weight": float(kwargs.get("max_weight", settings.max_weight)),
            "risk_aversion": float(kwargs.get("risk_aversion", settings.risk_aversion)),
            "names": list(names) if names is not None else kwargs.get("names"),
            "current_weights": current_weights,
        }
        if "max_gross" in kwargs or settings.max_gross is not None:
            opt_kwargs["max_gross"] = float(kwargs.get("max_gross", settings.max_gross))
        if "min_weight" in kwargs:
            opt_kwargs["min_weight"] = kwargs["min_weight"]
        if "constraints" in kwargs:
            opt_kwargs["constraints"] = kwargs["constraints"]
        if "budget" in kwargs:
            opt_kwargs["budget"] = kwargs["budget"]

        # method-specific
        if method_key in ("min_cvar", "cvar"):
            opt_kwargs["alpha"] = float(
                kwargs.get("alpha", settings.objective.cvar_confidence)
            )
            if kwargs.get("scenarios") is not None:
                opt_kwargs["scenarios"] = kwargs["scenarios"]
            elif returns is not None:
                opt_kwargs["scenarios"] = returns
        if method_key in ("turnover_aware", "turnover"):
            opt_kwargs["turnover_penalty"] = float(
                kwargs.get("turnover_penalty", settings.objective.turnover_penalty or 0.01)
            )
            if kwargs.get("max_turnover") is not None or settings.max_turnover is not None:
                opt_kwargs["max_turnover"] = float(
                    kwargs.get("max_turnover", settings.max_turnover)
                )
        if method_key == "risk_parity" or method_key == "erc":
            opt_kwargs["method"] = "erc" if method_key == "erc" else kwargs.get("rp_method", "risk_parity")
        if method_key == "black_litterman":
            for k in ("P", "Q", "omega", "tau", "market_weights", "views"):
                if k in kwargs:
                    opt_kwargs[k] = kwargs[k]
            opt_kwargs.setdefault("tau", settings.expected_returns.bl_tau)
        if method_key == "drawdown" and returns is not None:
            opt_kwargs.setdefault("returns", returns)
        # pass through remaining safe extras
        for k, v in kwargs.items():
            if k not in opt_kwargs and k not in (
                "long_only",
                "max_weight",
                "max_gross",
                "risk_aversion",
                "names",
                "fallback",
            ):
                opt_kwargs[k] = v

        try:
            raw = fn(mu=mu, cov=cov, **opt_kwargs)
        except TypeError:
            # some optimizers may not accept all kwargs
            filtered = {
                k: v
                for k, v in opt_kwargs.items()
                if k
                in (
                    "long_only",
                    "max_weight",
                    "max_gross",
                    "min_weight",
                    "risk_aversion",
                    "names",
                    "current_weights",
                    "constraints",
                    "budget",
                    "scenarios",
                    "alpha",
                    "turnover_penalty",
                    "max_turnover",
                    "method",
                    "P",
                    "Q",
                    "omega",
                    "tau",
                    "market_weights",
                    "returns",
                )
            }
            try:
                raw = fn(mu=mu, cov=cov, **filtered)
            except Exception as exc:  # noqa: BLE001
                return self._apply_optimize_fallback(
                    reason=f"optimizer error: {exc}",
                    mu=mu,
                    cov=cov,
                    current_weights=current_weights,
                    names=names,
                    method=method_key,
                )
        except Exception as exc:  # noqa: BLE001
            return self._apply_optimize_fallback(
                reason=f"optimizer error: {exc}",
                mu=mu,
                cov=cov,
                current_weights=current_weights,
                names=names,
                method=method_key,
            )

        result = dict_to_optimization_result(
            raw if isinstance(raw, dict) else {"success": False, "weights": [], "failure_reason": "non-dict"},
            names=names,
            method=method_key,
            data_version=settings.data_version,
            model_version=settings.model_version,
            seed=settings.seed,
            mu=mu,
            cov=cov,
        )

        if not result.success:
            return self._apply_optimize_fallback(
                reason=result.failure_reason or "optimization failed",
                mu=mu,
                cov=cov,
                current_weights=current_weights,
                names=result.names or names,
                method=method_key,
                prior=result,
            )
        return result

    def _apply_optimize_fallback(
        self,
        *,
        reason: str,
        mu: Any | None,
        cov: Any | None,
        current_weights: Any | None,
        names: Sequence[str] | None,
        method: str,
        prior: OptimizationResult | None = None,
    ) -> OptimizationResult:
        """Apply configured fallback without silently relaxing hard constraints."""
        kind = str(self.settings.fallback)
        n = 0
        if cov is not None:
            c = np.asarray(cov, dtype=np.float64)
            n = int(c.shape[0]) if c.ndim == 2 else 0
        if n == 0 and current_weights is not None:
            n = int(np.asarray(current_weights).reshape(-1).size)
        if n == 0 and names is not None:
            n = len(names)
        name_list = list(names) if names is not None else [f"a{i}" for i in range(n)]

        reasons = [reason, f"fallback={kind}"]

        if kind == "current" and current_weights is not None:
            w = _extract_weights(current_weights, n or None)
            return OptimizationResult(
                success=True,
                weights=w,
                names=name_list[: len(w)] if name_list else [f"a{i}" for i in range(len(w))],
                status="fallback",
                failure_reason=reason,
                fallback_used=True,
                fallback_kind="current",
                method=method,
                data_version=self.settings.data_version,
                model_version=self.settings.model_version,
                seed=self.settings.seed,
                diagnostics={"fallback_reasons": reasons, "prior": prior.to_dict() if prior else None},
                audit={"fallback_reasons": reasons},
            )

        if kind == "min_variance" and cov is not None:
            try:
                raw = optimize_minimum_variance(
                    mu=mu,
                    cov=cov,
                    long_only=self.settings.long_only,
                    max_weight=self.settings.max_weight,
                    max_gross=self.settings.max_gross,
                    names=name_list,
                )
                res = dict_to_optimization_result(
                    raw,
                    names=name_list,
                    method="min_variance",
                    data_version=self.settings.data_version,
                    model_version=self.settings.model_version,
                    seed=self.settings.seed,
                    mu=mu,
                    cov=cov,
                    fallback_used=True,
                    fallback_kind="min_variance",
                )
                res.failure_reason = reason
                res.status = "fallback" if res.success else "failed"
                res.diagnostics = {**res.diagnostics, "fallback_reasons": reasons}
                res.audit = {**res.audit, "fallback_reasons": reasons}
                if res.success:
                    return res
                reasons.append(res.failure_reason or "min_variance fallback failed")
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"min_variance fallback error: {exc}")

        # cash fallback (also when current unavailable)
        w_cash = [0.0] * n
        return OptimizationResult(
            success=True,
            weights=w_cash,
            names=name_list if len(name_list) == n else [f"a{i}" for i in range(n)],
            status="fallback",
            failure_reason=reason,
            fallback_used=True,
            fallback_kind="cash",
            method=method,
            data_version=self.settings.data_version,
            model_version=self.settings.model_version,
            seed=self.settings.seed,
            diagnostics={"fallback_reasons": reasons},
            audit={"fallback_reasons": reasons, "cash": True},
            expected_return=0.0,
            expected_variance=0.0,
        )

    # ---------------------------------------------------------------- construct
    def construct(
        self,
        *,
        signals: Any | None = None,
        forecasts: Any | None = None,
        forecast_confidence: Any | None = None,
        returns: Any | None = None,
        current_portfolio: Any | None = None,
        capital: float | None = None,
        prices: Any | None = None,
        **kwargs: Any,
    ) -> PortfolioResult:
        """Express forecasts/signals into a constrained portfolio (no alpha gen)."""
        settings = self.settings
        method = str(kwargs.get("method", settings.method))
        names = kwargs.get("names")

        current_weights = None
        if current_portfolio is not None:
            if isinstance(current_portfolio, Portfolio):
                current_weights = current_portfolio.weight_array()
                if names is None:
                    names = list(current_portfolio.names)
            elif isinstance(current_portfolio, TargetWeights):
                current_weights = current_portfolio.as_array()
                if names is None:
                    names = list(current_portfolio.names)
            else:
                current_weights = np.asarray(current_portfolio, dtype=np.float64).reshape(-1)

        # Expected returns from forecasts (expression only) or returns history
        mu = kwargs.get("mu")
        mu_meta: dict[str, Any] = {}
        if mu is None and forecasts is not None:
            conf = forecast_confidence
            if conf is None and settings.expected_returns.confidence_shrink:
                conf = None  # forecast_expected_returns defaults to ones
            er = forecast_expected_returns(
                forecasts,
                confidence=conf if settings.expected_returns.confidence_shrink else np.ones(
                    np.asarray(forecasts).reshape(-1).size
                ),
                prior=kwargs.get("prior"),
                uncertainty=kwargs.get("uncertainty"),
                names=names,
            )
            mu = np.asarray(er["mu"], dtype=np.float64)
            mu_meta = er
        elif mu is None and signals is not None and forecasts is None:
            # signal path: optional direct signal→weight, else treat as forecast
            if kwargs.get("signal_method"):
                sig = signals_to_raw_weights(
                    signals,
                    method=str(kwargs["signal_method"]),
                    long_only=settings.long_only,
                    names=names,
                )
                # Still run through optimizer when cov available; else use signal weights
                mu = np.asarray(signals, dtype=np.float64).reshape(-1)
                mu_meta = {"signal_weights": sig}
            else:
                er = forecast_expected_returns(
                    signals,
                    confidence=forecast_confidence,
                    names=names,
                )
                mu = np.asarray(er["mu"], dtype=np.float64)
                mu_meta = er

        cov = kwargs.get("cov")
        cov_meta: dict[str, Any] = {}
        if cov is None and returns is not None:
            cov_meta = self.covariance(returns=returns, method=kwargs.get("cov_method"))
            cov = np.asarray(cov_meta["matrix"], dtype=np.float64)

        n = 0
        if cov is not None:
            n = int(np.asarray(cov).shape[0])
        elif mu is not None:
            n = int(np.asarray(mu).reshape(-1).size)
        elif returns is not None:
            n = int(np.asarray(returns).shape[1]) if np.asarray(returns).ndim == 2 else 1

        if names is None:
            names = [f"a{i}" for i in range(n)]

        opt_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "mu",
                "cov",
                "names",
                "method",
                "signal_method",
                "cov_method",
                "prior",
                "uncertainty",
                "include_transaction_costs",
                "adv",
                "spreads",
                "vols",
                "multipliers",
                "lot_sizes",
                "fx_rates",
                "factor_loadings",
                "scenarios",
            )
        }
        if kwargs.get("scenarios") is not None:
            opt_kwargs["scenarios"] = kwargs["scenarios"]

        if cov is None and signals is not None and kwargs.get("signal_method"):
            # unconstrained expression path when no risk model
            tw = self._constructor.signals_to_weights(
                signals, method=str(kwargs["signal_method"]), names=names
            )
            opt = OptimizationResult(
                success=True,
                weights=list(tw.weights),
                names=list(tw.names),
                status="ok",
                method=f"signal:{kwargs['signal_method']}",
                data_version=settings.data_version,
                model_version=settings.model_version,
                seed=settings.seed,
            )
        elif cov is None:
            opt = OptimizationResult.failure(
                reason="covariance required for optimization (provide cov or returns)",
                names=names,
                method=method,
                data_version=settings.data_version,
                model_version=settings.model_version,
            )
            opt = self._apply_optimize_fallback(
                reason=opt.failure_reason or "missing cov",
                mu=mu,
                cov=None,
                current_weights=current_weights,
                names=names,
                method=method,
                prior=opt,
            )
        else:
            opt = self.optimize(
                mu=mu,
                cov=cov,
                returns=returns,
                method=method,
                current_weights=current_weights,
                names=names,
                **opt_kwargs,
            )

        w = np.asarray(opt.weights, dtype=np.float64)
        name_list = list(opt.names)

        # Risk validation (final authority)
        risk_val: dict[str, Any] | None = None
        fallback_reasons: list[str] = []
        if opt.fallback_used:
            fallback_reasons.append(f"optimization_fallback:{opt.fallback_kind}")

        if settings.require_risk_validation:
            risk_val = self._run_risk_validation(
                weights=w,
                returns=returns,
                forecast_confidence=forecast_confidence,
            )
            action = str(
                (risk_val or {}).get("action")
                or (risk_val or {}).get("decision")
                or (risk_val or {}).get("status")
                or ""
            ).upper()
            rejected = action in ("REJECT", "REJECTED", "HALT", "BLOCK") or (
                risk_val is not None and risk_val.get("approved") is False
            )
            if rejected:
                fb = self._apply_optimize_fallback(
                    reason=f"risk_validation rejected: {risk_val}",
                    mu=mu,
                    cov=cov,
                    current_weights=current_weights,
                    names=name_list,
                    method=method,
                    prior=opt,
                )
                opt = fb
                w = np.asarray(opt.weights, dtype=np.float64)
                name_list = list(opt.names)
                fallback_reasons.append("risk_validation_reject")

        tw = build_target_weights(
            w,
            names=name_list,
            method=opt.method or method,
            source="construction",
            long_only=settings.long_only,
            meta={"optimization_status": opt.status},
        )

        cap = float(capital) if capital is not None else float(kwargs.get("nav", 1.0))
        px = prices if prices is not None else np.ones(len(w))
        positions = weights_to_positions(
            tw,
            capital=cap,
            prices=px,
            names=name_list,
            multipliers=kwargs.get("multipliers"),
            lot_sizes=kwargs.get("lot_sizes"),
            fx_rates=kwargs.get("fx_rates"),
        )

        # Metrics
        ereturn = opt.expected_return
        evar = opt.expected_variance
        if ereturn is None and mu is not None and w.size:
            m = np.asarray(mu, dtype=np.float64).reshape(-1)
            if m.size == w.size:
                ereturn = float(w @ m)
        if evar is None and cov is not None and w.size:
            c = np.asarray(cov, dtype=np.float64)
            if c.shape[0] == w.size:
                evar = float(w @ c @ w)
        evol = float(np.sqrt(max(evar, 0.0))) if evar is not None else None
        rf = float(settings.objective.risk_free_rate)
        esharpe = None
        if ereturn is not None and evol is not None and evol > 1e-12:
            esharpe = float((ereturn - rf) / evol)

        ecvar = None
        scenarios = kwargs.get("scenarios")
        if scenarios is not None and w.size:
            scen = np.asarray(scenarios, dtype=np.float64)
            if scen.ndim == 2 and scen.shape[1] == w.size:
                port_rets = scen @ w
                losses = -port_rets
                alpha = float(settings.objective.cvar_confidence)
                q = float(np.quantile(losses, alpha))
                tail = losses[losses >= q]
                ecvar = float(np.mean(tail)) if tail.size else float(q)

        # Drawdown proxy from returns @ weights (PIT historical)
        edd = None
        if returns is not None and w.size:
            R = np.asarray(returns, dtype=np.float64)
            if R.ndim == 2 and R.shape[1] == w.size:
                pr = R @ w
                wealth = np.cumprod(1.0 + pr)
                peak = np.maximum.accumulate(wealth)
                dd = (wealth - peak) / np.maximum(peak, 1e-12)
                edd = float(np.min(dd)) if dd.size else None

        rc: dict[str, Any] = {}
        if cov is not None and w.size:
            try:
                rc = pr_risk_contribution(w, cov)
            except Exception as exc:  # noqa: BLE001
                rc = {"error": str(exc)}

        to = 0.0
        if current_weights is not None:
            to = float(one_way_turnover(current_weights, w))

        tc: dict[str, Any] = {}
        if kwargs.get("include_transaction_costs", True) and current_weights is not None:
            tc = total_transaction_cost(
                current_weights,
                w,
                capital=cap,
                prices=px,
                adv=kwargs.get("adv"),
                spreads=kwargs.get("spreads"),
                vols=kwargs.get("vols"),
            )
        elif kwargs.get("include_transaction_costs", False):
            tc = total_transaction_cost(
                np.zeros_like(w),
                w,
                capital=cap,
                prices=px,
                adv=kwargs.get("adv"),
                spreads=kwargs.get("spreads"),
                vols=kwargs.get("vols"),
            )

        factor_exp: dict[str, Any] = {}
        if kwargs.get("factor_loadings") is not None:
            try:
                from iqrp.app.portfolio.constraints.factor import portfolio_factor_exposures

                factor_exp = portfolio_factor_exposures(
                    w, factor_loadings=kwargs["factor_loadings"], factor_names=kwargs.get("factor_names")
                )
            except Exception as exc:  # noqa: BLE001
                factor_exp = {"error": str(exc)}

        liq_exp: dict[str, Any] = {}
        if kwargs.get("adv") is not None:
            adv = np.asarray(kwargs["adv"], dtype=np.float64).reshape(-1)
            notionals = np.abs(w) * cap
            m = min(adv.size, notionals.size)
            part = np.zeros(notionals.size)
            if m:
                part[:m] = notionals[:m] / np.maximum(adv[:m], 1e-12)
            liq_exp = {
                "participation": part.tolist(),
                "max_participation": float(np.max(part)) if part.size else 0.0,
                "adv": adv.tolist(),
            }

        gross = float(np.sum(np.abs(w)))
        net = float(np.sum(w))

        constraints_audit = {
            "long_only": settings.long_only,
            "max_weight": settings.max_weight,
            "max_gross": settings.max_gross,
            "max_leverage": settings.max_leverage,
            "max_turnover": settings.max_turnover,
        }

        return PortfolioResult(
            portfolio_weights=tw,
            target_positions=positions,
            expected_return=ereturn,
            expected_volatility=evol,
            expected_sharpe=esharpe,
            expected_cvar=ecvar,
            expected_drawdown=edd,
            gross_exposure=gross,
            net_exposure=net,
            turnover=to,
            transaction_cost=tc,
            risk_contribution=rc,
            factor_exposure=factor_exp,
            liquidity_exposure=liq_exp,
            optimization=opt,
            fallback_used=bool(opt.fallback_used or bool(fallback_reasons)),
            fallback_kind=opt.fallback_kind,
            fallback_reasons=fallback_reasons,
            risk_validation=risk_val,
            method=opt.method or method,
            constraints=constraints_audit,
            timestamp=_utc_now(),
            data_version=settings.data_version,
            model_version=settings.model_version,
            seed=settings.seed,
            audit={
                "mu_meta": {k: v for k, v in mu_meta.items() if k not in ("mu", "vector", "forecasts", "prior")},
                "cov_method": cov_meta.get("method") or cov_meta.get("name"),
                "require_risk_validation": settings.require_risk_validation,
                "risk_skip_reason": self._risk_skip_reason,
                "note": "expresses provided forecasts/signals only; does not generate alpha",
            },
            names=name_list,
            weights=[float(x) for x in w.tolist()],
            success=bool(opt.success),
            status=opt.status,
        )

    def _run_risk_validation(
        self,
        *,
        weights: np.ndarray,
        returns: Any | None,
        forecast_confidence: Any | None,
    ) -> dict[str, Any]:
        if self.settings.require_risk_validation:
            self._ensure_risk_engine()

        conf_arr = None
        if forecast_confidence is not None:
            conf_arr = np.asarray(forecast_confidence, dtype=np.float64).reshape(-1)
        mean_conf = float(np.mean(conf_arr)) if conf_arr is not None and conf_arr.size else 0.0

        out: dict[str, Any] = {
            "timestamp": _utc_now(),
            "require_risk_validation": True,
        }

        if self.risk_engine is None and self.risk_ensemble is None:
            out["status"] = "skipped"
            out["action"] = "SKIP"
            out["approved"] = True  # cannot block without risk — logged
            out["reason"] = self._risk_skip_reason or "no risk engine available"
            logger.warning("Risk validation skipped: %s", out["reason"])
            return out

        # Portfolio-level limit check
        breaches: list[Any] = []
        if self.risk_engine is not None and hasattr(self.risk_engine, "check_limits"):
            try:
                breaches = list(self.risk_engine.check_limits(weights=weights) or [])
                out["breaches"] = [b.to_dict() if hasattr(b, "to_dict") else str(b) for b in breaches]
            except Exception as exc:  # noqa: BLE001
                out["check_limits_error"] = str(exc)

        # Pre-trade validate_position on max-weight name
        decision_payload: dict[str, Any] | None = None
        if returns is not None and weights.size:
            idx = int(np.argmax(np.abs(weights)))
            proposed = float(weights[idx])
            # validate with proposed already in weights
            r_in = returns
            R = np.asarray(returns, dtype=np.float64)
            if R.ndim == 2:
                # portfolio returns for drawdown state
                r_in = R @ weights if R.shape[1] == weights.size else R[:, 0]

            try:
                if self.risk_ensemble is not None and hasattr(self.risk_ensemble, "validate_position"):
                    dec = self.risk_ensemble.validate_position(
                        proposed_weight=proposed,
                        weights=weights,
                        returns=r_in,
                        forecast_confidence=mean_conf,
                        asset_index=idx,
                    )
                    decision_payload = dec.to_dict() if hasattr(dec, "to_dict") else dict(dec)
                elif self.risk_engine is not None and hasattr(self.risk_engine, "validate_position"):
                    dec = self.risk_engine.validate_position(
                        proposed_weight=proposed,
                        weights=weights,
                        returns=r_in,
                        forecast_confidence=mean_conf,
                        asset_index=idx,
                    )
                    decision_payload = dec.to_dict() if hasattr(dec, "to_dict") else dict(dec)
            except Exception as exc:  # noqa: BLE001
                out["validate_position_error"] = str(exc)

        if decision_payload is not None:
            out["decision"] = decision_payload
            action = str(
                decision_payload.get("action")
                or decision_payload.get("decision")
                or decision_payload.get("status")
                or ""
            ).upper()
            # RiskDecision uses approved bool; ensemble may use action enum
            if "approved" in decision_payload and decision_payload["approved"] is not None:
                approved = bool(decision_payload["approved"])
                if not action:
                    action = "APPROVE" if approved else "REJECT"
            else:
                approved = action not in ("REJECT", "REJECTED", "HALT", "BLOCK", "")
                if action == "":
                    approved = True
            out["action"] = action or ("APPROVE" if approved else "REJECT")
            out["approved"] = approved
            out["reason"] = decision_payload.get("reason")
            # Hard breaches from check_limits also reject
            hard = [
                b
                for b in breaches
                if getattr(b, "severity", None) is not None
                and str(getattr(b.severity, "value", b.severity)).lower() == "hard"
            ]
            if hard:
                out["action"] = "REJECT"
                out["approved"] = False
                out["reason"] = "hard limit breach(es) from check_limits"
        else:
            hard = [
                b
                for b in breaches
                if getattr(b, "severity", None) is not None
                and str(getattr(b.severity, "value", b.severity)).lower() == "hard"
            ]
            if hard:
                out["action"] = "REJECT"
                out["approved"] = False
                out["reason"] = "hard limit breach(es)"
            elif breaches:
                out["action"] = "CAUTION"
                out["approved"] = True
                out["reason"] = "soft breaches only"
            else:
                out["action"] = "APPROVE"
                out["approved"] = True
                out["reason"] = "limits clear" if returns is not None else "limits clear (no returns for position gate)"

        return out

    # ----------------------------------------------------------- target helpers
    def target_weights(self, *args: Any, **kwargs: Any) -> TargetWeights:
        if args and not kwargs.get("weights") and not kwargs.get("signals"):
            # target_weights(weights, names=...)
            return build_target_weights(
                args[0],
                names=kwargs.get("names"),
                method=str(kwargs.get("method", self.settings.method)),
                long_only=bool(kwargs.get("long_only", self.settings.long_only)),
                meta=kwargs.get("meta"),
            )
        if kwargs.get("signals") is not None:
            return self._constructor.signals_to_weights(
                kwargs["signals"],
                method=str(kwargs.get("method", "zscore")),
                names=kwargs.get("names"),
                long_only=kwargs.get("long_only", self.settings.long_only),
                temperature=kwargs.get("temperature", 1.0),
                budget=kwargs.get("budget", 1.0),
            )
        if kwargs.get("weights") is not None:
            return build_target_weights(
                kwargs["weights"],
                names=kwargs.get("names"),
                method=str(kwargs.get("method", self.settings.method)),
                long_only=bool(kwargs.get("long_only", self.settings.long_only)),
                meta=kwargs.get("meta"),
            )
        # construct from forecasts then extract
        result = self.construct(**kwargs)
        return result.portfolio_weights or TargetWeights.cash()

    def target_positions(self, *args: Any, **kwargs: Any) -> list[Position] | TargetPositions:
        as_list = bool(kwargs.pop("as_list", False))
        if args and kwargs.get("capital") is not None and kwargs.get("prices") is not None:
            tp = weights_to_positions(args[0], **kwargs)
            return list(tp.positions) if as_list else tp
        if kwargs.get("weights") is not None:
            tp = weights_to_positions(**kwargs)
            return list(tp.positions) if as_list else tp
        result = self.construct(**kwargs)
        pos = result.target_positions
        if pos is None:
            return TargetPositions()
        return pos

    # -------------------------------------------------------------- estimators
    def expected_returns(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        method = str(kwargs.pop("method", self.settings.expected_returns.method)).lower()
        returns = kwargs.pop("returns", args[0] if args else None)
        forecasts = kwargs.pop("forecasts", None)
        confidence = kwargs.pop("confidence", kwargs.pop("forecast_confidence", None))
        names = kwargs.pop("names", None)

        if method == "forecast" or forecasts is not None:
            if forecasts is None:
                if returns is None:
                    raise ValueError("forecasts or returns required for expected_returns")
                # historical as forecast proxy is not alpha gen — explicit historical path
                return historical_expected_returns(returns, names=names, **kwargs)
            return forecast_expected_returns(
                forecasts, confidence=confidence, names=names, **kwargs
            )
        if method == "historical":
            if returns is None:
                raise ValueError("returns required for historical expected returns")
            return historical_expected_returns(returns, names=names, **kwargs)
        if method == "shrinkage":
            if returns is None:
                raise ValueError("returns required for shrinkage expected returns")
            return shrinkage_expected_returns(returns, names=names, **kwargs)
        if method in ("black_litterman", "bl"):
            cov = kwargs.pop("cov", None)
            if cov is None and returns is not None:
                cov = np.asarray(self.covariance(returns=returns)["matrix"])
            return black_litterman_posterior(cov=cov, names=names, **kwargs)
        if returns is not None:
            return historical_expected_returns(returns, names=names, **kwargs)
        raise ValueError(f"Unknown expected return method '{method}'")

    def covariance(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        method = str(kwargs.pop("method", self.settings.covariance.method)).lower()
        returns = kwargs.pop("returns", args[0] if args else None)
        if returns is None:
            raise ValueError("returns required for covariance estimation")

        if method == "sample":
            return sample_covariance(returns, **kwargs)
        if method == "ewma":
            return ewma_covariance(
                returns,
                lambda_=float(
                    kwargs.pop("lambda_", kwargs.pop("lam", self.settings.covariance.ewma_lambda))
                ),
                **kwargs,
            )
        if method in ("shrinkage", "shrink"):
            intensity = kwargs.pop("intensity", self.settings.covariance.shrinkage_intensity)
            return shrinkage_covariance(returns, intensity=intensity, **kwargs)
        if method in ("ledoit_wolf", "lw"):
            return ledoit_wolf_covariance(returns, **kwargs)
        if method == "factor":
            kwargs.setdefault("asset_returns", returns)
            return factor_covariance(**kwargs)
        if method == "robust":
            return robust_covariance(returns, **kwargs)
        return shrinkage_covariance(returns, **kwargs)

    def risk_contribution(self, weights: Any, cov: Any) -> dict[str, Any]:
        return pr_risk_contribution(weights, cov)

    # ---------------------------------------------------------------- rebalance
    def rebalance(self, *args: Any, **kwargs: Any) -> RebalancePlan:
        if len(args) >= 2:
            current_weights, target_weights = args[0], args[1]
        else:
            current_weights = kwargs.pop("current_weights", kwargs.pop("current", None))
            target_weights = kwargs.pop("target_weights", kwargs.pop("target", None))
        if current_weights is None or target_weights is None:
            raise ValueError("rebalance requires current_weights and target_weights")
        bands = kwargs.pop("bands", None)
        if bands is None and any(k in kwargs for k in ("absolute_band", "relative_band", "min_trade")):
            bands = RebalanceBands(
                absolute=float(kwargs.pop("absolute_band", 0.0)),
                relative=float(kwargs.pop("relative_band", 0.0)),
                min_trade=float(kwargs.pop("min_trade", 0.0)),
            )
        return plan_rebalance(current_weights, target_weights, bands=bands, **kwargs)

    # ---------------------------------------------------------------- validate
    def validate(self, weights: Any, **constraints: Any) -> ValidationReport:
        """Check portfolio constraints and optional Risk Intelligence pre-trade."""
        settings = self.settings
        risk_flag = constraints.pop("risk_validation", None)
        if risk_flag is None:
            risk_flag = bool(settings.require_risk_validation)

        # merge defaults
        ckwargs = {
            "max_weight": constraints.pop("max_weight", settings.max_weight),
            "max_gross": constraints.pop("max_gross", settings.max_gross),
            "max_leverage": constraints.pop("max_leverage", settings.max_leverage),
            "long_only": constraints.pop("long_only", settings.long_only),
        }
        returns = constraints.pop("returns", None)
        forecast_confidence = constraints.pop("forecast_confidence", None)
        ckwargs.update(constraints)

        violations = check_all_constraints(weights, **ckwargs)
        hard = [v for v in violations if getattr(v, "hard", True)]
        soft = [v for v in violations if not getattr(v, "hard", True)]

        risk_decision = None
        risk_breaches: list[Any] = []
        messages: list[str] = []
        if risk_flag:
            risk_decision = self._run_risk_validation(
                weights=np.asarray(weights, dtype=np.float64).reshape(-1),
                returns=returns,
                forecast_confidence=forecast_confidence,
            )
            risk_breaches = list(risk_decision.get("breaches") or [])
            if risk_decision.get("approved") is False:
                messages.append(str(risk_decision.get("reason") or "risk rejected"))

        valid = len(hard) == 0 and (
            risk_decision is None or risk_decision.get("approved", True) is not False
        )
        return ValidationReport(
            valid=valid,
            violations=list(violations),
            hard_violations=hard,
            soft_violations=soft,
            risk_decision=risk_decision,
            risk_breaches=risk_breaches,
            messages=messages,
            meta={"constraints": {k: v for k, v in ckwargs.items() if not callable(v)}, "risk_validation": risk_flag},
        )

    # --------------------------------------------------------------- costs etc
    def transaction_cost(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) >= 2:
            w_old, w_new = args[0], args[1]
        else:
            w_old = kwargs.pop("weights_old", kwargs.pop("w_old", None))
            w_new = kwargs.pop("weights_new", kwargs.pop("w_new", None))
        if w_old is None or w_new is None:
            raise ValueError("transaction_cost requires weights_old and weights_new")
        return total_transaction_cost(w_old, w_new, **kwargs)

    def turnover(self, w_old: Any, w_new: Any) -> float | dict[str, Any]:
        to = float(one_way_turnover(w_old, w_new))
        return {"turnover": to, "one_way": True, "two_way": to * 2.0}

    def diagnostics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        weights = args[0] if args else kwargs.pop("weights", None)
        if weights is None:
            raise ValueError("diagnostics requires weights")
        return portfolio_diagnostics(
            weights,
            cov=kwargs.get("cov"),
            mu=kwargs.get("mu"),
            max_weight=kwargs.get("max_weight", self.settings.max_weight),
            max_gross=kwargs.get("max_gross", self.settings.max_gross),
            long_only=bool(kwargs.get("long_only", self.settings.long_only)),
        )

    def save(self, path: str | Path, obj: Any | None = None) -> Path:
        """Persist engine settings and optional last result / portfolio."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": self.settings.model_dump(),
            "object": None,
            "timestamp": _utc_now(),
            "data_version": self.settings.data_version,
            "model_version": self.settings.model_version,
        }
        if obj is not None:
            if hasattr(obj, "to_dict"):
                payload["object"] = obj.to_dict()
                payload["object_type"] = type(obj).__name__
            else:
                payload["object"] = obj
                payload["object_type"] = type(obj).__name__
        import json

        from iqrp.app.portfolio.serializer import _to_jsonable

        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> dict[str, Any]:
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "settings" in data:
            try:
                self.settings = PortfolioSettings.from_mapping(data["settings"])
            except Exception:  # noqa: BLE001
                pass
        return data
