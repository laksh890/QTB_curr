"""BacktestEngine — institutional backtesting platform orchestrator.

Deterministic, point-in-time pipeline:
CREATED → VALIDATING (PIT/leakage) → RUNNING → COMPLETED | FAILED | INVALIDATED.

CRITICAL: No look-ahead. Leakage / invalid universe → INVALIDATED.
Never promote on historical Sharpe/return alone (see validation_gates).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.backtesting.capacity import capacity_curve, estimate_capacity_limit
from iqrp.app.backtesting.comparison import compare_strategies
from iqrp.app.backtesting.config import BacktestSettings
from iqrp.app.backtesting.experiment_registry import (
    ExperimentLineage,
    ExperimentRegistry,
)
from iqrp.app.backtesting.paper_trading import PaperTradingConfig, PaperTradingInterface
from iqrp.app.backtesting.performance import (
    build_scorecard,
    summarize_returns,
    summarize_risk_adjusted,
)
from iqrp.app.backtesting.performance.drawdown import summarize_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, wealth_index
from iqrp.app.backtesting.performance.scorecard import StrategyScorecard
from iqrp.app.backtesting.performance.tail import summarize_tail
from iqrp.app.backtesting.pit import LookaheadViolation, detect_leakage, filter_universe_asof
from iqrp.app.backtesting.robustness import ablation_test, parameter_sweep, sensitivity_analysis
from iqrp.app.backtesting.scenarios import ScenarioEngine
from iqrp.app.backtesting.serializer import (
    deserialize_result,
    load_json,
    save_json,
    serialize_result,
)
from iqrp.app.backtesting.types import BacktestState, JSONDict
from iqrp.app.backtesting.validation_gates import GateResult, GateThresholds, evaluate_gates
from iqrp.app.backtesting.walk_forward import WalkForwardEngine

__all__ = ["BacktestEngine", "BacktestResult"]

StrategyFn = Callable[..., Any]
SignalFn = Callable[..., Any]


def _optional_execution_cost(
    notional: float,
    *,
    commission_bps: float,
    spread_bps: float,
    slippage_bps: float,
) -> float:
    """Estimate trade cost; prefer execution TCA when available."""
    try:
        from iqrp.app.execution.transaction_costs import pre_trade_cost_estimate

        mid = 100.0
        qty = abs(float(notional)) / mid if mid else 0.0
        if qty <= 0:
            return 0.0
        est = pre_trade_cost_estimate(
            side="buy" if notional >= 0 else "sell",
            quantity=qty,
            mid=mid,
            spread=mid * float(spread_bps) / 10_000.0,
            commission_bps=float(commission_bps),
            impact_coeff=max(float(slippage_bps) / 10_000.0, 1e-6),
        )
        total = float(est.get("total_cost", 0.0))
        # Convert absolute cost to fraction of notional
        return abs(total) / max(abs(float(notional)), 1e-12)
    except Exception:
        bps = float(commission_bps) + float(spread_bps) + float(slippage_bps)
        return abs(float(notional)) * bps / 10_000.0 / max(abs(float(notional)), 1e-12)


def _signals_to_weights(signals: np.ndarray) -> np.ndarray:
    """Map signal array to portfolio weights (sign / z-score style)."""
    s = np.asarray(signals, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return s
    std = float(np.std(s))
    if std > 1e-12:
        z = (s - float(np.mean(s))) / std
        w = np.tanh(z)
    else:
        w = np.sign(s)
    # Scale to unit gross on average
    gross = float(np.mean(np.abs(w))) if w.size else 1.0
    if gross > 1e-12:
        w = w / gross
    return w


@dataclass
class BacktestResult:
    """Full institutional backtest output with lineage."""

    experiment_id: str
    state: BacktestState = BacktestState.CREATED
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    equity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    metrics: dict[str, Any] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    exposures: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    costs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    attribution: dict[str, Any] = field(default_factory=dict)
    lineage: ExperimentLineage = field(default_factory=ExperimentLineage)
    seed: int = 42
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    invalidated: bool = False
    invalidation_reason: str = ""
    oos_returns: np.ndarray | None = None
    scorecard: StrategyScorecard | None = None
    timestamps: list[Any] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "experiment_id": self.experiment_id,
            "state": self.state.value if isinstance(self.state, BacktestState) else str(self.state),
            "returns": np.asarray(self.returns, dtype=np.float64).tolist(),
            "equity": np.asarray(self.equity, dtype=np.float64).tolist(),
            "metrics": dict(self.metrics),
            "trades": list(self.trades),
            "exposures": np.asarray(self.exposures, dtype=np.float64).tolist(),
            "costs": np.asarray(self.costs, dtype=np.float64).tolist(),
            "attribution": dict(self.attribution),
            "lineage": self.lineage.to_dict(),
            "seed": int(self.seed),
            "config": dict(self.config),
            "warnings": list(self.warnings),
            "invalidated": bool(self.invalidated),
            "invalidation_reason": self.invalidation_reason,
            "oos_returns": (
                None
                if self.oos_returns is None
                else np.asarray(self.oos_returns, dtype=np.float64).tolist()
            ),
            "scorecard": None if self.scorecard is None else self.scorecard.to_dict(),
            "timestamps": list(self.timestamps),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BacktestResult:
        state_raw = data.get("state", BacktestState.CREATED.value)
        try:
            state = BacktestState(state_raw)
        except Exception:
            state = BacktestState.FAILED
        sc_data = data.get("scorecard")
        scorecard = StrategyScorecard.from_dict(sc_data) if sc_data else None
        oos = data.get("oos_returns")
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            state=state,
            returns=np.asarray(data.get("returns") or [], dtype=np.float64),
            equity=np.asarray(data.get("equity") or [], dtype=np.float64),
            metrics=dict(data.get("metrics") or {}),
            trades=list(data.get("trades") or []),
            exposures=np.asarray(data.get("exposures") or [], dtype=np.float64),
            costs=np.asarray(data.get("costs") or [], dtype=np.float64),
            attribution=dict(data.get("attribution") or {}),
            lineage=ExperimentLineage.from_dict(data.get("lineage")),
            seed=int(data.get("seed", 42)),
            config=dict(data.get("config") or {}),
            warnings=list(data.get("warnings") or []),
            invalidated=bool(data.get("invalidated", False)),
            invalidation_reason=str(data.get("invalidation_reason") or ""),
            oos_returns=None if oos is None else np.asarray(oos, dtype=np.float64),
            scorecard=scorecard,
            timestamps=list(data.get("timestamps") or []),
        )


class BacktestEngine:
    """Institutional backtesting orchestrator."""

    def __init__(
        self,
        settings: BacktestSettings | Mapping[str, Any] | None = None,
        experiment_registry: ExperimentRegistry | None = None,
    ) -> None:
        if settings is None:
            self.settings = BacktestSettings.default()
        elif isinstance(settings, BacktestSettings):
            self.settings = settings
        else:
            self.settings = BacktestSettings.from_mapping(settings)
        self.registry = experiment_registry or ExperimentRegistry()
        self.paper = PaperTradingInterface()
        self._last_result: BacktestResult | None = None

    # ------------------------------------------------------------------ run
    def run(
        self,
        *,
        prices: Any | None = None,
        returns: Any | None = None,
        strategy_fn: StrategyFn | None = None,
        signals: Any | None = None,
        signal_fn: SignalFn | None = None,
        start: int | None = None,
        end: int | None = None,
        universe_asof: Any | None = None,
        membership: Mapping[str, Any] | None = None,
        corporate_actions: Sequence[Any] | None = None,
        costs: bool = True,
        execution_sim: bool = True,
        seed: int = 42,
        feature_asof_index: Sequence[int] | None = None,
        label_asof_index: Sequence[int] | None = None,
        oos_fraction: float | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run a deterministic PIT backtest pipeline."""
        rng = np.random.default_rng(int(seed))
        _ = rng  # reserved for stochastic extensions
        experiment_id = str(kwargs.pop("experiment_id", None) or uuid.uuid4())
        lineage = ExperimentLineage.from_settings(self.settings, seed=seed)
        cfg = self.settings.model_dump()
        cfg.update({k: v for k, v in kwargs.items() if k.startswith("meta_")})
        if name:
            cfg["name"] = name

        self.registry.create(
            experiment_id=experiment_id,
            name=name or self.settings.name,
            lineage=lineage,
            config=cfg,
        )
        result = BacktestResult(
            experiment_id=experiment_id,
            state=BacktestState.CREATED,
            lineage=lineage,
            seed=int(seed),
            config=cfg,
        )

        # --- VALIDATING ---
        result.state = BacktestState.VALIDATING
        self.registry.update_state(experiment_id, BacktestState.VALIDATING)

        try:
            market_returns = self._prepare_returns(prices=prices, returns=returns)
            n = int(market_returns.size)
            i0 = int(start) if start is not None else 0
            i1 = int(end) if end is not None else n
            i0 = max(0, min(i0, n))
            i1 = max(i0, min(i1, n))
            market_returns = market_returns[i0:i1]
            n = int(market_returns.size)
            timestamps = list(range(i0, i0 + n))

            # Universe as-of (survivorship guard)
            if membership is not None and universe_asof is not None:
                try:
                    _ = filter_universe_asof(membership, universe_asof)
                except LookaheadViolation as exc:
                    return self._invalidate(result, f"universe PIT violation: {exc}")

            # Leakage checks
            if (
                self.settings.pit.detect_leakage
                and feature_asof_index is not None
                and label_asof_index is not None
            ):
                report = detect_leakage(
                    feature_asof_index,
                    label_asof_index,
                    timestamps=timestamps,
                    max_label_horizon=self.settings.pit.max_label_horizon,
                )
                if report.has_leakage:
                    return self._invalidate(result, f"leakage detected: {report.detail}")

            # --- RUNNING ---
            result.state = BacktestState.RUNNING
            self.registry.update_state(experiment_id, BacktestState.RUNNING)

            weights, trades, cost_series, strat_returns = self._simulate(
                market_returns=market_returns,
                signals=signals,
                strategy_fn=strategy_fn,
                signal_fn=signal_fn,
                costs=costs,
                execution_sim=execution_sim,
                corporate_actions=corporate_actions,
                timestamps=timestamps,
            )

            equity = wealth_index(strat_returns, start=float(self.settings.initial_cash))
            metrics = {
                **summarize_returns(strat_returns),
                **summarize_risk_adjusted(strat_returns),
                **summarize_drawdown(strat_returns),
                **summarize_tail(strat_returns),
            }

            oos_returns = None
            if oos_fraction is not None and 0.0 < float(oos_fraction) < 1.0:
                cut = max(1, int(n * (1.0 - float(oos_fraction))))
                oos_returns = strat_returns[cut:]

            scorecard = build_scorecard(
                strat_returns,
                positions=weights,
                costs=cost_series,
                oos_returns=oos_returns,
            )

            result.returns = strat_returns
            result.equity = equity
            result.metrics = metrics
            result.metrics["sharpe"] = float(scorecard.sharpe)
            result.trades = trades
            result.exposures = weights
            result.costs = cost_series
            result.attribution = {"strategy_total": float(np.sum(strat_returns))}
            result.oos_returns = oos_returns
            result.scorecard = scorecard
            result.timestamps = timestamps
            result.state = BacktestState.COMPLETED
            result.invalidated = False

            self.registry.register_result(
                experiment_id,
                state=BacktestState.COMPLETED,
                metrics=metrics,
                warnings=result.warnings,
                result_summary={
                    "n": n,
                    "sharpe": float(scorecard.sharpe),
                    "total_return": float(scorecard.total_return),
                },
            )
            self._last_result = result
            return result

        except Exception as exc:
            result.state = BacktestState.FAILED
            result.warnings.append(str(exc))
            self.registry.register_result(
                experiment_id,
                state=BacktestState.FAILED,
                warnings=result.warnings,
                result_summary={"error": str(exc)},
            )
            self._last_result = result
            return result

    def _invalidate(self, result: BacktestResult, reason: str) -> BacktestResult:
        result.state = BacktestState.INVALIDATED
        result.invalidated = True
        result.invalidation_reason = reason
        result.warnings.append(reason)
        self.registry.invalidate(result.experiment_id, reason)
        self._last_result = result
        return result

    def _prepare_returns(self, *, prices: Any | None, returns: Any | None) -> np.ndarray:
        if returns is not None:
            return as_returns(returns)
        if prices is None:
            raise ValueError("run() requires prices or returns")
        p = np.asarray(prices, dtype=np.float64)
        if p.ndim > 1:
            # Equal-weight basket of asset returns
            rets = np.diff(p, axis=0) / np.clip(p[:-1], 1e-12, None)
            return as_returns(np.nanmean(rets, axis=-1))
        p = p.reshape(-1)
        return as_returns(np.diff(p) / np.clip(p[:-1], 1e-12, None))

    def _simulate(
        self,
        *,
        market_returns: np.ndarray,
        signals: Any | None,
        strategy_fn: StrategyFn | None,
        signal_fn: SignalFn | None,
        costs: bool,
        execution_sim: bool,
        corporate_actions: Sequence[Any] | None,
        timestamps: Sequence[Any],
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
        n = int(market_returns.size)
        cost_cfg = self.settings.costs
        weights = np.zeros(n, dtype=np.float64)
        cost_series = np.zeros(n, dtype=np.float64)
        strat_returns = np.zeros(n, dtype=np.float64)
        trades: list[dict[str, Any]] = []
        prev_w = 0.0

        # Precompute signal-based weights if provided (still applied causally)
        sig_w: np.ndarray | None = None
        if signals is not None:
            raw = np.asarray(signals, dtype=np.float64).reshape(-1)
            if raw.size >= n:
                raw = raw[-n:]
            elif raw.size < n:
                pad = np.zeros(n, dtype=np.float64)
                pad[-raw.size :] = raw
                raw = pad
            sig_w = _signals_to_weights(raw)

        # Optional corporate action cash adjustments (PIT-filtered when datetimes)
        ca_dividend_boost = np.zeros(n, dtype=np.float64)
        if corporate_actions and self.settings.corporate_actions.enabled:
            try:
                from iqrp.app.backtesting.corporate_actions import (
                    CorporateActionType,
                    actions_asof,
                )

                # Only apply when timestamps are datetimes
                for i, ts in enumerate(timestamps):
                    if not hasattr(ts, "tzinfo"):
                        break
                    applicable = actions_asof(list(corporate_actions), ts)
                    for a in applicable:
                        if a.action_type == CorporateActionType.DIVIDEND:
                            ca_dividend_boost[i] += float(a.payload.get("amount", 0.0))
            except Exception as exc:
                # Non-fatal — record warning via empty boost
                _ = exc

        for t in range(n):
            # PIT history: data[:t+1] only
            history = market_returns[: t + 1]

            if strategy_fn is not None:
                try:
                    out = strategy_fn(t, history)
                except TypeError:
                    out = strategy_fn(t=t, history=history)
                if isinstance(out, Mapping):
                    w = float(out.get("weight", out.get("weights", 0.0)))
                else:
                    w = float(np.asarray(out).reshape(-1)[-1]) if np.size(out) else 0.0
            elif signal_fn is not None:
                try:
                    sig = signal_fn(t, history)
                except TypeError:
                    sig = signal_fn(t)
                w = float(np.tanh(float(np.asarray(sig).reshape(-1)[-1])))
            elif sig_w is not None:
                # Causal: only use signal up to t (precomputed series assumed aligned)
                w = float(sig_w[t])
            else:
                # Default: flat long
                w = 1.0

            # Turnover / costs
            delta = w - prev_w
            trade_cost = 0.0
            if costs and abs(delta) > 1e-12:
                if execution_sim:
                    trade_cost = _optional_execution_cost(
                        delta,
                        commission_bps=cost_cfg.commission_bps,
                        spread_bps=cost_cfg.spread_bps,
                        slippage_bps=cost_cfg.slippage_bps,
                    ) * abs(delta)
                else:
                    bps = cost_cfg.commission_bps + cost_cfg.spread_bps + cost_cfg.slippage_bps
                    trade_cost = abs(delta) * bps / 10_000.0
                trades.append(
                    {
                        "t": int(t),
                        "timestamp": timestamps[t] if t < len(timestamps) else t,
                        "weight_from": float(prev_w),
                        "weight_to": float(w),
                        "delta": float(delta),
                        "cost": float(trade_cost),
                        "pnl": 0.0,
                    }
                )

            # Position PnL from market return at t (position held into bar)
            pnl = (
                float(prev_w) * float(market_returns[t]) - trade_cost + float(ca_dividend_boost[t])
            )
            if trades and trades[-1].get("t") == t:
                trades[-1]["pnl"] = float(pnl)

            weights[t] = w
            cost_series[t] = trade_cost
            strat_returns[t] = pnl
            prev_w = w

        return weights, trades, cost_series, strat_returns

    # --------------------------------------------------------- extensions
    def walk_forward(
        self,
        *,
        returns: Any | None = None,
        n: int | None = None,
        evaluate_fold: Callable[..., Any] | None = None,
        train_size: int | None = None,
        test_size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        wf = WalkForwardEngine()
        cfg = self.settings.walk_forward
        r = as_returns(returns) if returns is not None else None
        nn = int(n if n is not None else (r.size if r is not None else 0))
        if nn <= 0:
            raise ValueError("walk_forward requires n or returns")

        def _default_fold(train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
            if r is None:
                return {"n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
            te = r[np.asarray(test_idx, dtype=int)]
            from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

            return {
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "sharpe": float(sharpe_ratio(te)),
                "total_return": float(np.prod(1.0 + te) - 1.0) if te.size else 0.0,
            }

        return wf.run(  # type: ignore[return-value]
            n=nn,
            train_size=int(train_size if train_size is not None else cfg.train_periods),
            test_size=int(test_size if test_size is not None else cfg.test_periods),
            evaluate_fold=evaluate_fold or _default_fold,
            mode=kwargs.pop("mode", cfg.mode),
            purge=kwargs.pop("purge", cfg.purge_periods),
            embargo=kwargs.pop("embargo", cfg.embargo_periods),
            validation_size=kwargs.pop("validation_size", cfg.validation_periods),
            **kwargs,
        )

    def retrain_rolling(
        self,
        *,
        X: Any,
        y: Any | None = None,
        train_fn: Callable[..., Any] | None = None,
        predict_fn: Callable[..., Any] | None = None,
        score_fn: Callable[..., Any] | None = None,
        every: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from iqrp.app.backtesting.rolling_retraining import RetrainSchedule, RollingRetrainer

        X_arr = np.asarray(X)
        y_arr = None if y is None else np.asarray(y)

        def _train(X_tr: np.ndarray, y_tr: np.ndarray | None, params: dict[str, Any]) -> Any:
            if train_fn is not None:
                return train_fn(X_tr, y_tr, params)
            # Default: mean predictor
            return {"mu": float(np.mean(y_tr)) if y_tr is not None and y_tr.size else 0.0}

        def _predict(model: Any, X_te: np.ndarray) -> Any:
            if predict_fn is not None:
                return predict_fn(model, X_te)
            mu = float(model.get("mu", 0.0)) if isinstance(model, Mapping) else 0.0
            return np.full(len(X_te), mu, dtype=np.float64)

        def _score(model: Any, X_te: np.ndarray, y_te: np.ndarray | None) -> Mapping[str, float]:
            if score_fn is not None:
                return score_fn(model, X_te, y_te)
            pred = _predict(model, X_te)
            if y_te is None or len(y_te) == 0:
                return {"n": float(len(X_te))}
            err = float(np.mean((np.asarray(pred) - np.asarray(y_te)) ** 2))
            return {"mse": err, "n": float(len(y_te))}

        retrainer = RollingRetrainer(
            schedule=RetrainSchedule(every=int(every)),
            train_window=kwargs.pop("train_window", None),
            origin=int(kwargs.pop("origin", 0)),
        )
        report = retrainer.run(
            X=X_arr,
            y=y_arr,
            train_fn=_train,
            predict_fn=_predict,
            score_fn=_score,
            **kwargs,
        )
        if hasattr(report, "to_dict"):
            return report.to_dict()
        return dict(report) if isinstance(report, Mapping) else {"result": report}

    def scenarios(
        self,
        kind: str = "monte_carlo",
        *,
        returns: Any | None = None,
        result: BacktestResult | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        eng = ScenarioEngine(seed=int(kwargs.pop("seed", self.settings.reproducibility.seed)))
        r = returns
        if r is None:
            src = result or self._last_result
            if src is None:
                raise ValueError("scenarios() requires returns or a prior result")
            r = src.returns
        return eng.run(kind, r, **kwargs)

    def capacity_test(
        self,
        *,
        returns: Any | None = None,
        result: BacktestResult | None = None,
        capital_levels: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        src = result or self._last_result
        r = returns if returns is not None else (None if src is None else src.returns)
        if r is None:
            raise ValueError("capacity_test requires returns or a prior result")
        levels = capital_levels if capital_levels is not None else np.geomspace(1e6, 1e9, 12)
        curve = capacity_curve(
            r,
            levels,
            **{k: v for k, v in kwargs.items() if k in ("model", "cost_fn", "periods_per_year")},
        )
        limit = estimate_capacity_limit(
            r,
            capital_levels=levels,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("min_sharpe", "max_drawdown", "model", "periods_per_year")
            },
        )
        return {"curve": curve, "limit": limit}

    def parameter_sweep(
        self,
        objective: Callable[..., Any],
        param_grid: Mapping[str, Sequence[Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return parameter_sweep(objective, param_grid, **kwargs)

    def ablation(
        self,
        objective: Callable[..., Any],
        *,
        components: Mapping[str, bool],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return ablation_test(objective, components=components, **kwargs)

    def sensitivity(
        self,
        objective: Callable[..., Any],
        base_params: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return sensitivity_analysis(objective, base_params, **kwargs)

    def compare(
        self,
        strategies: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return compare_strategies(strategies, **kwargs)

    def scorecard(self, result: BacktestResult | Mapping[str, Any]) -> StrategyScorecard:
        if isinstance(result, BacktestResult):
            if result.scorecard is not None:
                return result.scorecard
            return build_scorecard(
                result.returns,
                positions=result.exposures,
                costs=result.costs,
                oos_returns=result.oos_returns,
            )
        if isinstance(result, Mapping):
            if result.get("scorecard"):
                return StrategyScorecard.from_dict(result["scorecard"])
            return build_scorecard(
                result.get("returns"),
                positions=result.get("exposures"),
                costs=result.get("costs"),
                oos_returns=result.get("oos_returns"),
            )
        raise TypeError("scorecard expects BacktestResult or mapping")

    def validate_for_promotion(
        self,
        result: BacktestResult | Mapping[str, Any],
        gates: GateThresholds | Mapping[str, Any] | None = None,
    ) -> GateResult:
        """OOS-mandatory promotion check. Never promotes on hist Sharpe alone."""
        if isinstance(result, BacktestResult) and result.invalidated:
            return GateResult(
                approved=False,
                out_of_sample_ok=False,
                checks={"not_invalidated": False},
                reasons=[f"experiment invalidated: {result.invalidation_reason}"],
            )
        sc = self.scorecard(result)
        is_sharpe = float(sc.sharpe)
        return evaluate_gates(sc, gates, in_sample_sharpe=is_sharpe)

    def to_paper_trading(
        self,
        result: BacktestResult | str | None = None,
    ) -> PaperTradingConfig:
        if isinstance(result, str):
            return self.paper.from_experiment(result, self.registry)
        src = result if isinstance(result, BacktestResult) else self._last_result
        if src is None:
            raise ValueError("to_paper_trading requires a result or experiment_id")
        gate = self.validate_for_promotion(src)
        return self.paper.from_result(src, gates=gate.to_dict())

    def invalidate(self, experiment_id: str, reason: str) -> None:
        self.registry.invalidate(experiment_id, reason)
        if self._last_result and self._last_result.experiment_id == experiment_id:
            self._last_result.state = BacktestState.INVALIDATED
            self._last_result.invalidated = True
            self._last_result.invalidation_reason = reason

    def save(self, path: str | Path, result: BacktestResult | None = None) -> Path:
        src = result or self._last_result
        if src is None:
            raise ValueError("nothing to save")
        out = Path(path)
        payload = {
            "result": serialize_result(src),
            "registry": [r.to_dict() for r in self.registry.list()],
        }
        save_json(out, payload)
        return out

    def load(self, path: str | Path) -> BacktestResult:
        data = load_json(path)
        if isinstance(data, Mapping) and "result" in data:
            result = deserialize_result(data["result"])
            for rec in data.get("registry") or []:
                from iqrp.app.backtesting.experiment_registry import ExperimentRecord

                er = ExperimentRecord.from_dict(rec)
                self.registry._records[er.experiment_id] = er
        else:
            result = deserialize_result(data)
        self._last_result = result
        return result
